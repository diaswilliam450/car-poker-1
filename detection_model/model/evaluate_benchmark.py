from __future__ import annotations

import argparse
import time
import csv
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .dataset import ChunkSample, augment_chunk_windows, load_public_benchmark
from .inference import Poker44BotDetector
from .scoring import reward_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Poker44 model on public benchmark chunks."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to public_miner_benchmark.json.gz",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved PyTorch .pt model artifact.",
    )

    parser.add_argument(
        "--xgb-model",
        default=None,
        help="Optional path to saved XGBoost .joblib model.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="all",
        help="Which benchmark split to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )

    # NOTE: there is intentionally no --threshold flag. The validator classifies
    # with a hard np.round(score) (boundary 0.5) and the embedded calibrator is
    # what positions the scores against it, so every metric here is reported at
    # that exact boundary. A custom threshold would only diverge from how you are
    # actually scored on-chain.

    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )

    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to save per-chunk predictions as CSV.",
    )

    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to save metrics as JSON.",
    )

    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of per-chunk rows to print.",
    )

    # ------------------------------------------------------------------
    # Evaluation modes
    # ------------------------------------------------------------------

    parser.add_argument(
        "--augment-windows",
        action="store_true",
        help=(
            "Evaluate generated fixed-length windows as separate samples. "
            "Example: [1,2,3,4], [2,3,4,5], ..."
        ),
    )

    parser.add_argument(
        "--window-inference",
        action="store_true",
        help=(
            "Evaluate original chunks by splitting each original chunk into windows, "
            "scoring each window, then aggregating back to one score per original chunk. "
            "This mirrors miner.py window inference."
        ),
    )

    parser.add_argument(
        "--window-hands",
        type=int,
        default=20,
        help="Number of consecutive hands per sliding window (match training / miner).",
    )

    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help="Sliding window stride (match training / miner).",
    )

    parser.add_argument(
        "--window-agg",
        choices=["mean", "max", "topk_mean"],
        default="mean",
        help="How to aggregate window scores back to original chunk score.",
    )

    parser.add_argument(
        "--keep-short-window-chunks",
        action="store_true",
        help="Keep chunks shorter than --window-hands instead of dropping them.",
    )

    return parser.parse_args()


def load_detector(
    model_path: str,
    device: str,
    xgb_model_path: Optional[str],
) -> Poker44BotDetector:
    """
    Supports both old and new Poker44BotDetector.load() signatures.

    Expected signature:
        load(path, device=..., xgb_path=...)

    Old signature:
        load(path, device=...)
    """

    load_fn = Poker44BotDetector.load
    sig = inspect.signature(load_fn)

    kwargs: Dict[str, Any] = {
        "device": device,
    }

    if xgb_model_path and "xgb_path" in sig.parameters:
        kwargs["xgb_path"] = xgb_model_path

    detector = load_fn(model_path, **kwargs)

    if xgb_model_path and not getattr(detector, "xgb_model", None):
        try:
            import joblib

            payload = joblib.load(xgb_model_path)
            detector.xgb_model = payload["xgb_model"]
            print(f"Loaded XGBoost model manually from: {xgb_model_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not load XGBoost model from {xgb_model_path}: {exc}"
            ) from exc

    return detector


def select_samples(
    train_samples: List[ChunkSample],
    val_samples: List[ChunkSample],
    split: str,
) -> List[ChunkSample]:
    if split == "train":
        return train_samples

    if split == "val":
        return val_samples

    return train_samples + val_samples


# ---------------------------------------------------------------------------
# Data loading that also accepts the raw validator payload (data/chunks.json)
# ---------------------------------------------------------------------------

def _load_raw_json(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_hand(obj: Any) -> bool:
    return isinstance(obj, dict) and "actions" in obj and "players" in obj


def _is_labeled_structure(raw: Any) -> bool:
    """True when the file carries is_bot/label info (use the benchmark loader).

    The raw validator payload (``data/chunks.json``) is a bare
    ``List[List[hand]]`` with no labels, so this returns False and we score it
    directly instead of trying to compute label-dependent reward metrics.
    """
    if isinstance(raw, dict):
        for key in ("train", "training", "validation", "valid", "val", "dev",
                    "test", "labeled_chunks", "splits"):
            if raw.get(key):
                return True
        for key in ("samples", "data", "chunks", "items", "records"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            return False
    if not isinstance(raw, list) or not raw:
        return False
    head = raw[0]
    if isinstance(head, list):
        return False  # bare chunk = list of hands -> unlabeled
    if isinstance(head, dict):
        return any(
            isinstance(it, dict) and ("is_bot" in it or "label" in it)
            for it in raw[:50]
        )
    return False


def load_unlabeled_chunks(raw: Any) -> List[ChunkSample]:
    """Build label-less samples from a raw validator-style payload.

    Accepts the bare ``List[List[hand]]`` (data/chunks.json) format, a wrapper
    dict around such a list, and label-less ``[{"hands": [...]}, ...]`` items.
    """
    if isinstance(raw, dict):
        for key in ("chunks", "labeled_chunks", "samples", "data", "items", "records"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Unsupported chunks structure: {type(raw)}")

    samples: List[ChunkSample] = []
    for idx, item in enumerate(raw):
        if isinstance(item, list):
            chunk = item
        elif isinstance(item, dict):
            chunk = item.get("hands") or item.get("chunk")
            if chunk is None and _looks_like_hand(item):
                chunk = [item]
        else:
            continue
        if not chunk:
            continue
        chunk_id = (item.get("chunk_id") if isinstance(item, dict) else None) or f"chunk_{idx}"
        samples.append(ChunkSample(chunk=chunk, label=-1, chunk_id=str(chunk_id)))
    return samples


def load_eval_samples(path: str | Path, split: str) -> Tuple[List[ChunkSample], bool]:
    """Return ``(samples, labeled)``.

    Labeled benchmarks go through the normal train/val split logic; unlabeled
    validator payloads (data/chunks.json) are scored as-is with ``label = -1``.
    """
    raw = _load_raw_json(path)
    if _is_labeled_structure(raw):
        train_samples, val_samples = load_public_benchmark(path)
        return select_samples(train_samples, val_samples, split), True
    return load_unlabeled_chunks(raw), False


def scoring_only_metrics(scores: np.ndarray) -> Dict[str, Any]:
    """Diagnostics for unlabeled data: distribution + predicted-class split at
    the validator's 0.5 boundary. No reward/AP/FPR — those need labels."""
    n = int(len(scores))
    preds = np.round(scores).astype(np.int32)
    pred_bot = int(preds.sum())
    return {
        "count": n,
        "labeled": False,
        "boundary": "validator np.round (0.5)",
        "score_min": float(scores.min()) if n else 0.0,
        "score_max": float(scores.max()) if n else 0.0,
        "score_mean": float(scores.mean()) if n else 0.0,
        "score_std": float(scores.std()) if n else 0.0,
        "pred_bot": pred_bot,
        "pred_human": n - pred_bot,
        "pred_bot_rate": float(pred_bot / n) if n else 0.0,
    }


def make_windows_for_chunk(
    chunk: List[Dict[str, Any]],
    window_hands: int,
    window_stride: int,
    keep_short: bool,
) -> List[List[Dict[str, Any]]]:
    if not chunk:
        return []

    n = len(chunk)

    if n < window_hands:
        return [chunk] if keep_short else []

    windows: List[List[Dict[str, Any]]] = []

    for start in range(0, n - window_hands + 1, window_stride):
        end = start + window_hands
        windows.append(chunk[start:end])

    return windows


def aggregate_scores(scores: List[float], method: str) -> float:
    if not scores:
        return 0.5

    scores = [float(s) for s in scores]

    if method == "max":
        value = max(scores)

    elif method == "topk_mean":
        k = min(3, len(scores))
        top_scores = sorted(scores, reverse=True)[:k]
        value = sum(top_scores) / len(top_scores)

    else:
        value = sum(scores) / len(scores)

    return round(max(0.0, min(1.0, value)), 6)


def predict_original_chunks_with_windows(
    detector: Poker44BotDetector,
    samples: List[ChunkSample],
    batch_size: int,
    window_hands: int,
    window_stride: int,
    window_agg: str,
    keep_short: bool,
) -> Tuple[List[float], List[int]]:
    """
    Same idea as miner.py:

        original chunk
          -> split into windows
          -> score all windows
          -> aggregate windows
          -> one score per original chunk

    Returns:
        final_scores: one score per original sample
        window_counts: how many windows were generated per original sample
    """

    all_windows: List[List[Dict[str, Any]]] = []
    ranges: List[Tuple[int, int]] = []
    window_counts: List[int] = []

    cursor = 0

    for sample in samples:
        windows = make_windows_for_chunk(
            chunk=sample.chunk,
            window_hands=window_hands,
            window_stride=window_stride,
            keep_short=keep_short,
        )

        if not windows:
            # Keep output shape valid.
            windows = [sample.chunk]

        start = cursor
        all_windows.extend(windows)
        cursor += len(windows)
        end = cursor

        ranges.append((start, end))
        window_counts.append(len(windows))

    if not all_windows:
        return [0.5 for _ in samples], [0 for _ in samples]

    window_scores = detector.predict_chunks(
        all_windows,
        batch_size=batch_size,
    )

    final_scores: List[float] = []

    for start, end in ranges:
        final_scores.append(
            aggregate_scores(
                scores=window_scores[start:end],
                method=window_agg,
            )
        )

    return final_scores, window_counts


def safe_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    # Classify at the validator's EXACT boundary so every metric below
    # reconciles with the validator_* block. The validator does
    # `preds = np.round(scores)` (0.5 rounds DOWN to human), nothing else.
    # Using `scores >= threshold` with any other threshold is what made the
    # confusion matrix disagree with validator_fpr.
    preds = np.round(scores).astype(np.int32)

    metrics: Dict[str, Any] = {}

    metrics["count"] = int(len(labels))
    metrics["boundary"] = "validator np.round (0.5)"

    metrics["human_count"] = int((labels == 0).sum())
    metrics["bot_count"] = int((labels == 1).sum())

    metrics["score_min"] = float(scores.min()) if len(scores) else 0.0
    metrics["score_max"] = float(scores.max()) if len(scores) else 0.0
    metrics["score_mean"] = float(scores.mean()) if len(scores) else 0.0
    metrics["score_std"] = float(scores.std()) if len(scores) else 0.0

    human_scores = scores[labels == 0]
    bot_scores = scores[labels == 1]

    metrics["human_score_mean"] = float(human_scores.mean()) if len(human_scores) else None
    metrics["bot_score_mean"] = float(bot_scores.mean()) if len(bot_scores) else None

    metrics["accuracy"] = float(accuracy_score(labels, preds)) if len(labels) else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        average="binary",
        zero_division=0,
    )

    metrics["precision_bot"] = float(precision)
    metrics["recall_bot"] = float(recall)
    metrics["f1_bot"] = float(f1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics["confusion_matrix"] = {
        "tn_human_pred_human": int(cm[0, 0]),
        "fp_human_pred_bot": int(cm[0, 1]),
        "fn_bot_pred_human": int(cm[1, 0]),
        "tp_bot_pred_bot": int(cm[1, 1]),
    }

    if len(set(labels.tolist())) > 1:
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)

        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
        metrics["log_loss"] = float(log_loss(labels, clipped, labels=[0, 1]))
        metrics["brier"] = float(brier_score_loss(labels, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None
        metrics["brier"] = None

    # The numbers the validator actually pays on, computed on the (calibrated)
    # scores exactly as the on-chain reward does. validator_fpr must stay < 0.10.
    reward = reward_metrics(labels, scores)
    metrics["validator_reward"] = reward["validator_reward"]
    metrics["validator_fpr"] = reward["validator_fpr"]
    metrics["validator_bot_recall"] = reward["validator_bot_recall"]
    metrics["validator_ap"] = reward["validator_ap_score"]
    metrics["human_prob_max"] = reward["human_prob_max"]
    metrics["bot_prob_min"] = reward["bot_prob_min"]

    return metrics


def build_rows(
    samples: List[ChunkSample],
    scores: List[float],
    window_counts: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for idx, (sample, score) in enumerate(zip(samples, scores)):
        label = int(sample.label)
        labeled = label >= 0
        # Same boundary the validator uses (np.round), so per-chunk predictions
        # match the confusion matrix and validator_* metrics.
        pred = int(np.round(score))

        chunk_id = getattr(sample, "chunk_id", None) or f"chunk_{idx}"

        row = {
            "idx": idx,
            "chunk_id": chunk_id,
            "label": label,
            "label_name": ("bot" if label == 1 else "human") if labeled else "unknown",
            "score": float(score),
            "prediction": pred,
            "prediction_name": "bot" if pred == 1 else "human",
            "correct": int(label == pred) if labeled else None,
            "chunk_size_hands": len(sample.chunk),
        }

        if window_counts is not None:
            row["num_windows"] = int(window_counts[idx])

        rows.append(row)

    return rows


def save_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-chunk predictions CSV: {path}")


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved metrics JSON: {path}")


def main() -> None:
    args = parse_args()

    if args.augment_windows and args.window_inference:
        raise ValueError(
            "Use either --augment-windows OR --window-inference, not both.\n"
            "--augment-windows evaluates each generated window as a row.\n"
            "--window-inference evaluates original chunks by aggregating window scores."
        )

    print(f"Loading data: {args.data}")
    samples, labeled = load_eval_samples(args.data, args.split)

    if not samples:
        raise RuntimeError(f"No samples found for split: {args.split}")

    print(f"Original selected chunks: {len(samples)}")
    if not labeled:
        print("No labels detected -> scoring-only mode "
              "(validator reward/FPR/recall need labels and are skipped).")

    # Mode 1:
    # Evaluate generated windows directly as samples.
    if args.augment_windows:
        before = len(samples)

        samples = augment_chunk_windows(
            samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )

        print(
            f"Window-augmented evaluation samples: {before} -> {len(samples)} "
            f"(window_hands={args.window_hands}, stride={args.window_stride})"
        )

    print(f"Loading model: {args.model}")
    detector = load_detector(
        model_path=args.model,
        device=args.device,
        xgb_model_path=args.xgb_model,
    )

    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int32)

    print(f"Evaluating split: {args.split}")
    print(f"Evaluation rows: {len(samples)}")
    if labeled:
        print(f"Human labels: {int((labels == 0).sum())}")
        print(f"Bot labels: {int((labels == 1).sum())}")
    else:
        print("Labels: none (unlabeled scoring-only)")
    print("Boundary: validator np.round (0.5)")
    print(f"XGBoost enabled: {bool(getattr(detector, 'xgb_model', None))}")

    # Mode 2:
    # Original chunks -> windows -> aggregate back to original chunks.
    if args.window_inference:
        scores, window_counts = predict_original_chunks_with_windows(
            detector=detector,
            samples=samples,
            batch_size=args.batch_size,
            window_hands=args.window_hands,
            window_stride=args.window_stride,
            window_agg=args.window_agg,
            keep_short=args.keep_short_window_chunks,
        )

        print(
            f"Window inference enabled: window_hands={args.window_hands}, "
            f"stride={args.window_stride}, agg={args.window_agg}"
        )

    # Default:
    # Direct one score per current sample.
    else:
        chunks = [sample.chunk for sample in samples]

        scores = detector.predict_chunks(
            chunks,
            batch_size=args.batch_size,
        )

        window_counts = None

    if len(scores) != len(samples):
        raise RuntimeError(
            f"Wrong score count. samples={len(samples)}, scores={len(scores)}"
        )

    scores_arr = np.asarray(scores, dtype=np.float32)

    if labeled:
        metrics = safe_metrics(labels, scores_arr)
        metrics["labeled"] = True
    else:
        metrics = scoring_only_metrics(scores_arr)

    metrics["split"] = args.split
    metrics["model"] = str(args.model)
    metrics["xgb_model"] = str(args.xgb_model) if args.xgb_model else None
    metrics["augment_windows"] = bool(args.augment_windows)
    metrics["window_inference"] = bool(args.window_inference)
    metrics["window_hands"] = int(args.window_hands)
    metrics["window_stride"] = int(args.window_stride)
    metrics["window_agg"] = str(args.window_agg)

    rows = build_rows(
        samples=samples,
        scores=scores,
        window_counts=window_counts,
    )

    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    print(f"\n=== First {args.show} predictions ===")
    for row in rows[: args.show]:
        extra = ""
        if "num_windows" in row:
            extra = f" windows={row['num_windows']}"

        correct = "-" if row["correct"] is None else row["correct"]
        print(
            f"idx={row['idx']:04d} "
            f"chunk_id={row['chunk_id']} "
            f"label={row['label_name']:<7} "
            f"score={row['score']:.6f} "
            f"pred={row['prediction_name']:<5} "
            f"correct={correct} "
            f"hands={row['chunk_size_hands']}"
            f"{extra}"
        )

    # Mistakes need ground-truth labels; skip entirely for unlabeled data.
    if labeled:
        print(f"\n=== First {args.show} mistakes ===")
        mistake_rows = [row for row in rows if row["correct"] == 0]

        mistake_rows = sorted(
            mistake_rows,
            key=lambda r: abs(float(r["score"]) - float(r["label"])),
            reverse=True,
        )

        for row in mistake_rows[: args.show]:
            extra = ""
            if "num_windows" in row:
                extra = f" windows={row['num_windows']}"

            print(
                f"idx={row['idx']:04d} "
                f"chunk_id={row['chunk_id']} "
                f"label={row['label_name']:<7} "
                f"score={row['score']:.6f} "
                f"pred={row['prediction_name']:<5} "
                f"hands={row['chunk_size_hands']}"
                f"{extra}"
            )

    if args.out_csv:
        save_csv(args.out_csv, rows)

    if args.out_json:
        save_json(args.out_json, metrics)


if __name__ == "__main__":
    start = time.perf_counter()

    main()

    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")