from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from .action_vectorizer import ActionVectorizer
from .calibration import ScoreCalibrator
from .dataset import augment_chunk_prefixes, augment_chunk_windows, load_public_benchmark
from .features import FeatureVectorizer
from .hierarchical_dataset import HierarchicalPokerChunkDataset, hierarchical_collate_batch
from .hierarchical_model import HierarchicalChunkClassifier
from .scoring import format_reward_line, print_reward_diagnostics, reward_metrics
from .stacked import StackedEnsemble


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        artifact = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location=device)
    if not isinstance(artifact, dict):
        raise ValueError(f"Checkpoint should be dict, got: {type(artifact)}")
    return artifact


def load_checkpoint(path: Optional[str], device: torch.device) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path_obj}")
    print(f"Loading checkpoint from: {path_obj}")
    checkpoint = safe_torch_load(path_obj, device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    return checkpoint


def backup_existing_file(path: Path, overwrite: bool) -> None:
    if overwrite or not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    print(f"Existing checkpoint backed up to: {backup_path}")


def apply_checkpoint_config(args: argparse.Namespace, checkpoint: Optional[Dict[str, Any]]) -> None:
    if checkpoint is None:
        return
    config = checkpoint.get("model_config") or {}
    if not isinstance(config, dict) or not config:
        return

    print("Using architecture config from checkpoint:")
    print(json.dumps(config, indent=2, default=str))
    args.max_actions_per_hand = int(config.get("max_actions_per_hand", args.max_actions_per_hand))
    args.max_hands = int(config.get("max_hands", args.max_hands))
    args.d_model = int(config.get("d_model", args.d_model))
    args.layers = int(config.get("n_layers", args.layers))
    args.heads = int(config.get("n_heads", args.heads))
    args.chunk_layers = int(config.get("chunk_layers", config.get("gru_layers", args.chunk_layers)))
    args.dropout = float(config.get("dropout", args.dropout))


def fit_or_load_action_vectorizer(
    checkpoint: Optional[Dict[str, Any]],
    train_chunks: List[List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> ActionVectorizer:
    if checkpoint is not None and "action_vectorizer" in checkpoint:
        action_vectorizer = ActionVectorizer.from_state_dict(checkpoint["action_vectorizer"])
        print("Loaded ActionVectorizer from checkpoint")
        return action_vectorizer

    action_vectorizer = ActionVectorizer(max_actions_per_hand=args.max_actions_per_hand)
    action_vectorizer.fit(train_chunks, min_freq=args.min_freq)
    print("Fitted ActionVectorizer from training chunks")
    print("Street vocab size:", action_vectorizer.street_vocab_size)
    print("Action type vocab size:", action_vectorizer.action_type_vocab_size)
    print("Seat vocab size:", action_vectorizer.seat_vocab_size)
    print("Numeric dim:", action_vectorizer.numeric_dim)
    return action_vectorizer


def fit_or_load_feature_vectorizer(
    checkpoint: Optional[Dict[str, Any]],
    train_chunks: List[List[Dict[str, Any]]],
) -> FeatureVectorizer:
    if checkpoint is not None and "vectorizer" in checkpoint:
        vectorizer = FeatureVectorizer.from_state_dict(checkpoint["vectorizer"])
        print("Loaded FeatureVectorizer from checkpoint")
        return vectorizer

    vectorizer = FeatureVectorizer()
    vectorizer.fit(train_chunks)
    print("Fitted FeatureVectorizer from training chunks")
    print("Feature dim:", len(vectorizer.feature_names))
    return vectorizer


def make_model(
    action_vectorizer: ActionVectorizer,
    feature_vectorizer: FeatureVectorizer,
    args: argparse.Namespace,
) -> HierarchicalChunkClassifier:
    if args.d_model % args.heads != 0:
        raise ValueError(f"d_model must be divisible by heads. Got d_model={args.d_model}, heads={args.heads}")

    return HierarchicalChunkClassifier(
        street_vocab_size=action_vectorizer.street_vocab_size,
        action_type_vocab_size=action_vectorizer.action_type_vocab_size,
        seat_vocab_size=action_vectorizer.seat_vocab_size,
        numeric_dim=action_vectorizer.numeric_dim,
        feature_dim=len(feature_vectorizer.feature_names),
        amount_bucket_vocab_size=action_vectorizer.amount_bucket_vocab_size,
        pot_flow_vocab_size=action_vectorizer.pot_flow_vocab_size,
        first_in_street_vocab_size=action_vectorizer.first_in_street_vocab_size,
        actor_role_vocab_size=action_vectorizer.actor_role_vocab_size,
        street_position_vocab_size=action_vectorizer.street_position_vocab_size,
        hand_end_vocab_size=action_vectorizer.hand_end_vocab_size,
        hand_meta_dim=action_vectorizer.hand_meta_dim,
        max_actions_per_hand=args.max_actions_per_hand,
        max_hands=args.max_hands,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        chunk_layers=args.chunk_layers,
        chunk_encoder=args.chunk_encoder,
        use_hand_position=not args.no_hand_position,
        dropout=args.dropout,
        pad_id=0,
    )


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_model(model: HierarchicalChunkClassifier, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        action_cat=batch["action_cat"],
        action_num=batch["action_num"],
        action_mask=batch["action_mask"],
        hand_mask=batch["hand_mask"],
        features=batch["features"],
        hand_meta=batch["hand_meta"],
        hand_end=batch["hand_end"],
    )
    return logits.view(-1)


def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(labels) == 0:
        return {"best_threshold": 0.5, "best_f1": 0.0, "best_accuracy": 0.0}
    if len(set(labels.tolist())) < 2:
        return {
            "best_threshold": 0.5,
            "best_f1": 0.0,
            "best_accuracy": float(accuracy_score(labels, scores >= 0.5)),
        }

    candidate_thresholds = np.unique(
        np.concatenate([np.linspace(0.05, 0.95, 91, dtype=np.float32), scores.astype(np.float32)])
    )
    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = -1.0

    for threshold in candidate_thresholds:
        preds = (scores >= threshold).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        acc = float(accuracy_score(labels, preds))
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and acc > best_accuracy):
            best_f1 = float(f1)
            best_accuracy = float(acc)
            best_threshold = float(threshold)

    return {
        "best_threshold": float(best_threshold),
        "best_f1": float(max(best_f1, 0.0)),
        "best_accuracy": float(max(best_accuracy, 0.0)),
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32).clip(1e-6, 1 - 1e-6)
    preds = (scores >= threshold).astype(np.int32)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics: Dict[str, Any] = {
        "count": int(len(labels)),
        "threshold": float(threshold),
        "human_count": int((labels == 0).sum()),
        "bot_count": int((labels == 1).sum()),
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else 0.0,
        "confusion_matrix": {
            "tn_human_pred_human": int(cm[0, 0]),
            "fp_human_pred_bot": int(cm[0, 1]),
            "fn_bot_pred_human": int(cm[1, 0]),
            "tp_bot_pred_bot": int(cm[1, 1]),
        },
    }
    metrics.update(find_best_threshold(labels, scores))

    human_scores = scores[labels == 0]
    bot_scores = scores[labels == 1]
    metrics["human_score_mean"] = float(human_scores.mean()) if len(human_scores) else None
    metrics["bot_score_mean"] = float(bot_scores.mean()) if len(bot_scores) else None

    if len(set(labels.tolist())) > 1:
        metrics["log_loss"] = float(log_loss(labels, scores, labels=[0, 1]))
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        metrics["log_loss"] = 0.0
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    return metrics


@torch.no_grad()
def evaluate_neural(
    model: HierarchicalChunkClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    labels_all: List[float] = []
    scores_all: List[float] = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = forward_model(model, batch)
        labels = batch["labels"].view(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        scores_all.extend(probs.detach().cpu().numpy().tolist())

    metrics = compute_metrics(
        labels=np.asarray(labels_all, dtype=np.int32),
        scores=np.asarray(scores_all, dtype=np.float32),
        threshold=threshold,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    # Reward-aware view: the numbers the validator actually pays on (raw,
    # pre-calibration). AP here is the ceiling the calibrator later banks.
    reward = reward_metrics(labels_all, scores_all)
    metrics["validator_reward"] = reward["validator_reward"]
    metrics["validator_fpr"] = reward["validator_fpr"]
    metrics["validator_bot_recall"] = reward["validator_bot_recall"]
    metrics["validator_ap"] = reward["validator_ap_score"]
    metrics["reward_line"] = format_reward_line(reward)
    return metrics


def train_one_epoch(
    model: HierarchicalChunkClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    end_epoch: int,
) -> float:
    model.train()
    running_losses: List[float] = []

    for batch in tqdm(loader, desc=f"epoch {epoch}/{end_epoch}"):
        batch = batch_to_device(batch, device)
        labels = batch["labels"].view(-1)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        running_losses.append(float(loss.item()))

    return float(np.mean(running_losses)) if running_losses else 0.0


@torch.no_grad()
def extract_xgb_matrix(
    model: HierarchicalChunkClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []

    for batch in tqdm(loader, desc="extract xgb features"):
        batch = batch_to_device(batch, device)
        chunk_embedding = model.extract_chunk_embedding(
            action_cat=batch["action_cat"],
            action_num=batch["action_num"],
            action_mask=batch["action_mask"],
            hand_mask=batch["hand_mask"],
            hand_meta=batch["hand_meta"],
            hand_end=batch["hand_end"],
        )
        emb_np = chunk_embedding.detach().cpu().numpy().astype(np.float32)
        feat_np = batch["features"].detach().cpu().numpy().astype(np.float32)
        labels_np = batch["labels"].detach().cpu().numpy().astype(np.int32)
        x_rows.append(np.concatenate([emb_np, feat_np], axis=1))
        y_rows.append(labels_np)

    if not x_rows:
        raise RuntimeError("No samples available for XGBoost training.")
    return np.vstack(x_rows).astype(np.float32), np.concatenate(y_rows).astype(np.int32)


def make_xgb_classifier(args: argparse.Namespace, y_train: np.ndarray) -> XGBClassifier:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1.0)
    return XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_lambda=args.xgb_reg_lambda,
        reg_alpha=args.xgb_reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=args.xgb_tree_method,
        random_state=args.seed,
        n_jobs=args.xgb_n_jobs,
        scale_pos_weight=scale_pos_weight,
    )


def build_head(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Any, str]:
    """Fit and return the final ranking head plus its name.

    ``stacked`` builds the OOF ensemble (ported from the reference model);
    ``xgboost`` keeps the original single-booster head. Both expose
    ``predict_proba`` so the rest of the pipeline is head-agnostic.
    """
    if args.head == "stacked":
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        head = StackedEnsemble(
            scale_pos_weight=neg / max(pos, 1.0),
            n_folds=args.stack_folds,
            top_k=(args.stack_top_k or None),
            use_lightgbm=not args.no_stack_lightgbm,
            use_catboost=not args.no_stack_catboost,
            meta_c=args.stack_meta_c,
            random_state=args.seed,
            n_jobs=args.xgb_n_jobs,
        )
        print("Training stacked ensemble head...")
        head.fit(x_train, y_train)
        print(f"  base learners: {head.base_names_}")
        if head.selected_idx_ is not None:
            print(f"  selected top-{len(head.selected_idx_)} of {x_train.shape[1]} columns")
        return head, "stacked"

    head = make_xgb_classifier(args, y_train)
    print("Training final XGBoost head...")
    head.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return head, "xgboost"


def predict_xgb_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=np.float32)
    raw = np.asarray(model.predict(x), dtype=np.float32).reshape(-1)
    if raw.min(initial=0.0) < 0.0 or raw.max(initial=1.0) > 1.0:
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw.astype(np.float32)


def save_artifact(
    out_path: Path,
    model: HierarchicalChunkClassifier,
    optimizer: torch.optim.Optimizer,
    action_vectorizer: ActionVectorizer,
    vectorizer: FeatureVectorizer,
    args: argparse.Namespace,
    history: List[Dict[str, Any]],
    epoch: int,
    best_metric: float,
    threshold: float,
    xgb_model: Any | None,
    xgb_metrics: Dict[str, Any] | None,
    xgb_input_dim: int | None,
    calibrator: ScoreCalibrator | None = None,
    head_name: str = "xgboost",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "architecture": f"v3_action_transformer_hand_{model.chunk_encoder}_{head_name}_head",
        "schema_version": int(model.config.get("schema_version", 3)),
        "final_head": head_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": dict(model.config),
        "action_vectorizer": action_vectorizer.state_dict(),
        "vectorizer": vectorizer.state_dict(),
        # `head_model` is the canonical key; `xgb_model` is kept pointing at the
        # same object for backward-compatible inference loaders.
        "head_model": xgb_model,
        "xgb_model": xgb_model,
        "xgb_input_dim": xgb_input_dim,
        "calibrator": calibrator.to_dict() if calibrator is not None else None,
        "threshold": float(threshold),
        "history": history,
        "xgb_metrics": xgb_metrics or {},
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "training_data": str(args.data),
        "train_args": vars(args),
        "feature_description": "concat(neural_chunk_embedding, compact_engineered_chunk_features)",
    }
    torch.save(artifact, out_path)
    print(f"Saved full model artifact to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first-model hierarchical encoder and final XGBoost head."
    )

    parser.add_argument("--data", required=True, help="Path to public_miner_benchmark.json.gz")
    parser.add_argument("--out", default="artifacts/p44_first_arch_xgb.pt")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-hands", type=int, default=20)
    parser.add_argument("--max-actions-per-hand", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1, help="Action-level Transformer layers")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--chunk-layers", type=int, default=1, help="Chunk encoder layers (Transformer or GRU)")
    parser.add_argument(
        "--chunk-encoder",
        choices=("transformer", "gru"),
        default="transformer",
        help="Hand->chunk encoder. transformer (recommended): permutation-invariant set encoder. "
        "gru: the original ordered bidirectional GRU.",
    )
    parser.add_argument(
        "--no-hand-position",
        action="store_true",
        help="Drop the soft normalized-hand-index position channel (pure set encoding).",
    )
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--augment-prefixes", action="store_true")
    parser.add_argument("--min-prefix-hands", type=int, default=4)
    parser.add_argument("--max-prefixes-per-chunk", type=int, default=32)
    parser.add_argument("--augment-windows", action="store_true")
    parser.add_argument("--augment-validation-windows", action="store_true")
    parser.add_argument("--window-hands", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--keep-short-window-chunks", action="store_true")

    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience. 0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--no-auto-threshold", action="store_true")

    parser.add_argument("--calibrate-visible-actions", action="store_true")
    parser.add_argument("--min-visible-action-window-size", type=int, default=5)
    parser.add_argument("--max-visible-action-window-size", type=int, default=8)
    parser.add_argument("--calibrate-validation-visible-actions", action="store_true")

    # Final head selection. `stacked` (default) is the OOF ensemble ported from
    # the reference model (XGBoost + ExtraTrees + RandomForest + optional
    # LightGBM/CatBoost, combined by a logistic meta-learner) and lifts AP.
    # `xgboost` keeps the original single-booster head.
    parser.add_argument(
        "--head",
        choices=("stacked", "xgboost"),
        default="stacked",
        help="Final ranking head: stacked OOF ensemble (default) or single XGBoost.",
    )
    parser.add_argument("--stack-folds", type=int, default=5, help="OOF folds for the stacked meta-learner.")
    parser.add_argument("--stack-top-k", type=int, default=0, help="Keep top-K important input columns (0 = all).")
    parser.add_argument("--stack-meta-c", type=float, default=1.0, help="Inverse-reg strength of the logistic meta-learner.")
    parser.add_argument("--no-stack-lightgbm", action="store_true", help="Exclude LightGBM from the ensemble.")
    parser.add_argument("--no-stack-catboost", action="store_true", help="Exclude CatBoost from the ensemble.")

    # Final XGBoost head. This replaces the old separate model.train_xgboost step.
    parser.add_argument("--xgb-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--xgb-n-jobs", type=int, default=0)

    # Reward-aware score calibration (fit on the validation split, embedded in
    # the artifact, and applied automatically by the miner at inference time).
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit and embed a reward-aware ScoreCalibrator (default on).",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=("reward", "ap_first", "recall"),
        default="reward",
        help="What the calibrator maximizes. reward (recommended): AP is invariant "
        "under monotone calibration, so optimizing the full reward correctly trades recall vs FPR.",
    )
    parser.add_argument(
        "--calibration-target-fpr",
        type=float,
        default=0.04,
        help="Conformal FPR target used to seed the logit shift (stay below the 0.10 cliff).",
    )
    parser.add_argument(
        "--calibration-max-fpr",
        type=float,
        default=0.05,
        help="Hard ceiling: calibration configs at/above this validation FPR are rejected.",
    )
    parser.add_argument(
        "--no-calibration-isotonic",
        action="store_true",
        help="Disable the isotonic recalibration stage (keep only remap + logit shift).",
    )
    parser.add_argument(
        "--calibration-spread",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit a monotone quantile-spread anti-collapse stage (default on). "
        "Re-expands collapsed score bands; AP-invariant in-distribution.",
    )
    parser.add_argument(
        "--calibration-spread-blend",
        type=float,
        default=0.9,
        help="Spread strength: 1.0 = pure rank/uniform, 0.0 = identity (no spread).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(args.resume_from, device)
    apply_checkpoint_config(args, checkpoint)

    train_samples, val_samples = load_public_benchmark(args.data, seed=args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Dataset loading failed: train or validation split is empty.")

    print(f"Original train chunks: {len(train_samples)}")
    print(f"Original validation chunks: {len(val_samples)}")

    if args.augment_prefixes and args.augment_windows:
        raise ValueError("Use either --augment-prefixes or --augment-windows, not both.")

    if args.augment_prefixes:
        before = len(train_samples)
        train_samples = augment_chunk_prefixes(
            train_samples,
            min_prefix_hands=args.min_prefix_hands,
            max_prefixes_per_chunk=args.max_prefixes_per_chunk,
            include_full_chunk=True,
        )
        print(f"Prefix-augmented train chunks: {before} -> {len(train_samples)}")

    if args.augment_windows:
        before = len(train_samples)
        train_samples = augment_chunk_windows(
            train_samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )
        print(f"Sliding-window train chunks: {before} -> {len(train_samples)}")

    if args.augment_validation_windows:
        before = len(val_samples)
        val_samples = augment_chunk_windows(
            val_samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )
        print(f"Sliding-window validation chunks: {before} -> {len(val_samples)}")

    train_chunks = [sample.chunk for sample in train_samples]
    action_vectorizer = fit_or_load_action_vectorizer(checkpoint, train_chunks, args)
    vectorizer = fit_or_load_feature_vectorizer(checkpoint, train_chunks)

    train_ds = HierarchicalPokerChunkDataset(
        samples=train_samples,
        action_vectorizer=action_vectorizer,
        feature_vectorizer=vectorizer,
        max_hands=args.max_hands,
        calibrate_visible_actions=args.calibrate_visible_actions,
        min_visible_action_window_size=args.min_visible_action_window_size,
        max_visible_action_window_size=args.max_visible_action_window_size,
        recompute_features_after_calibration=True,
    )
    val_ds = HierarchicalPokerChunkDataset(
        samples=val_samples,
        action_vectorizer=action_vectorizer,
        feature_vectorizer=vectorizer,
        max_hands=args.max_hands,
        calibrate_visible_actions=args.calibrate_validation_visible_actions,
        min_visible_action_window_size=args.min_visible_action_window_size,
        max_visible_action_window_size=args.max_visible_action_window_size,
        recompute_features_after_calibration=True,
    )

    collate = lambda b: hierarchical_collate_batch(
        b,
        cat_pad_id=0,
        numeric_dim=action_vectorizer.numeric_dim,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        drop_last=False,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        drop_last=False,
        collate_fn=collate,
    )
    train_extract_loader = DataLoader(
        train_ds,
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        drop_last=False,
        collate_fn=collate,
    )
    val_extract_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        drop_last=False,
        collate_fn=collate,
    )

    print("Final train chunks:", len(train_ds))
    print("Validation chunks:", len(val_ds))
    print("Action numeric_dim:", action_vectorizer.numeric_dim)
    print("Engineered feature_dim:", len(vectorizer.feature_names))

    model = make_model(action_vectorizer, vectorizer, args).to(device)

    if checkpoint is not None:
        state = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint.get("state_dict")
        if state is None:
            raise KeyError(f"No model state found in checkpoint. Keys={list(checkpoint.keys())}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded model weights with compatibility mode: missing={len(missing)}, unexpected={len(unexpected)}")

    labels = np.asarray([sample.label for sample in train_samples], dtype=np.float32)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    if args.no_pos_weight:
        pos_weight = None
        print("Using BCEWithLogitsLoss without pos_weight")
    else:
        pos_weight_value = neg / max(pos, 1.0)
        pos_weight = torch.tensor([pos_weight_value], device=device)
        print(f"Class balance: human={int(neg)}, bot={int(pos)}, pos_weight={pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_state = None
    best_threshold = float(args.threshold)
    epochs_without_improvement = 0

    if checkpoint is not None:
        if isinstance(checkpoint.get("history"), list):
            history = checkpoint["history"]
        if isinstance(checkpoint.get("best_metric"), (int, float)):
            best_metric = float(checkpoint["best_metric"])
        if isinstance(checkpoint.get("threshold"), (int, float)):
            best_threshold = float(checkpoint["threshold"])
        if not args.fine_tune:
            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                    print("Loaded optimizer state from checkpoint")
                except Exception as exc:
                    print(f"Could not load optimizer state. Fresh optimizer will be used. Error: {exc}")
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"Resume mode: starting from epoch {start_epoch}")
        else:
            print("Fine-tune mode: loaded weights, fresh optimizer.")

    out_path = Path(args.out).expanduser()
    backup_existing_file(out_path, overwrite=args.overwrite)
    final_epoch = start_epoch - 1
    end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        final_epoch = epoch
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            end_epoch=end_epoch,
        )
        metrics = evaluate_neural(model, val_loader, device, criterion, threshold=args.threshold)
        metrics["train_loss"] = float(train_loss)
        metrics["epoch"] = int(epoch)
        metrics["stage"] = "neural_backbone"
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        roc_auc = float(metrics.get("roc_auc", 0.0) or 0.0)
        pr_auc = float(metrics.get("pr_auc", 0.0) or 0.0)
        best_f1 = float(metrics.get("best_f1", 0.0) or 0.0)
        log_loss_value = float(metrics.get("log_loss", 0.0) or 0.0)
        selection_metric = roc_auc + pr_auc + best_f1 - min(log_loss_value, 5.0) * 0.01
        improved = selection_metric > (best_metric + float(args.min_delta))

        if improved:
            best_metric = selection_metric
            epochs_without_improvement = 0
            best_threshold = float(metrics.get("best_threshold", args.threshold)) if not args.no_auto_threshold else float(args.threshold)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            print(f"New best neural backbone metric: {best_metric:.6f}; probe threshold={best_threshold:.4f}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print("Extracting neural embeddings + engineered features for XGBoost...")
    x_train, y_train = extract_xgb_matrix(model, train_extract_loader, device)
    x_val, y_val = extract_xgb_matrix(model, val_extract_loader, device)
    print(f"XGBoost train matrix: {x_train.shape}")
    print(f"XGBoost validation matrix: {x_val.shape}")

    xgb_model, head_name = build_head(args, x_train, y_train, x_val, y_val)

    train_scores = predict_xgb_scores(xgb_model, x_train)
    val_scores = predict_xgb_scores(xgb_model, x_val)
    fixed_val_metrics = compute_metrics(y_val, val_scores, threshold=args.threshold)
    xgb_threshold = float(fixed_val_metrics.get("best_threshold", args.threshold)) if not args.no_auto_threshold else float(args.threshold)
    train_metrics = compute_metrics(y_train, train_scores, threshold=xgb_threshold)
    val_metrics = compute_metrics(y_val, val_scores, threshold=xgb_threshold)

    print("Final XGBoost train metrics:")
    print(json.dumps(train_metrics, indent=2))
    print("Final XGBoost validation metrics:")
    print(json.dumps(val_metrics, indent=2))

    xgb_selection_metric = (
        float(val_metrics.get("roc_auc", 0.0) or 0.0)
        + float(val_metrics.get("pr_auc", 0.0) or 0.0)
        + float(val_metrics.get("best_f1", 0.0) or 0.0)
        - min(float(val_metrics.get("log_loss", 0.0) or 0.0), 5.0) * 0.01
    )

    xgb_metrics = {
        "head": head_name,
        "train": train_metrics,
        "validation": val_metrics,
        "head_params": xgb_model.get_params() if hasattr(xgb_model, "get_params") else {},
    }

    # Reward-aware calibration. Fit on the held-out validation scores (the same
    # split used for threshold selection) so the embedded calibrator maximizes
    # the validator reward objective while holding FPR under the safety cliff.
    calibrator: ScoreCalibrator | None = None
    if args.calibrate:
        print("Fitting reward-aware calibrator on validation split...")
        print_reward_diagnostics("validation (raw, pre-calibration)", y_val, val_scores)
        calibrator = ScoreCalibrator(
            objective=args.calibration_objective,
            target_fpr=args.calibration_target_fpr,
            max_fpr=args.calibration_max_fpr,
        )
        calibrator.fit(
            val_scores,
            y_val,
            use_spread=args.calibration_spread,
            spread_blend=args.calibration_spread_blend,
            use_isotonic=not args.no_calibration_isotonic,
        )
        print("  " + calibrator.summary(y_val, val_scores))
        print_reward_diagnostics(
            "validation (calibrated)", y_val, calibrator.transform(val_scores)
        )
        if calibrator.is_identity:
            print("  Calibrator stayed identity (no config beat raw under the FPR ceiling).")

    save_artifact(
        out_path=out_path,
        model=model,
        optimizer=optimizer,
        action_vectorizer=action_vectorizer,
        vectorizer=vectorizer,
        args=args,
        history=history,
        epoch=final_epoch,
        best_metric=xgb_selection_metric,
        threshold=xgb_threshold,
        xgb_model=xgb_model,
        xgb_metrics=xgb_metrics,
        xgb_input_dim=int(x_train.shape[1]),
        calibrator=calibrator,
        head_name=head_name,
    )

    print("Training finished.")
    print(f"Saved threshold: {xgb_threshold:.6f}")
    print(f"Saved artifact with embedded '{head_name}' head: {out_path}")
    print(f"Embedded reward-aware calibrator: {'yes' if calibrator else 'no'}")


if __name__ == "__main__":
    main()
