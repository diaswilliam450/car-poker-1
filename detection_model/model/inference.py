from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .action_vectorizer import CAT_DIM, HAND_META_DIM, ActionVectorizer
from .calibration import ScoreCalibrator
from .features import FeatureVectorizer
from .hierarchical_model import SCHEMA_VERSION, HierarchicalChunkClassifier


def _alias_project_packages() -> None:
    """Make pickled project classes resolve under both package roots.

    Artifacts are typically trained via ``cd detection_model; python -m
    model.train_hierarchical`` so project classes embedded in the ``.pt`` (e.g.
    the stacked-ensemble head) are pickled under the top-level ``model.*``
    package. The miner instead imports ``detection_model.model.*``. Without an
    alias the unpickler raises ``ModuleNotFoundError: No module named 'model'``
    and the miner silently drops to its heuristic fallback. Registering each
    submodule under both roots in ``sys.modules`` lets the same artifact load
    from either entry point.
    """
    this_pkg = __package__ or "model"
    roots = {"model", "detection_model.model", this_pkg}
    submodules = (
        "stacked", "calibration", "features", "action_vectorizer",
        "hierarchical_model", "inference", "scoring", "dataset",
        "hierarchical_dataset",
    )
    pkg_obj = sys.modules.get(this_pkg)
    for root in roots:
        if pkg_obj is not None:
            sys.modules.setdefault(root, pkg_obj)
    for name in submodules:
        try:
            mod = importlib.import_module(f"{this_pkg}.{name}")
        except Exception:
            continue
        for root in roots:
            sys.modules.setdefault(f"{root}.{name}", mod)


class Poker44BotDetector:
    """
    Inference wrapper for the structured-action hierarchical encoder.

    Preferred final scoring path:
        neural chunk embedding + engineered chunk features -> embedded head
        (stacked OOF ensemble by default, or a single XGBoost head)

    If an old artifact has no head model, inference falls back to the auxiliary
    torch probe so older checkpoints still run.
    """

    def __init__(
        self,
        model: HierarchicalChunkClassifier,
        action_vectorizer: ActionVectorizer,
        vectorizer: FeatureVectorizer,
        threshold: float = 0.5,
        device: str | torch.device | None = None,
        xgb_model: Any = None,
        calibrator: ScoreCalibrator | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.action_vectorizer = action_vectorizer
        self.vectorizer = vectorizer
        self.threshold = float(threshold)
        self.xgb_model = xgb_model
        # Reward-aware score post-processor. When present it recenters the score
        # geometry so the decision boundary sits at 0.5 with FPR margin; the
        # miner then needs no extra hand-rolled calibration.
        self.calibrator = calibrator

        model_config = getattr(self.model, "config", {}) or {}
        self.max_hands = int(model_config.get("max_hands", getattr(self.model, "max_hands", 4)))
        self.numeric_dim = int(self.action_vectorizer.numeric_dim)

    @staticmethod
    def _torch_load_artifact(path: Path) -> Dict[str, Any]:
        _alias_project_packages()
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(path, map_location="cpu")
        if not isinstance(artifact, dict):
            raise ValueError(f"Model artifact should be a dict, got {type(artifact)}")
        return artifact

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        device: str | torch.device | None = None,
        xgb_path: str | Path | None = None,
    ) -> "Poker44BotDetector":
        artifact_path = Path(artifact_path).expanduser()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        artifact = cls._torch_load_artifact(artifact_path)
        for key in ("action_vectorizer", "vectorizer", "model_config"):
            if key not in artifact:
                raise KeyError(f"Artifact does not contain '{key}'. Artifact keys: {list(artifact.keys())}")

        action_vectorizer = ActionVectorizer.from_state_dict(artifact["action_vectorizer"])
        vectorizer = FeatureVectorizer.from_state_dict(artifact["vectorizer"])

        model_config = dict(artifact["model_config"])
        if int(model_config.get("schema_version", 1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Model artifact is schema v{model_config.get('schema_version', 1)} but this code is "
                f"v{SCHEMA_VERSION}. The architecture changed (richer tokens + pluggable chunk encoder); "
                f"retrain with `python -m model.train_hierarchical`."
            )
        # Compatibility with the removed hand-feature fusion config.
        model_config.pop("hand_numeric_dim", None)
        model = HierarchicalChunkClassifier(**model_config)

        model_state = artifact.get("model_state_dict") or artifact.get("model") or artifact.get("state_dict")
        if model_state is None:
            raise KeyError(f"Artifact does not contain model weights. Artifact keys: {list(artifact.keys())}")

        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing or unexpected:
            print(f"Loaded model with compatibility mode: missing={len(missing)}, unexpected={len(unexpected)}")

        threshold = float(artifact.get("threshold", 0.5))
        # `head_model` is the canonical key (stacked ensemble or XGBoost); older
        # artifacts only carry `xgb_model`. Both expose predict_proba.
        xgb_model = (
            artifact.get("head_model")
            or artifact.get("xgb_model")
            or artifact.get("model_head")
        )
        calibrator = ScoreCalibrator.from_dict(artifact.get("calibrator"))

        if xgb_path:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError("joblib is required to load an external XGBoost model.") from exc
            xgb_path = Path(xgb_path).expanduser()
            if not xgb_path.exists():
                raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")
            xgb_payload = joblib.load(xgb_path)
            if isinstance(xgb_payload, dict):
                xgb_model = xgb_payload.get("xgb_model") or xgb_payload.get("model")
                if "threshold" in xgb_payload:
                    threshold = float(xgb_payload["threshold"])
            else:
                xgb_model = xgb_payload
            if xgb_model is None:
                raise KeyError(f"Could not find XGBoost model inside {xgb_path}")

        return cls(
            model=model,
            action_vectorizer=action_vectorizer,
            vectorizer=vectorizer,
            threshold=threshold,
            device=device,
            xgb_model=xgb_model,
            calibrator=calibrator,
        )

    @property
    def has_calibrator(self) -> bool:
        """True when the artifact embeds a fitted, non-identity calibrator."""
        return self.calibrator is not None and not self.calibrator.is_identity

    def _make_inference_batch(self, chunks: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        if not chunks:
            raise ValueError("No chunks provided for inference.")

        encoded = [self.action_vectorizer.encode_chunk(chunk, max_hands=self.max_hands) for chunk in chunks]

        batch_size = len(chunks)
        max_hands = max(1, max(len(enc["cat"]) for enc in encoded))
        max_actions = max(
            1, max((len(rows) for enc in encoded for rows in enc["cat"]), default=1)
        )

        action_cat = torch.zeros((batch_size, max_hands, max_actions, CAT_DIM), dtype=torch.long)
        action_num = torch.zeros((batch_size, max_hands, max_actions, self.numeric_dim), dtype=torch.float32)
        action_mask = torch.zeros((batch_size, max_hands, max_actions), dtype=torch.bool)
        hand_mask = torch.zeros((batch_size, max_hands), dtype=torch.bool)
        hand_meta = torch.zeros((batch_size, max_hands, HAND_META_DIM), dtype=torch.float32)
        hand_end = torch.zeros((batch_size, max_hands), dtype=torch.long)

        for batch_idx, enc in enumerate(encoded):
            metas, ends = enc["hand_meta"], enc["hand_end"]
            for hand_idx, (cat_rows, num_rows) in enumerate(zip(enc["cat"], enc["num"])):
                if hand_idx >= max_hands:
                    break
                length = min(len(cat_rows), max_actions)
                if length <= 0:
                    continue

                cat_arr = np.asarray(cat_rows[:length], dtype=np.int64)
                num_arr = np.asarray(num_rows[:length], dtype=np.float32)
                action_cat[batch_idx, hand_idx, :length, :] = torch.tensor(cat_arr[:, :CAT_DIM], dtype=torch.long)
                action_num[batch_idx, hand_idx, :length, :] = torch.tensor(num_arr, dtype=torch.float32)
                action_mask[batch_idx, hand_idx, :length] = True
                hand_mask[batch_idx, hand_idx] = True
                if hand_idx < len(metas):
                    hand_meta[batch_idx, hand_idx, :] = torch.tensor(metas[hand_idx], dtype=torch.float32)
                if hand_idx < len(ends):
                    hand_end[batch_idx, hand_idx] = int(ends[hand_idx])

        features = torch.tensor(np.asarray(self.vectorizer.transform(chunks), dtype=np.float32), dtype=torch.float32)

        return {
            "action_cat": action_cat.to(self.device),
            "action_num": action_num.to(self.device),
            "action_mask": action_mask.to(self.device),
            "hand_mask": hand_mask.to(self.device),
            "hand_meta": hand_meta.to(self.device),
            "hand_end": hand_end.to(self.device),
            "features": features.to(self.device),
        }

    @torch.no_grad()
    def _predict_neural_batch(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        logits = self.model(
            action_cat=batch["action_cat"],
            action_num=batch["action_num"],
            action_mask=batch["action_mask"],
            hand_mask=batch["hand_mask"],
            features=batch["features"],
            hand_meta=batch["hand_meta"],
            hand_end=batch["hand_end"],
        ).view(-1)
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy().astype(float).tolist()

    @torch.no_grad()
    def _predict_xgb_batch(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model is not loaded.")

        chunk_embedding = self.model.extract_chunk_embedding(
            action_cat=batch["action_cat"],
            action_num=batch["action_num"],
            action_mask=batch["action_mask"],
            hand_mask=batch["hand_mask"],
            hand_meta=batch["hand_meta"],
            hand_end=batch["hand_end"],
        )
        emb_np = chunk_embedding.detach().cpu().numpy().astype(np.float32)
        feat_np = batch["features"].detach().cpu().numpy().astype(np.float32)
        xgb_x = np.concatenate([emb_np, feat_np], axis=1)

        if hasattr(self.xgb_model, "predict_proba"):
            proba = self.xgb_model.predict_proba(xgb_x)
            probs = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
        else:
            probs = self.xgb_model.predict(xgb_x)
        return np.asarray(probs, dtype=float).reshape(-1).tolist()

    @torch.no_grad()
    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        batch_size: int = 64,
        min_action_size: int = 5,
        max_action_size: int = 8,
        return_raw: bool = False,
    ) -> List[float]:
        """Score chunks and apply the embedded reward-aware calibrator.

        Returns one bot-risk score per chunk (higher = more bot-like). The score
        is the model's raw probability passed through :class:`ScoreCalibrator`
        when one is embedded in the artifact; otherwise the raw probability is
        returned unchanged. Set ``return_raw=True`` to get pre-calibration scores
        instead (useful for the ``diagnose`` tooling).
        """
        if not chunks:
            return []

        raw_scores: List[float] = []
        self.model.eval()
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch = self._make_inference_batch(batch_chunks)
            probs = self._predict_xgb_batch(batch) if self.xgb_model is not None else self._predict_neural_batch(batch)
            for prob in probs:
                raw_scores.append(max(0.0, min(1.0, float(prob))))

        if len(raw_scores) != len(chunks):
            raise RuntimeError(f"Wrong score count: chunks={len(chunks)}, scores={len(raw_scores)}")

        if return_raw or self.calibrator is None:
            return [round(score, 6) for score in raw_scores]

        calibrated = self.calibrator.transform(raw_scores)
        return [round(float(score), 6) for score in calibrated]

    @torch.no_grad()
    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        scores = self.predict_chunks([chunk], batch_size=1)
        return scores[0] if scores else 0.5

    def predict_labels(self, chunks: List[List[Dict[str, Any]]], batch_size: int = 64) -> List[bool]:
        scores = self.predict_chunks(chunks, batch_size=batch_size)
        return [score >= self.threshold for score in scores]
