"""Serving wrapper for the v2 feature-based model (used by the miner).

    detector = Poker44V2Detector.load("artifacts/p44_v2_lgbm_canon.joblib")
    scores = detector.predict_chunks(chunks)   # one calibrated bot score per chunk

Whole-chunk, order-invariant scoring: each chunk's hands are turned into the fixed
feature vector (actions sorted by ``action_id`` exactly as in training), scored by
the LightGBM model, then passed through the embedded cliff-aware calibrator so the
0.5 boundary already sits under the FPR cliff.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .calibrate import apply_calibrator
from .features import chunk_feature_vector
from .schema import _sort_actions


class Poker44V2Detector:
    """LightGBM tabular detector + embedded monotone cliff-aware calibrator."""

    def __init__(
        self,
        model: Any,
        calibrator: Dict[str, Any],
        feature_names: List[str],
        threshold: float = 0.5,
        metadata: Dict[str, Any] | None = None,
        seq_model: Any = None,
        blend_weights: Any = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.metadata = dict(metadata or {})
        # Optional TCN sequence learner blended with the LightGBM. When present,
        # the served score is blend_weights[0]*lgbm + blend_weights[1]*tcn.
        self.seq_model = seq_model
        self.blend_weights = tuple(blend_weights) if blend_weights is not None else None

    @classmethod
    def load(cls, path: str | Path) -> "Poker44V2Detector":
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"v2 model artifact not found: {path}")
        import joblib

        art = joblib.load(path)
        if not isinstance(art, dict) or "feature_names" not in art:
            raise ValueError(f"Not a v2 artifact (missing feature_names): {path}")
        # Blend artifact (LGBM + TCN) or single-model artifact.
        lgbm = art.get("lgbm_model") or art.get("model")
        if lgbm is None:
            raise ValueError(f"v2 artifact has no model/lgbm_model: {path}")
        return cls(
            model=lgbm,
            calibrator=art.get("calibrator") or {},
            feature_names=art["feature_names"],
            threshold=float(art.get("threshold", 0.5)),
            metadata={"backend": art.get("backend"), "val_metrics": art.get("val_metrics")},
            seq_model=art.get("seq_model"),
            blend_weights=art.get("blend_weights"),
        )

    @property
    def has_calibrator(self) -> bool:
        return bool(self.calibrator) and "grid" in self.calibrator

    def _feature_rows(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        rows: List[List[float]] = []
        for chunk in chunks:
            hands = [_sort_actions(h) for h in (chunk or []) if isinstance(h, dict)]
            feats = chunk_feature_vector(hands)
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        x = np.asarray(rows, dtype=np.float64)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _raw_scores(self, x: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(x)
        return np.asarray(proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba, dtype=float)

    def predict_chunks(self, chunks: List[List[Dict[str, Any]]], return_raw: bool = False) -> List[float]:
        """Return one calibrated bot-risk score per chunk (higher = more bot-like)."""
        if not chunks:
            return []
        x = self._feature_rows(chunks)
        raw = self._raw_scores(x)
        if self.seq_model is not None and self.blend_weights is not None:
            seq = np.asarray(self.seq_model.predict_proba(chunks))[:, 1]
            w0, w1 = self.blend_weights
            raw = w0 * raw + w1 * seq
        raw = np.clip(raw, 0.0, 1.0)
        if return_raw or not self.has_calibrator:
            return [round(float(v), 6) for v in raw]
        cal = apply_calibrator(self.calibrator, raw)
        return [round(float(v), 6) for v in cal]

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        scores = self.predict_chunks([chunk])
        return scores[0] if scores else 0.5
