"""Inference on an (unlabeled) evaluation file — e.g. chunks1.json.

    python -m model_v2.predict --model artifacts/p44_v2_lgbm.joblib \
        --data data/chunks1.json --out eval_predictions.json

Emits one record per chunk with the calibrated bot probability, the 0.5 decision,
a confidence band, and the top contributing features (global tree importance
intersected with this chunk's most extreme feature values).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .calibrate import apply_calibrator
from .dataset import build_feature_matrix


def _load(model_path: str):
    import joblib
    art = joblib.load(Path(model_path).expanduser())
    return art["model"], art["calibrator"], art["feature_names"]


def _importances(model, names: List[str]) -> np.ndarray:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return np.ones(len(names))
    imp = np.asarray(imp, dtype=float)
    return imp / (imp.sum() + 1e-12)


def _confidence(p: float) -> str:
    d = abs(p - 0.5)
    return "high" if d >= 0.35 else "medium" if d >= 0.15 else "low"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eval_predictions.json")
    ap.add_argument("--top-features", type=int, default=3)
    args = ap.parse_args()

    model, cal, names = _load(args.model)
    x, _, feat_names, ids = build_feature_matrix(args.data)
    if feat_names != names:
        raise SystemExit("Feature columns of data do not match the trained model.")

    raw = model.predict_proba(x)
    raw = np.asarray(raw[:, 1] if raw.ndim == 2 else raw, dtype=float)
    proba = apply_calibrator(cal, raw)

    imp = _importances(model, names)
    col_mean = x.mean(axis=0)
    col_std = x.std(axis=0) + 1e-9

    records = []
    for i, cid in enumerate(ids):
        z = (x[i] - col_mean) / col_std        # how unusual each feature is for this chunk
        salience = np.abs(z) * imp             # important AND unusual
        top_idx = np.argsort(salience)[::-1][: args.top_features]
        records.append({
            "chunk_index": int(cid),
            "bot_probability": round(float(proba[i]), 4),
            "prediction": int(proba[i] >= 0.5),
            "confidence": _confidence(float(proba[i])),
            "top_features": [names[j] for j in top_idx],
        })

    Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    n_bot = sum(r["prediction"] for r in records)
    print(f"Scored {len(records)} chunks -> {args.out}")
    print(f"pred_bot={n_bot} ({n_bot/max(len(records),1):.1%}) | "
          f"score mean={proba.mean():.3f} std={proba.std():.3f} "
          f"range=[{proba.min():.3f},{proba.max():.3f}]")


if __name__ == "__main__":
    main()
