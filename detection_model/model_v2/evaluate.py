"""Evaluate a trained v2 model on a LABELED file (out-of-sample reward).

    python -m model_v2.evaluate --model artifacts/p44_v2_lgbm_canon.joblib \
        --data data/<labeled>.json

Reports the validator-aligned reward and its components at the 0.5 boundary, plus
a confusion matrix. Use a labeled file the model did NOT train on (e.g. a later
date) for a true generalization estimate. Evaluate the *canonical* model on
*canonical* labeled data so the distribution matches serving.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .calibrate import apply_calibrator
from .dataset import build_feature_matrix
from .metrics import format_metrics, reward_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="Labeled JSON (must contain is_bot).")
    ap.add_argument("--raw", action="store_true", help="Also report pre-calibration metrics.")
    args = ap.parse_args()

    import joblib
    art = joblib.load(Path(args.model).expanduser())
    model, cal, names = art["model"], art["calibrator"], art["feature_names"]

    x, y, feat_names, _ = build_feature_matrix(args.data)
    if y is None:
        raise SystemExit("Evaluation data has no labels (is_bot). Use a labeled file.")
    if feat_names != names:
        raise SystemExit("Feature columns of data do not match the trained model.")

    raw = model.predict_proba(x)
    raw = np.asarray(raw[:, 1] if raw.ndim == 2 else raw, dtype=float)
    proba = apply_calibrator(cal, raw)

    m = reward_metrics(y, proba)
    print(f"Eval file : {args.data}")
    print(f"chunks    : {len(y)} | bot={int(y.sum())} human={int((y == 0).sum())}")
    print(f"CALIBRATED: {format_metrics(m)}")
    if args.raw:
        print(f"RAW       : {format_metrics(reward_metrics(y, raw))}")

    pred = (np.round(np.clip(proba, 1e-6, 1 - 1e-6)) >= 1).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    print(f"confusion : TP={tp} FP={fp} TN={tn} FN={fn}  (boundary=0.5)")
    print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2))


if __name__ == "__main__":
    main()
