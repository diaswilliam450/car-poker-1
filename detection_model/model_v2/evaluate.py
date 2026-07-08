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

from .inference import Poker44V2Detector
from .metrics import format_metrics, reward_metrics
from .schema import load_chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="Labeled JSON (must contain is_bot).")
    ap.add_argument("--raw", action="store_true", help="Also report pre-calibration metrics.")
    args = ap.parse_args()

    # Detector handles both single-model and LGBM+TCN blend artifacts uniformly.
    det = Poker44V2Detector.load(args.model)
    chunks = load_chunks(args.data)
    labels = [c.label for c in chunks]
    if any(v is None for v in labels):
        raise SystemExit("Evaluation data has no labels (is_bot). Use a labeled file.")
    y = np.asarray([int(v) for v in labels], dtype=int)

    proba = np.asarray(det.predict_chunks([c.hands for c in chunks]), dtype=float)

    m = reward_metrics(y, proba)
    print(f"Eval file : {args.data}")
    print(f"model     : {args.model} | backend={det.metadata.get('backend')}"
          + (f" | blend={det.blend_weights}" if det.seq_model is not None else ""))
    print(f"chunks    : {len(y)} | bot={int(y.sum())} human={int((y == 0).sum())}")
    print(f"CALIBRATED: {format_metrics(m)}")
    if args.raw:
        raw = np.asarray(det.predict_chunks([c.hands for c in chunks], return_raw=True), dtype=float)
        print(f"RAW       : {format_metrics(reward_metrics(y, raw))}")

    pred = (np.round(np.clip(proba, 1e-6, 1 - 1e-6)) >= 1).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    print(f"confusion : TP={tp} FP={fp} TN={tn} FN={fn}  (boundary=0.5)")
    print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2))


if __name__ == "__main__":
    main()
