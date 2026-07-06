"""Train the v2 chunk classifier — LightGBM (fallback: sklearn HistGB) + isotonic.

    python -m model_v2.train --data data/<labeled>.json --out artifacts/p44_v2_lgbm.joblib

One row per chunk, order-invariant features, chunk-level train/val split. The raw
tree score is monotonically recalibrated with isotonic regression on the held-out
split so the 0.5 boundary is sensible and the score is usable as a probability.
Everything needed at serve time (model, calibrator, feature_names) is pickled.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .calibrate import apply_calibrator, fit_calibrator
from .dataset import build_feature_matrix, chunk_split
from .metrics import format_metrics, reward_metrics

try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:  # pragma: no cover
    _HAVE_LGBM = False
from sklearn.ensemble import HistGradientBoostingClassifier


def build_model(seed: int, n_pos: int, n_neg: int):
    if _HAVE_LGBM:
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            max_depth=-1, min_child_samples=20, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=seed, n_jobs=0, verbose=-1,
            class_weight="balanced" if abs(n_pos - n_neg) > 0.1 * (n_pos + n_neg) else None,
        )
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
        l2_regularization=2.0, random_state=seed,
    )


def _proba(model, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    return np.asarray(p[:, 1] if p.ndim == 2 else p, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Labeled benchmark JSON.")
    ap.add_argument("--out", default="artifacts/p44_v2_lgbm.joblib")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--target-fpr", type=float, default=0.04)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    args = ap.parse_args()

    x, y, names, _ = build_feature_matrix(args.data)
    if y is None:
        raise SystemExit("Training data has no labels (is_bot). Use a benchmark file.")
    print(f"Loaded {x.shape[0]} chunks, {x.shape[1]} features | bot={int(y.sum())} human={int((y==0).sum())}")

    tr, va = chunk_split(len(y), val_ratio=args.val_ratio, seed=args.seed)
    model = build_model(args.seed, int(y[tr].sum()), int((y[tr] == 0).sum()))
    model.fit(x[tr], y[tr])

    val_raw = _proba(model, x[va])
    # isotonic + cliff-aware boundary remap fit on the held-out split.
    cal = fit_calibrator(val_raw, y[va], target_fpr=args.target_fpr, max_fpr=args.max_fpr)
    print(f"Calibrator: boundary cut={cal['cut']:.4f} (target_fpr={args.target_fpr}, max_fpr={args.max_fpr})")

    val_cal = apply_calibrator(cal, val_raw)
    tr_cal = apply_calibrator(cal, _proba(model, x[tr]))
    print("Train :", format_metrics(reward_metrics(y[tr], tr_cal)))
    print("Val   :", format_metrics(reward_metrics(y[va], val_cal)))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(
        {
            "model": model,
            "calibrator": cal,
            "feature_names": names,
            "backend": "lightgbm" if _HAVE_LGBM else "sklearn_histgb",
            "val_metrics": reward_metrics(y[va], val_cal),
        },
        out,
    )
    print(f"Saved model -> {out}")
    print(json.dumps({k: round(v, 4) for k, v in reward_metrics(y[va], val_cal).items()}, indent=2))


if __name__ == "__main__":
    main()
