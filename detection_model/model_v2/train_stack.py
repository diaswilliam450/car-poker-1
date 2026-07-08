"""A/B the TCN sequence model against the LightGBM — does stacking help?

    python -m model_v2.train_stack --data data/<canonical-labeled>.json

Builds out-of-fold (OOF) predictions for both base learners with StratifiedKFold,
then reports the validator reward for: LightGBM alone, TCN alone, a fixed 0.6/0.4
blend, and a logistic stack. The one number that matters is whether blend/stack
beats LGBM-alone — if not, the sequence model isn't earning its complexity.

Note: the cliff-aware calibrator is fit and evaluated on the same OOF scores, so
absolute rewards are mildly optimistic — but the procedure is identical for every
variant, so the *comparison* between them is fair. AP is calibration-free.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .calibrate import apply_calibrator, fit_calibrator
from .dataset import build_feature_matrix
from .metrics import format_metrics, reward_metrics
from .schema import load_chunks
from .sequence_model import TCNSequenceModel
from .train import _proba, build_model


def _report(tag: str, y: np.ndarray, scores: np.ndarray) -> dict:
    cal = fit_calibrator(scores, y)
    m = reward_metrics(y, apply_calibrator(cal, scores))
    print(f"  {tag:16s}: {format_metrics(m)}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Canonical labeled JSON.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out", default="", help="If set, refit on all data and save a deployable blend artifact.")
    ap.add_argument("--blend", type=float, default=0.6, help="LGBM weight in the blend (TCN gets 1-blend).")
    args = ap.parse_args()

    x, y, names, _ = build_feature_matrix(args.data)
    if y is None:
        raise SystemExit("Data has no labels.")
    chunks = [c.hands for c in load_chunks(args.data)]
    n = len(y)
    print(f"Loaded {n} chunks | bot={int(y.sum())} human={int((y==0).sum())} | features={x.shape[1]}")

    oof_lgbm = np.zeros(n)
    oof_seq = np.zeros(n)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(x, y), start=1):
        lg = build_model(args.seed, int(y[tr].sum()), int((y[tr] == 0).sum()))
        lg.fit(x[tr], y[tr])
        oof_lgbm[va] = _proba(lg, x[va])

        sq = TCNSequenceModel(seed=args.seed, epochs=args.epochs, verbose=False)
        sq.fit([chunks[i] for i in tr], y[tr])
        oof_seq[va] = sq.predict_proba([chunks[i] for i in va])[:, 1]
        print(f"fold {k}/{args.folds}: lgbm_ap={_ap(y[va], oof_lgbm[va]):.3f} tcn_ap={_ap(y[va], oof_seq[va]):.3f}")

    blend = 0.6 * oof_lgbm + 0.4 * oof_seq
    stacker = LogisticRegression(max_iter=1000)
    feats = np.column_stack([oof_lgbm, oof_seq])
    stacker.fit(feats, y)
    oof_stack = stacker.predict_proba(feats)[:, 1]

    print("\nOut-of-fold reward comparison (calibrated at the FPR ceiling):")
    m_lgbm = _report("LGBM alone", y, oof_lgbm)
    _report("TCN alone", y, oof_seq)
    m_blend = _report("blend 0.6/0.4", y, blend)
    m_stack = _report("logistic stack", y, oof_stack)

    print(f"\nstacker weights: lgbm={stacker.coef_[0][0]:+.3f} tcn={stacker.coef_[0][1]:+.3f}")
    best = max([("blend", m_blend["reward"]), ("stack", m_stack["reward"])], key=lambda t: t[1])
    delta = best[1] - m_lgbm["reward"]
    verdict = (
        f"{best[0]} beats LGBM-alone by {delta:+.4f} reward -> stacking helps"
        if delta > 1e-3 else
        f"no combo beats LGBM-alone (best {best[0]} {delta:+.4f}) -> ship the tabular model"
    )
    print(f"\nVERDICT: {verdict}")

    if args.out:
        from pathlib import Path
        import joblib
        w = float(args.blend)
        print(f"\nRefitting LGBM + TCN on all {n} chunks and saving blend (w_lgbm={w}, w_tcn={1-w:.2f})...")
        lgbm_full = build_model(args.seed, int(y.sum()), int((y == 0).sum()))
        lgbm_full.fit(x, y)
        seq_full = TCNSequenceModel(seed=args.seed, epochs=args.epochs, verbose=False).fit(chunks, y)
        # Calibrate on the (unbiased) OOF blend so the 0.5 boundary is honest.
        blend_oof = w * oof_lgbm + (1.0 - w) * oof_seq
        cal = fit_calibrator(blend_oof, y)
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "blend_v1",
                "lgbm_model": lgbm_full,
                "seq_model": seq_full,
                "feature_names": names,
                "blend_weights": [w, 1.0 - w],
                "calibrator": cal,
                "backend": "lgbm+tcn-blend",
                "val_metrics": reward_metrics(y, apply_calibrator(cal, blend_oof)),
            },
            out,
        )
        print(f"Saved blend artifact -> {out}")
        print(f"OOF blend (calibrated): {format_metrics(reward_metrics(y, apply_calibrator(cal, blend_oof)))}")


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if len(set(y.tolist())) > 1 else 0.0


if __name__ == "__main__":
    main()
