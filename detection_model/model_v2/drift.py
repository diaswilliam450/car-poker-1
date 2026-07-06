"""Train-vs-eval feature drift report (PSI / KS / JS).

    python -m model_v2.drift --train data/<benchmark>.json --eval data/chunks1.json

Because the eval set is unlabeled and distributionally different, we can't measure
accuracy there — but we *can* measure how far each feature has moved, so we know
which features to distrust. High-PSI features are the ones most likely to mislead
the model on the live feed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from .dataset import build_feature_matrix


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over quantile bins of the expected distribution."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _js(expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    e = np.histogram(expected, bins=edges)[0] + 1e-6
    a = np.histogram(actual, bins=edges)[0] + 1e-6
    return float(jensenshannon(e, a))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="feature_drift_report.csv")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    xt, _, names, _ = build_feature_matrix(args.train)
    xe, _, names_e, _ = build_feature_matrix(args.eval)
    assert names == names_e, "feature columns must match between train and eval"

    rows = []
    for j, name in enumerate(names):
        a, b = xt[:, j], xe[:, j]
        rows.append({
            "feature": name,
            "psi": round(_psi(a, b), 4),
            "ks": round(float(ks_2samp(a, b).statistic), 4),
            "js": round(_js(a, b), 4),
            "train_mean": round(float(a.mean()), 4),
            "eval_mean": round(float(b.mean()), 4),
        })
    rows.sort(key=lambda r: r["psi"], reverse=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out} ({len(rows)} features). PSI guide: <0.1 stable, 0.1-0.25 moderate, >0.25 large.")
    print(f"\nTop {args.top} drifting features (by PSI):")
    print(f"{'feature':40s} {'psi':>7} {'ks':>7} {'js':>7} {'train':>9} {'eval':>9}")
    for r in rows[: args.top]:
        print(f"{r['feature']:40s} {r['psi']:>7} {r['ks']:>7} {r['js']:>7} {r['train_mean']:>9} {r['eval_mean']:>9}")
    n_large = sum(1 for r in rows if r["psi"] > 0.25)
    print(f"\nLarge-drift features (PSI>0.25): {n_large}/{len(rows)}")


if __name__ == "__main__":
    main()
