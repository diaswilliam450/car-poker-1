"""Cliff-aware, monotone score calibration for the v2 pipeline.

Two monotone stages, fit on the held-out split, serialized as plain lists:

    raw ──▶ isotonic ──▶ boundary remap (cut -> 0.5) ──▶ calibrated

* **isotonic** turns the tree score into a better-behaved probability (AP-invariant).
* **boundary remap** slides the decision boundary to the ``cut`` that *maximizes
  the validator reward subject to FPR < ``max_fpr``*. This is what keeps the
  chunk-level FPR under the 0.10 cliff — plain isotonic does not, and a model that
  ranks perfectly still scores 0 reward if its 0.5 boundary sits in the human tail.

All monotone, so ranking (and therefore average precision) is preserved exactly.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .metrics import reward_metrics


def _remap(scores: np.ndarray, cut: float) -> np.ndarray:
    """Monotone piecewise-linear map sending ``cut`` -> 0.5 (order preserved)."""
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    cut = min(max(float(cut), 1e-6), 1.0 - 1e-6)
    out = np.where(s < cut, (s / cut) * 0.5, 0.5 + ((s - cut) / (1.0 - cut)) * 0.5)
    return np.clip(out, 0.0, 1.0)


def apply_calibrator(cal: Dict, raw: np.ndarray) -> np.ndarray:
    grid = np.asarray(cal["grid"], dtype=float)
    iso_y = np.asarray(cal["iso_y"], dtype=float)
    iso = np.clip(np.interp(np.clip(raw, 0, 1), grid, iso_y), 0.0, 1.0)
    return _remap(iso, float(cal.get("cut", 0.5)))


def fit_calibrator(
    val_raw: np.ndarray,
    y_val: np.ndarray,
    *,
    target_fpr: float = 0.04,
    max_fpr: float = 0.05,
    identity_blend: float = 0.05,
) -> Dict:
    val_raw = np.asarray(val_raw, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_raw, y_val.astype(float))
    grid = np.linspace(0.0, 1.0, 256)
    iso_y = np.clip(iso.predict(grid), 0.0, 1.0)
    iso_y = (1.0 - identity_blend) * iso_y + identity_blend * grid  # strict monotone

    iso_val = np.clip(np.interp(val_raw, grid, iso_y), 0.0, 1.0)

    # Grid-search the boundary cut to maximize reward under the FPR ceiling.
    cands = np.unique(np.quantile(iso_val, np.linspace(0.40, 0.999, 80)))
    best_key = None
    best_cut = 0.5
    for c in cands:
        m = reward_metrics(y_val, _remap(iso_val, c))
        if m["fpr_at_0.5"] >= max_fpr - 1e-9:
            continue
        key = (m["reward"], m["recall_at_0.5"])
        if best_key is None or key > best_key:
            best_key = key
            best_cut = float(c)
    if best_key is None:  # nothing cleared the ceiling: fall back to conformal cut
        human = iso_val[y_val == 0]
        best_cut = float(np.quantile(human, 1.0 - target_fpr)) if human.size else 0.5

    return {"grid": grid.tolist(), "iso_y": iso_y.tolist(), "cut": best_cut}
