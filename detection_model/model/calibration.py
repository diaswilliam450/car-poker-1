"""Reward-aware score calibration for the Poker44 detector.

The neural+XGBoost head produces a raw bot-risk score per chunk. That score is
good at *ranking* but its absolute scale is arbitrary: the decision boundary the
validator uses (``round(score)`` at 0.5) may sit in the wrong place, and a few
humans drifting above 0.5 can trip the 10% FPR cliff that zeroes the reward.

:class:`ScoreCalibrator` is a small, serializable post-processor that fixes the
score *geometry* without retraining the model. It applies up to four monotone
stages, in this fixed order::

    raw ──▶ quantile spread ──▶ isotonic ──▶ threshold_logit remap ──▶ logit shift ──▶ calibrated

* **quantile spread** (optional): a monotone empirical-CDF spreader (blended
  with the identity) borrowed from the heterogeneous-stack miner. Its sole job
  is *anti-collapse*: when the head emits all scores in a narrow band (e.g. an
  out-of-distribution chunk length squashing the embedding), the raw scores
  carry rank information but no usable spread, and every downstream boundary
  lands on the wrong side. Re-expanding the band toward uniform restores the
  separation the later stages need. Monotone, so AP is untouched in-distribution;
  it is insurance that pays off only when a band collapses.
* **isotonic** (optional): a monotone recalibration fit on held-out scores so
  the score is a better-behaved probability for average precision.
* **threshold_logit remap**: ``sigmoid((s - threshold) / temperature)``. Recenters
  the decision boundary so the chosen ``threshold`` lands exactly on 0.5. This is
  what moves humans below 0.5 and bots above it.
* **logit shift**: ``sigmoid((logit(s) + bias) / temperature)``. A final nudge in
  logit space for fine FPR control and recall trade-off.

Every parameter is chosen by :meth:`fit` on a held-out calibration split by
**maximizing the validator reward objective subject to a hard FPR ceiling**
(default: keep FPR < 0.05, comfortably under the 0.10 cliff). All stages are
monotone, so ranking — and therefore average precision — is preserved exactly.

Serialize with :meth:`to_dict` (embed in the ``.pt`` artifact) and restore with
:meth:`from_dict` at inference time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scoring import VALIDATOR_FPR_CLIFF, reward_metrics, validator_reward

try:  # isotonic is optional; the remap+logit stages work without it.
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore[assignment]


def _clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_quantile_spread(
    scores: np.ndarray,
    *,
    blend: float = 0.9,
    n_knots: int = 256,
) -> Optional[Tuple[List[float], List[float]]]:
    """Fit a monotone empirical-CDF spreader on a fixed [0, 1] grid.

    Returns ``(grid, y)`` knot lists for ``np.interp`` (same storage pattern as
    isotonic), or ``None`` when the input has no spread to learn from. The map is
    ``y = blend * empiricalCDF(grid) + (1 - blend) * grid``: at ``blend=1`` it is
    the pure rank transform (output ~uniform), at ``blend=0`` it is the identity.
    Strictly monotone by construction, so it never reorders scores.
    """
    raw = np.asarray(scores, dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size < 8 or float(np.ptp(raw)) < 1e-9:
        return None  # constant / degenerate band: nothing to spread
    blend = float(np.clip(blend, 0.0, 1.0))
    n_knots = int(max(8, n_knots))
    grid = np.linspace(0.0, 1.0, n_knots)
    sorted_raw = np.sort(np.clip(raw, 0.0, 1.0))
    # Empirical CDF of the calibration scores evaluated on the grid -> rank in [0, 1].
    cdf = np.searchsorted(sorted_raw, grid, side="right") / float(sorted_raw.size)
    y = blend * cdf + (1.0 - blend) * grid
    # Guarantee strict monotonicity for np.interp stability.
    y = np.maximum.accumulate(y)
    y = y + np.linspace(0.0, 1e-6, n_knots)
    y = _clamp01(y)
    return grid.tolist(), y.tolist()


def _threshold_logit(scores: np.ndarray, threshold: float, temperature: float) -> np.ndarray:
    """Map ``threshold`` to 0.5 with a sigmoid of slope ``1 / temperature``."""
    temperature = max(float(temperature), 1e-6)
    return _sigmoid((np.clip(scores, 1e-6, 1.0 - 1e-6) - float(threshold)) / temperature)


def _logit_shift(scores: np.ndarray, bias: float, temperature: float) -> np.ndarray:
    if abs(float(bias)) < 1e-12 and abs(float(temperature) - 1.0) < 1e-12:
        return _clamp01(scores)
    temperature = max(float(temperature), 1e-6)
    return _sigmoid((_logit(scores) + float(bias)) / temperature)


def _objective_key(metrics: Dict[str, float], objective: str) -> Tuple[float, float, float]:
    """Lexicographic sort key for the calibration grid search.

    ``reward`` (recommended default) ranks by the full validator reward. This is
    the correct objective for *post-hoc calibration*: every stage here is
    monotone, so average precision is invariant — calibration can only move
    recall and FPR, which is exactly what the reward's penalty term trades off.
    ``ap_first`` and ``recall`` are offered for experimentation but degenerate
    on the (rank-invariant) AP axis.
    """
    ap = float(metrics.get("validator_ap_score", 0.0))
    recall = float(metrics.get("validator_bot_recall", 0.0))
    reward = float(metrics.get("validator_reward", 0.0))
    if objective == "reward":
        return (reward, ap, recall)
    if objective == "recall":
        return (recall, reward, ap)
    return (ap, recall, reward)  # ap_first (default)


def _passes_fpr_constraint(metrics: Dict[str, float], max_fpr: float) -> bool:
    """Reject any candidate whose held-out FPR is unsafe.

    The constraint is the reward-relevant one: chunk-level FPR strictly below
    ``max_fpr``, which is itself set well under the validator's 0.10 cliff so
    live distribution drift has margin. We deliberately do *not* require the
    single worst human to clear 0.5 — that would force the boundary above the
    entire human tail and needlessly destroy bot recall.
    """
    return metrics.get("validator_fpr", 1.0) < max_fpr - 1e-9


def conformal_bias_for_target_fpr(
    human_scores: np.ndarray,
    target_fpr: float,
    *,
    max_abs_bias: float = 5.0,
) -> float:
    """Logit bias that drops the human-score ``1 - target_fpr`` quantile to ~0.5.

    Picks the shift that would put only ``target_fpr`` of humans above the 0.5
    boundary on the calibration set. Clipped to ``+-max_abs_bias`` because huge
    biases signal a collapsed score distribution and generalize poorly.
    """
    if human_scores.size == 0:
        return 0.0
    target_fpr = float(min(max(target_fpr, 1e-4), 0.5))
    quantile = float(np.quantile(human_scores, 1.0 - target_fpr))
    quantile = min(max(quantile, 1e-6), 1.0 - 1e-6)
    bias = -float(np.log(quantile / (1.0 - quantile)))
    return float(max(-abs(max_abs_bias), min(abs(max_abs_bias), bias)))


class ScoreCalibrator:
    """Monotone, reward-aware score post-processor (see module docstring)."""

    def __init__(
        self,
        *,
        spread_x: Optional[List[float]] = None,
        spread_y: Optional[List[float]] = None,
        isotonic_x: Optional[List[float]] = None,
        isotonic_y: Optional[List[float]] = None,
        remap: Optional[Dict[str, float]] = None,
        logit_bias: float = 0.0,
        logit_temperature: float = 1.0,
        objective: str = "reward",
        target_fpr: float = 0.04,
        max_fpr: float = 0.05,
    ) -> None:
        self.spread_x = spread_x
        self.spread_y = spread_y
        self.isotonic_x = isotonic_x
        self.isotonic_y = isotonic_y
        self.remap = remap or {}
        self.logit_bias = float(logit_bias)
        self.logit_temperature = float(logit_temperature)
        self.objective = str(objective)
        self.target_fpr = float(target_fpr)
        self.max_fpr = float(max_fpr)

    # ------------------------------------------------------------------ apply

    def _apply_spread(self, scores: np.ndarray) -> np.ndarray:
        if not self.spread_x or not self.spread_y:
            return _clamp01(scores)
        xp = np.asarray(self.spread_x, dtype=float)
        fp = np.asarray(self.spread_y, dtype=float)
        return _clamp01(np.interp(np.clip(scores, 0.0, 1.0), xp, fp))

    def _apply_isotonic(self, scores: np.ndarray) -> np.ndarray:
        if not self.isotonic_x or not self.isotonic_y:
            return _clamp01(scores)
        xp = np.asarray(self.isotonic_x, dtype=float)
        fp = np.asarray(self.isotonic_y, dtype=float)
        return _clamp01(np.interp(np.clip(scores, 0.0, 1.0), xp, fp))

    def _apply_remap(self, scores: np.ndarray) -> np.ndarray:
        if not self.remap:
            return _clamp01(scores)
        return _clamp01(
            _threshold_logit(
                scores,
                threshold=float(self.remap.get("threshold", 0.5)),
                temperature=float(self.remap.get("temperature", 0.25)),
            )
        )

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        """Apply quantile spread -> isotonic -> remap -> logit shift, in order."""
        out = np.asarray(scores, dtype=float)
        out = self._apply_spread(out)
        out = self._apply_isotonic(out)
        out = self._apply_remap(out)
        out = _logit_shift(out, self.logit_bias, self.logit_temperature)
        return _clamp01(out)

    @property
    def is_identity(self) -> bool:
        return (
            not self.spread_x
            and not self.isotonic_x
            and not self.remap
            and abs(self.logit_bias) < 1e-12
            and abs(self.logit_temperature - 1.0) < 1e-12
        )

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        *,
        use_spread: bool = True,
        spread_blend: float = 0.9,
        spread_knots: int = 256,
        use_isotonic: bool = True,
        isotonic_identity_blend: float = 0.05,
        remap_temperature_grid: Sequence[float] = (0.12, 0.18, 0.25, 0.35, 0.5, 0.65, 0.85, 1.0, 1.25),
        logit_bias_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        logit_temperature_grid: Sequence[float] = (0.6, 0.8, 1.0, 1.2),
    ) -> "ScoreCalibrator":
        """Tune all stages on a held-out calibration split.

        Stage 1 fits an optional isotonic recalibration. Stage 2 grid-searches
        the threshold-logit remap (recenters the boundary). Stage 3 grid-searches
        the logit shift (fine FPR/recall control). Each stage keeps the candidate
        that maximizes the objective subject to the FPR ceiling.
        """
        raw = np.asarray(scores, dtype=float)
        lab = np.asarray(labels, dtype=int)
        if raw.size == 0 or lab.sum() == 0 or lab.sum() == lab.size:
            return self  # need both classes to calibrate; leave as identity

        # Stage 0: quantile spread (monotone anti-collapse). Fit on the raw
        # calibration scores; all later stages then operate on the spread output.
        # Because the map is monotone, ranking/AP is unchanged and the remap below
        # re-optimizes the boundary, so held-out reward is never hurt -- the spread
        # only earns its keep when a live score band collapses.
        self.spread_x = self.spread_y = None
        if use_spread:
            knots = fit_quantile_spread(raw, blend=spread_blend, n_knots=spread_knots)
            if knots is not None:
                self.spread_x, self.spread_y = knots
        base = self._apply_spread(raw)

        # Stage 1: isotonic recalibration (monotone, preserves ranking).
        self.isotonic_x = self.isotonic_y = None
        if use_isotonic and IsotonicRegression is not None and len(set(lab.tolist())) >= 2:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(base, lab.astype(float))
            grid = np.linspace(0.0, 1.0, 256)
            iso_y = np.clip(iso.predict(grid), 0.0, 1.0)
            # Guarantee STRICT monotonicity. Isotonic fit on a (near-)perfectly
            # separated split degenerates into a step function with long flat
            # regions. A flat segment is monotone but maps an entire score
            # interval to a single value, so on live data where most chunks fall
            # in that interval every calibrated score collapses to a constant —
            # ranking, and therefore AP, is destroyed even though val AP looked
            # perfect. Blending a small slice of the identity keeps the curve
            # strictly increasing and preserves within-region ranking.
            blend = float(np.clip(isotonic_identity_blend, 0.0, 1.0))
            iso_y = (1.0 - blend) * iso_y + blend * grid
            self.isotonic_x = grid.tolist()
            self.isotonic_y = iso_y.tolist()
        post_iso = self._apply_isotonic(base)

        # Stage 2: threshold-logit remap. Candidate thresholds are drawn from the
        # human/bot score quantiles plus a few fixed anchors, so the boundary is
        # placed in the gap between the populations.
        humans = post_iso[lab == 0]
        bots = post_iso[lab == 1]
        thresholds: set[float] = set()
        for q in np.linspace(0.40, 0.995, 24):
            thresholds.add(float(np.quantile(humans, q)))
        for q in np.linspace(0.005, 0.60, 20):
            thresholds.add(float(np.quantile(bots, q)))
        thresholds.update({0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30})

        # Seed the incumbent from the no-remap baseline ONLY if it already meets
        # the FPR ceiling; otherwise start from "nothing valid yet" so a safe
        # (lower-recall) candidate can win instead of being blocked by an unsafe
        # baseline.
        self.remap = {}
        baseline = reward_metrics(lab, post_iso)
        best_key = (
            _objective_key(baseline, self.objective)
            if _passes_fpr_constraint(baseline, self.max_fpr)
            else None
        )
        for threshold in sorted(thresholds):
            for temperature in remap_temperature_grid:
                remapped = _threshold_logit(post_iso, threshold, temperature)
                metrics = reward_metrics(lab, remapped)
                if not _passes_fpr_constraint(metrics, self.max_fpr):
                    continue
                key = _objective_key(metrics, self.objective)
                if best_key is None or key > best_key:
                    best_key = key
                    self.remap = {"threshold": float(threshold), "temperature": float(temperature)}
        post_remap = self._apply_remap(post_iso)

        # Stage 3: logit shift. Seed the search with the conformal bias that
        # targets the desired FPR, then refine over the grid.
        conformal = conformal_bias_for_target_fpr(post_remap[lab == 0], self.target_fpr)
        bias_candidates = sorted(
            {float(b) for b in logit_bias_grid}
            | {conformal}
            | {conformal + d for d in (0.0, 0.25, 0.5, 1.0, 1.5)}
        )
        self.logit_bias, self.logit_temperature = 0.0, 1.0
        baseline = reward_metrics(lab, post_remap)
        best_key = (
            _objective_key(baseline, self.objective)
            if _passes_fpr_constraint(baseline, self.max_fpr)
            else None
        )
        for bias in bias_candidates:
            for temperature in logit_temperature_grid:
                if abs(bias) < 1e-12 and abs(temperature - 1.0) < 1e-12:
                    continue
                shifted = _logit_shift(post_remap, bias, temperature)
                metrics = reward_metrics(lab, shifted)
                if not _passes_fpr_constraint(metrics, self.max_fpr):
                    continue
                key = _objective_key(metrics, self.objective)
                if best_key is None or key > best_key:
                    best_key = key
                    self.logit_bias, self.logit_temperature = float(bias), float(temperature)
        return self

    # --------------------------------------------------------------- (de)serialize

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "score_calibrator_v1",
            "spread_x": self.spread_x,
            "spread_y": self.spread_y,
            "isotonic_x": self.isotonic_x,
            "isotonic_y": self.isotonic_y,
            "remap": dict(self.remap),
            "logit_bias": self.logit_bias,
            "logit_temperature": self.logit_temperature,
            "objective": self.objective,
            "target_fpr": self.target_fpr,
            "max_fpr": self.max_fpr,
        }

    @classmethod
    def from_dict(cls, state: Optional[Dict[str, Any]]) -> Optional["ScoreCalibrator"]:
        if not state or state.get("kind") != "score_calibrator_v1":
            return None
        return cls(
            spread_x=state.get("spread_x"),
            spread_y=state.get("spread_y"),
            isotonic_x=state.get("isotonic_x"),
            isotonic_y=state.get("isotonic_y"),
            remap=state.get("remap"),
            logit_bias=float(state.get("logit_bias", 0.0)),
            logit_temperature=float(state.get("logit_temperature", 1.0)),
            objective=str(state.get("objective", "ap_first")),
            target_fpr=float(state.get("target_fpr", 0.04)),
            max_fpr=float(state.get("max_fpr", 0.05)),
        )

    def summary(self, labels: Sequence[int], raw_scores: Sequence[float]) -> str:
        """Before/after reward summary, for logging at the end of training."""
        before = reward_metrics(labels, raw_scores)
        after = reward_metrics(labels, self.transform(raw_scores))
        reward_before, _ = validator_reward(np.asarray(raw_scores, float), np.asarray(labels, int))
        reward_after, _ = validator_reward(
            self.transform(raw_scores), np.asarray(labels, int)
        )
        return (
            f"spread={'on' if self.spread_x else 'off'} "
            f"remap={self.remap or None} logit_bias={self.logit_bias:.4f} "
            f"logit_temp={self.logit_temperature:.4f} isotonic={'on' if self.isotonic_x else 'off'} | "
            f"reward {reward_before:.4f}->{reward_after:.4f} "
            f"fpr {before['validator_fpr']:.4f}->{after['validator_fpr']:.4f} "
            f"recall {before['validator_bot_recall']:.4f}->{after['validator_bot_recall']:.4f} "
            f"human_prob_max {before['human_prob_max']:.4f}->{after['human_prob_max']:.4f}"
        )
