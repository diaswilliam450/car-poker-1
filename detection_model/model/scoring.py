"""Validator-reward mirror and reward-aware metrics for Poker44 training.

Everything here exists to make one number visible during training, evaluation
and calibration: *the reward the subnet validator will actually pay you.*

The live validator (``poker44/score/scoring.py``) scores a window of
``(prediction, label)`` pairs like this::

    preds      = round(scores)                         # 0.5 rounds DOWN to 0
    fpr        = false_positives / negatives           # humans flagged as bot
    bot_recall = true_positives  / positives
    ap_score   = average_precision_score(labels, scores)

    human_safety_penalty = (1 - fpr) ** 2
    if fpr >= 0.10:                                     # the hard safety cliff
        human_safety_penalty = 0.0

    base_score = 0.65 * ap_score + 0.35 * bot_recall
    reward     = base_score * human_safety_penalty

Two facts drive every training and calibration decision in this repo:

1. **Ranking dominates.** Average precision is 65% of ``base_score``. The model
   is paid far more for *ordering* bots above humans than for any single
   threshold call. Optimize AP first.
2. **The 10% FPR cliff is absolute.** One human in ten flagged as a bot zeroes
   the entire reward, regardless of recall. Calibration must hold FPR well under
   0.10 with margin (we target ~0.04 and reject configs at/above ~0.05).

This module is intentionally dependency-light (numpy + scikit-learn) so the
``detection_model`` package stays portable and does not import the validator.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

# The validator's hard human-safety cliff. At or above this chunk-level FPR the
# reward is zeroed, so calibration always keeps a safety margin below it.
VALIDATOR_FPR_CLIFF = 0.10

# Reward weighting (must match poker44/score/scoring.py).
AP_WEIGHT = 0.65
RECALL_WEIGHT = 0.35


def validator_reward(
    scores: np.ndarray,
    labels: np.ndarray,
    boundary: Optional[float] = None,
) -> tuple[float, Dict[str, float]]:
    """Reproduce the on-chain validator reward.

    ``scores`` are bot-risk probabilities in ``[0, 1]`` (higher = more bot-like).
    ``labels`` are 0 (human) / 1 (bot). Returns ``(reward, details)`` where
    ``details`` carries the component breakdown for diagnostics.

    ``boundary`` controls how scores become predictions. ``None`` (default)
    reproduces the validator EXACTLY (``np.round``, so 0.5 rounds down to human).
    A float classifies ``score >= boundary`` as bot, which moves FPR and recall
    (and therefore the reward) — a *what-if* for boundary placement. AP is
    rank-based and unaffected. On-chain the validator always uses 0.5, so a custom
    boundary answers "what reward would I get if the cut were here" — i.e. what a
    calibrator that maps this boundary to 0.5 would bank.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if boundary is None:
        preds = np.round(scores).astype(int)
    else:
        preds = (scores >= float(boundary)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    negatives = max(tn + fp, 1)
    positives = max(tp + fn, 1)
    fpr = fp / negatives
    bot_recall = tp / positives

    if scores.size and np.any(labels == 1):
        ap_score = float(average_precision_score(labels, scores))
    else:
        ap_score = 0.0

    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2
    if fpr >= VALIDATOR_FPR_CLIFF:
        human_safety_penalty = 0.0

    base_score = AP_WEIGHT * ap_score + RECALL_WEIGHT * bot_recall
    reward = base_score * human_safety_penalty

    return reward, {
        "fpr": float(fpr),
        "bot_recall": float(bot_recall),
        "ap_score": float(ap_score),
        "human_safety_penalty": float(human_safety_penalty),
        "base_score": float(base_score),
        "reward": float(reward),
    }


def reward_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    boundary: Optional[float] = None,
) -> Dict[str, float]:
    """Full reward-aware metric bundle for one set of scored chunks.

    Combines the validator reward with the score-separation diagnostics that
    explain *why* a reward is high or low: where humans top out, where bots
    bottom out, and how wide the gap at the 0.5 decision boundary is.

    ``boundary`` is forwarded to :func:`validator_reward` (``None`` = the exact
    on-chain 0.5 rule); set it to simulate the reward at a different cut.
    """
    labels = [int(v) for v in labels]
    safe = [float(max(0.0, min(1.0, v))) for v in scores]
    arr = np.asarray(safe, dtype=float)
    lab = np.asarray(labels, dtype=int)

    _, details = validator_reward(arr, lab, boundary=boundary)
    metrics: Dict[str, float] = {
        "validator_reward": details["reward"],
        "validator_fpr": details["fpr"],
        "validator_bot_recall": details["bot_recall"],
        "validator_ap_score": details["ap_score"],
        "validator_base_score": details["base_score"],
        "human_safety_penalty": details["human_safety_penalty"],
    }

    humans = arr[lab == 0]
    bots = arr[lab == 1]
    # The two numbers that decide whether you trip the FPR cliff: the worst
    # (highest) human score and the worst (lowest) bot score.
    metrics["human_prob_max"] = float(humans.max()) if humans.size else 0.0
    metrics["bot_prob_min"] = float(bots.min()) if bots.size else 1.0
    metrics["score_gap_at_0_5"] = metrics["bot_prob_min"] - metrics["human_prob_max"]
    metrics["prob_min"] = float(arr.min()) if arr.size else 0.0
    metrics["prob_max"] = float(arr.max()) if arr.size else 0.0
    metrics["prob_mean"] = float(arr.mean()) if arr.size else 0.0
    return metrics


def format_reward_line(metrics: Dict[str, float]) -> str:
    """One-line human-readable summary of a :func:`reward_metrics` bundle."""
    return (
        f"reward={metrics.get('validator_reward', 0.0):.4f} "
        f"ap={metrics.get('validator_ap_score', 0.0):.4f} "
        f"recall={metrics.get('validator_bot_recall', 0.0):.4f} "
        f"fpr={metrics.get('validator_fpr', 0.0):.4f} "
        f"human_prob_max={metrics.get('human_prob_max', 0.0):.4f} "
        f"bot_prob_min={metrics.get('bot_prob_min', 0.0):.4f} "
        f"gap={metrics.get('score_gap_at_0_5', 0.0):.4f}"
    )


def print_reward_diagnostics(
    title: str,
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    indent: str = "  ",
) -> Dict[str, float]:
    """Compute and print a reward bundle in one call (training-loop friendly)."""
    metrics = reward_metrics(labels, scores)
    print(f"{indent}{title}: {format_reward_line(metrics)}")
    return metrics
