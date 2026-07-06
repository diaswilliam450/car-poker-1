"""Validator-aligned metrics (self-contained, no external imports).

The subnet reward is::

    base   = 0.65 * average_precision + 0.35 * bot_recall
    reward = base * (1 - fpr) ** 2         # hard 0 if fpr >= 0.10 cliff

Classification uses the validator's boundary: ``round(score)`` at 0.5. We report
AP (rank-based), recall/FPR at 0.5, and the full reward so tuning tracks the real
objective rather than accuracy or log-loss.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

FPR_CLIFF = 0.10


def reward_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    pred = (np.round(s) >= 1).astype(int)   # validator 0.5 boundary (round-half-to-even)

    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    recall = tp / pos
    fpr = fp / neg
    ap = float(average_precision_score(y, s)) if len(set(y.tolist())) > 1 else 0.0
    base = 0.65 * ap + 0.35 * recall
    reward = 0.0 if fpr >= FPR_CLIFF else base * (1.0 - fpr) ** 2

    out = {
        "ap": ap,
        "roc_auc": float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else 0.0,
        "log_loss": float(log_loss(y, s, labels=[0, 1])) if len(set(y.tolist())) > 1 else 0.0,
        "recall_at_0.5": recall,
        "fpr_at_0.5": fpr,
        "reward": float(reward),
        "human_prob_max": float(s[y == 0].max()) if (y == 0).any() else 0.0,
        "bot_prob_min": float(s[y == 1].min()) if (y == 1).any() else 1.0,
    }
    return out


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"reward={m['reward']:.4f} ap={m['ap']:.4f} recall={m['recall_at_0.5']:.4f} "
        f"fpr={m['fpr_at_0.5']:.4f} auc={m['roc_auc']:.4f} logloss={m['log_loss']:.4f}"
    )
