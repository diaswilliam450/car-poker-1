"""OOF-stacked ensemble head for the Poker44 detector.

This is the ranking engine ported from the reference (competitor) stacked model.
Instead of a single XGBoost head on ``concat(neural_embedding, features)``, it
trains a *diverse* set of tree base learners and combines their **out-of-fold**
probabilities with a logistic-regression meta-learner:

    base learners  ──(OOF probs)──▶  logistic meta-learner  ──▶  P(bot)

Why this helps the reward:

* AP (ranking) is 65% of the validator reward and is the *one* thing the
  reward-aware :class:`ScoreCalibrator` cannot recover after the fact (it is
  monotone). Ensemble diversity — gradient-boosted (XGBoost/LightGBM/CatBoost)
  plus bagged (ExtraTrees/RandomForest) learners — reliably lifts AP over any
  single learner, especially on a small, noisy benchmark.
* Out-of-fold stacking means the meta-learner is trained on predictions the base
  learners did *not* see, so it does not simply rubber-stamp an over-fit base.

The ensemble exposes ``predict_proba`` returning an ``[n, 2]`` matrix, so it is a
drop-in replacement for the previous ``XGBClassifier`` everywhere the codebase
already calls ``predict_proba``. The fitted object is plain Python/sklearn/
booster state and is serialized inside the ``.pt`` artifact via ``torch.save``.

LightGBM and CatBoost are optional: if they are not installed they are silently
skipped, so the ensemble always runs with at least XGBoost + ExtraTrees +
RandomForest.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

try:  # optional booster
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore[assignment]

try:  # optional booster
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore[assignment]


BaseFactory = Callable[[], Any]


def _proba1(estimator: Any, x: np.ndarray) -> np.ndarray:
    """Positive-class probability as a 1-D float array, robust to estimator type."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(x))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float32)
        return proba.reshape(-1).astype(np.float32)
    raw = np.asarray(estimator.predict(x), dtype=np.float32).reshape(-1)
    if raw.min(initial=0.0) < 0.0 or raw.max(initial=1.0) > 1.0:
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw.astype(np.float32)


class StackedEnsemble:
    """Diverse tree base learners + OOF logistic meta-learner.

    Parameters
    ----------
    scale_pos_weight:
        Positive-class weight for the boosters (``neg / pos``), mirroring the
        single-XGBoost head so bots stay correctly ranked.
    n_folds:
        Folds for out-of-fold meta-feature generation.
    top_k:
        If set, keep only the ``top_k`` most important input columns (ranked by a
        quick XGBoost importance pass). ``None`` keeps all columns.
    use_lightgbm / use_catboost:
        Include these boosters when their library is importable.
    """

    def __init__(
        self,
        *,
        scale_pos_weight: float = 1.0,
        n_folds: int = 5,
        top_k: Optional[int] = None,
        use_lightgbm: bool = True,
        use_catboost: bool = True,
        meta_c: float = 1.0,
        random_state: int = 44,
        n_jobs: int = 0,
        xgb_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scale_pos_weight = float(scale_pos_weight)
        self.n_folds = int(n_folds)
        self.top_k = int(top_k) if top_k else None
        self.use_lightgbm = bool(use_lightgbm)
        self.use_catboost = bool(use_catboost)
        self.meta_c = float(meta_c)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs) if n_jobs else -1

        # Fitted state.
        self.base_models_: List[Tuple[str, Any]] = []
        self.meta_: Optional[LogisticRegression] = None
        self.selected_idx_: Optional[np.ndarray] = None
        self.base_names_: List[str] = []

    # ------------------------------------------------------------------ specs

    def _base_specs(self) -> List[Tuple[str, BaseFactory]]:
        spw = max(self.scale_pos_weight, 1e-6)
        specs: List[Tuple[str, BaseFactory]] = [
            (
                "xgboost",
                lambda: XGBClassifier(
                    n_estimators=600, max_depth=3, learning_rate=0.03,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
                    objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", random_state=self.random_state,
                    n_jobs=self.n_jobs, scale_pos_weight=spw,
                ),
            ),
            (
                "extratrees",
                lambda: ExtraTreesClassifier(
                    n_estimators=400, max_depth=None, min_samples_leaf=2,
                    class_weight="balanced", random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
            ),
            (
                "randomforest",
                lambda: RandomForestClassifier(
                    n_estimators=400, max_depth=None, min_samples_leaf=2,
                    class_weight="balanced", random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
            ),
        ]
        if self.use_lightgbm and LGBMClassifier is not None:
            specs.append((
                "lightgbm",
                lambda: LGBMClassifier(
                    n_estimators=600, max_depth=-1, num_leaves=31,
                    learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
                    reg_lambda=2.0, random_state=self.random_state,
                    n_jobs=self.n_jobs, scale_pos_weight=spw, verbosity=-1,
                ),
            ))
        if self.use_catboost and CatBoostClassifier is not None:
            specs.append((
                "catboost",
                lambda: CatBoostClassifier(
                    iterations=600, depth=4, learning_rate=0.03,
                    l2_leaf_reg=3.0, random_seed=self.random_state,
                    scale_pos_weight=spw, verbose=False, allow_writing_files=False,
                ),
            ))
        return specs

    # ------------------------------------------------------------------- fit

    def _select_features(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.top_k or self.top_k >= x.shape[1]:
            self.selected_idx_ = None
            return x
        ranker = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist",
            random_state=self.random_state, n_jobs=self.n_jobs,
            scale_pos_weight=max(self.scale_pos_weight, 1e-6),
        )
        ranker.fit(x, y)
        importances = np.asarray(ranker.feature_importances_, dtype=np.float64)
        order = np.argsort(importances)[::-1][: self.top_k]
        self.selected_idx_ = np.sort(order).astype(np.int64)
        return x[:, self.selected_idx_]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StackedEnsemble":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Bad shapes: x={x.shape}, y={y.shape}")

        x_sel = self._select_features(x, y)
        specs = self._base_specs()
        self.base_names_ = [name for name, _ in specs]

        # Keep folds feasible for the smaller class on tiny benchmarks.
        min_class = int(min(np.bincount(y, minlength=2)[:2]))
        n_folds = max(2, min(self.n_folds, min_class)) if min_class >= 2 else 2

        oof = np.zeros((x_sel.shape[0], len(specs)), dtype=np.float32)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        for tr_idx, va_idx in skf.split(x_sel, y):
            for j, (_, factory) in enumerate(specs):
                est = factory()
                est.fit(x_sel[tr_idx], y[tr_idx])
                oof[va_idx, j] = _proba1(est, x_sel[va_idx])

        # Meta-learner on out-of-fold base predictions.
        self.meta_ = LogisticRegression(
            C=self.meta_c, class_weight="balanced", max_iter=1000,
            random_state=self.random_state,
        )
        self.meta_.fit(oof, y)

        # Refit each base learner on the full training matrix for inference.
        self.base_models_ = [(name, factory().fit(x_sel, y)) for name, factory in specs]
        return self

    # --------------------------------------------------------------- predict

    def _stack(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.selected_idx_ is not None:
            x = x[:, self.selected_idx_]
        cols = [_proba1(model, x) for _, model in self.base_models_]
        return np.column_stack(cols).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.meta_ is None or not self.base_models_:
            raise RuntimeError("StackedEnsemble is not fitted.")
        return np.asarray(self.meta_.predict_proba(self._stack(x)), dtype=np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.int32)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # sklearn-style introspection
        return {
            "scale_pos_weight": self.scale_pos_weight,
            "n_folds": self.n_folds,
            "top_k": self.top_k,
            "use_lightgbm": self.use_lightgbm,
            "use_catboost": self.use_catboost,
            "meta_c": self.meta_c,
            "random_state": self.random_state,
            "base_learners": list(self.base_names_),
        }
