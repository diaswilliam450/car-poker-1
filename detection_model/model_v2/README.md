# model_v2 — feature-based chunk bot-detection

A from-scratch, tabular-first pipeline (see `../../poker_chunk_model_design_approach.md`).
Order-invariant, scale-robust features → LightGBM → cliff-aware calibration.

## Modules

| File | Role |
|---|---|
| `schema.py` | Normalize both file formats (labeled dicts / bare-list eval) → `Chunk`; sort actions by `action_id`. |
| `features.py` | 3-level features (action→hand→chunk): ratios, entropies, amount buckets, **anomaly rates**, set-level signatures. Permutation-invariant across hands. |
| `dataset.py` | Build the `(X, y, names, ids)` matrix; fixed column order. |
| `metrics.py` | Validator-aligned reward = `(0.65·AP + 0.35·recall)·(1−fpr)²`, 0.5 boundary, 0.10 cliff. |
| `calibrate.py` | Monotone isotonic + **boundary remap** that maximizes reward under an FPR ceiling. |
| `train.py` | LightGBM (fallback: sklearn HistGB) + chunk split + calibration. |
| `drift.py` | PSI / KS / JS train-vs-eval feature drift report. |
| `predict.py` | Inference on unlabeled eval → probability, decision, top features. |

## Usage

```bash
cd detection_model
# 1. Train (use CANONICAL data — matches the live/eval distribution)
python -m model_v2.train --data data/benchmark_chunks_canonical_2026-05-27_to_2026-07-01.json \
  --out artifacts/p44_v2_lgbm_canon.joblib
# 2. Drift check against the eval set
python -m model_v2.drift --train data/benchmark_chunks_canonical_2026-05-27_to_2026-07-01.json \
  --eval data/chunks1.json --out drift_canon_vs_eval.csv
# 3. Predict
python -m model_v2.predict --model artifacts/p44_v2_lgbm_canon.joblib \
  --data data/chunks1.json --out eval_predictions.json
```

## Key findings (measured)

- **Cliff-aware calibration is essential.** Plain isotonic left val FPR at 10.4% → reward 0. The boundary remap moved it under 5% → reward ~0.65–0.76.
- **Train on canonical, not raw.** Raw-trained model scored *every* chunks1 chunk as human (scores compressed to [0.23, 0.47]); the canonical-trained model produced a healthy [0.31, 0.69] spread with 8% flagged.
- **Residual gap = chunk size** (train ~35 hands vs eval ~85). Next step: window eval chunks to ~35 hands and average (see `predict.py` — add windowing) to close it.

## Notes

- Every feature is order-invariant across hands and scale-robust (ratios/buckets/entropy), so hand shuffling and stack-scale shifts don't move them.
- Validate on a **time-held-out canonical** slice, never on `chunks1` (unlabeled) or a random split.
