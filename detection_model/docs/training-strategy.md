# Poker44 Detection Model — Training Strategy

This guide explains **how to train the detector to maximize the subnet reward**,
every hyperparameter that matters, and the single recommended recipe. It assumes
the updated pipeline: hierarchical encoder → XGBoost head → **reward-aware
calibrator** (`model/calibration.py`), with reward diagnostics from
`model/scoring.py`.

---

## 1. What you are actually optimizing

The validator pays this, per scored window of `(score, label)` pairs:

```
fpr        = false_positives / negatives           # humans flagged as bot
bot_recall = true_positives  / positives
ap_score   = average_precision(labels, scores)

human_safety_penalty = (1 - fpr) ** 2
if fpr >= 0.10:  human_safety_penalty = 0.0        # HARD CLIFF
base_score = 0.65 * ap_score + 0.35 * bot_recall
reward     = base_score * human_safety_penalty
```

Three consequences drive every decision below:

1. **Ranking is 65% of the score.** Average precision depends only on the
   *order* of scores, not their absolute value. Train the model to separate bots
   from humans in rank.
2. **The 10% FPR cliff is absolute.** Flag 1 human in 10 as a bot and the reward
   is **zero**, regardless of recall. Calibrate to keep FPR well under 0.10.
3. **Calibration cannot change AP.** Every calibration stage is monotone, so it
   preserves ranking. It can only move recall and FPR. Therefore:
   - the **model** is responsible for AP (separation/ranking);
   - the **calibrator** is responsible for banking that AP safely (recall under
     the FPR cliff).

This split is the whole strategy: **train for ranking, calibrate for safety.**

---

## 2. The pipeline (what each stage does)

```
chunk (list of hands)
  │  ActionVectorizer (v3): 8 categorical channels
  │     [street, action_type, seat, amount_bucket, pot_flow, first_in_street,
  │      actor_role, street_position]
  │     + numeric block; per-hand meta (stack/actors/streets/hero) + hand_end
  ▼
Action Transformer + attention pool  ──▶ hand embedding         ─┐
  + per-hand meta fusion                                          │ HierarchicalChunkClassifier
  ▼                                                               │ (schema v3)
Chunk encoder = Transformer (default) | GRU  + attention pool    │ → chunk embedding
  ▼                                                              ─┘
concat(chunk_embedding, engineered features) ─▶ stacked head → raw P(bot)
  │   stacked head = {XGBoost, ExtraTrees, RandomForest, (LightGBM), (CatBoost)}
  │                  → OOF logistic meta-learner   (--head xgboost for single booster)
  ▼
ScoreCalibrator  (isotonic → threshold-logit remap → logit shift)   ← reward-aware
  ▼
calibrated score  (0.5 boundary correctly placed, FPR under the cliff)
```

**Chunk encoder: Transformer (default) vs GRU.** A chunk is a *homogeneous bag of
hands* (all one label) and "bot-ness" is a distributional / self-consistency
property, not an ordered one — so a permutation-invariant **Transformer** with
attention pooling is the better inductive fit and is the default. Self-attention
directly models cross-hand *consistency* (the strongest bot tell: "do these hands
look like the same mechanical policy?"). The original **GRU** (ordered) is kept as
`--chunk-encoder gru` for the case where reliable chronological order carries
session drift/tilt signal, or when the small benchmark makes the GRU more
sample-efficient. A soft normalized hand-index channel (`--no-hand-position` to
disable) keeps a light ordering hint either way. **A/B both on the reward.**

The calibrator is the snipped-in safety idea: it makes the model's ranking safe
to spend against the FPR cliff.

---

## 3. Recommended recipe (start here)

```bash
cd detection_model

python -m model.train_hierarchical \
  --data data/public_miner_benchmark.json.gz \
  --out artifacts/p44_hier_xgb_cal.pt \
  --epochs 60 --patience 8 \
  --batch-size 16 --lr 1e-4 --weight-decay 1e-4 \
  --d-model 64 --heads 4 --layers 1 --chunk-layers 1 --dropout 0.30 \
  --chunk-encoder transformer \
  --max-hands 20 --max-actions-per-hand 64 \
  --augment-windows --augment-validation-windows \
  --window-hands 20 --window-stride 10 \
  --calibrate-visible-actions --calibrate-validation-visible-actions \
  --xgb-n-estimators 600 --xgb-max-depth 3 --xgb-learning-rate 0.03 \
  --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 --xgb-reg-lambda 2.0 \
  --calibrate --calibration-objective reward \
  --calibration-target-fpr 0.04 --calibration-max-fpr 0.05 \
  --overwrite
```

Then run the miner with **matching** window settings and `mean` aggregation:

```bash
P44_MODEL_PATH=detection_model/artifacts/p44_hier_xgb_cal.pt \
P44_WINDOW_HANDS=20 P44_WINDOW_STRIDE=10 P44_WINDOW_AGG=mean
```

> **The single most common mistake:** training with one window size and serving
> with another. `--max-hands` / `--window-hands` at training time **must** match
> `P44_WINDOW_HANDS` (and `max_hands ≥ window_hands`) at inference, or the encoder
> sees a different number of hands than it was trained on.

---

## 4. Hyperparameters explained

### 4.1 Data, augmentation & train/serve parity

| Flag | Recommended | What it does | Why it matters for reward |
|---|---|---|---|
| `--augment-windows` | on | Slices each training chunk into fixed `--window-hands` windows | Each window is one scored unit at inference; train on the same unit you serve |
| `--window-hands` | 20 | Hands per window (also set `--max-hands` to this) | Bigger = more behavioral context per score (better separation/AP) but fewer, more correlated samples. 16–30 is the sweet spot |
| `--window-stride` | 10 | Step between windows | Smaller = more (overlapping) training samples; ~½ of `--window-hands` balances coverage vs redundancy |
| `--augment-validation-windows` | on | Apply the same windowing to validation | Validation/calibration must look like inference, or the calibrator is tuned on the wrong distribution |
| `--calibrate-visible-actions` | on | Trains on the **miner-visible** action view (5–8 visible actions) | Removes train/serve skew: the validator sanitizes payloads, so train on what you'll actually receive |
| `--calibrate-validation-visible-actions` | on | Same for validation | Keeps the calibration split honest |
| `--min-freq` | 1 | Min token frequency for the action vocab | Keep at 1 for the small benchmark; raise only if the vocab is huge/noisy |

Do **not** use `--augment-prefixes` together with `--augment-windows` (mutually
exclusive). Prefixes help only if you specifically want variable-length chunks.

### 4.2 Neural backbone (owns AP / ranking)

| Flag | Recommended | What it does | Why it matters for reward |
|---|---|---|---|
| `--d-model` | 64 | Embedding/hidden width | Capacity for separation. 64 is plenty for the benchmark; 128 only if you see underfitting (train AP low) |
| `--heads` | 4 | Attention heads (`d_model % heads == 0`) | More heads = finer action-interaction modeling; 4 is balanced |
| `--layers` | 1 | Action-Transformer depth (per hand) | Hands are short (≤ ~12 visible actions); 1–2 layers suffice, more overfits |
| `--chunk-encoder` | `transformer` | Hand→chunk encoder | **transformer** (permutation-invariant set encoder) is the best fit for a homogeneous bag of hands and models cross-hand consistency directly. `gru` = ordered, for drift signal or max sample-efficiency. A/B on the reward |
| `--chunk-layers` | 1 | Chunk-encoder depth | 1 layer suffices for ≤30 hands; 2 only if windows are long |
| `--no-hand-position` | off | Drop the soft hand-index channel | Leave off to keep a light ordering hint; set it for a pure set encoding |
| `--dropout` | 0.30 | Regularization | The benchmark is small; higher dropout protects ranking generalization to live data |
| `--epochs` | 60 | Max passes | Upper bound; early stopping ends it sooner |
| `--patience` | 8 | Early-stopping patience | Stops when the selection metric (ROC+PR+F1) plateaus, avoiding overfit. `0` disables (not recommended) |
| `--batch-size` | 16 | Samples per step | 8–32; larger smooths gradients, smaller regularizes. Tune to GPU/CPU memory |
| `--lr` | 1e-4 | AdamW learning rate | Conservative for a small Transformer; raise to 3e-4 only if loss stalls early |
| `--weight-decay` | 1e-4 | L2 regularization | Standard; helps generalization |
| `--max-grad-norm` | 1.0 | Gradient clipping | Stabilizes Transformer training |
| `--no-pos-weight` | off | Disables class balancing in BCE | Leave OFF: pos_weight upweights bots so the encoder learns to rank them, lifting AP |
| `--max-hands` | 20 | Hands the encoder ingests per window | Must be ≥ `--window-hands`; set equal |
| `--max-actions-per-hand` | 64 | Action cap per hand | 64 is safe; visible payloads rarely exceed ~12, so this is headroom |

### 4.3 XGBoost head (final ranker on embedding + features)

| Flag | Recommended | What it does | Why it matters for reward |
|---|---|---|---|
| `--xgb-n-estimators` | 600 | Boosting rounds | More trees = sharper separation; pair with low LR and regularization to avoid overfit |
| `--xgb-max-depth` | 3 | Tree depth | Shallow trees generalize; the embedding already encodes interactions, so keep depth low |
| `--xgb-learning-rate` | 0.03 | Shrinkage | Low LR + many trees is the stable, high-AP combination |
| `--xgb-subsample` | 0.9 | Row sampling | Mild stochasticity for generalization |
| `--xgb-colsample-bytree` | 0.9 | Column sampling | Same |
| `--xgb-reg-lambda` | 2.0 | L2 on leaf weights | Regularizes; raise if validation AP < train AP by a lot |
| `--xgb-reg-alpha` | 0.0 | L1 | Raise (0.5–1.0) for feature sparsity if overfitting |

`scale_pos_weight` is set automatically from the class balance — it pushes the
head to rank bots correctly, which is AP, which is 65% of the reward.

### 4.4 Reward-aware calibration (owns FPR safety — the new layer)

| Flag | Recommended | What it does | Why it matters for reward |
|---|---|---|---|
| `--calibrate` | on | Fit + embed the `ScoreCalibrator` | Without it the 0.5 boundary is arbitrary and you risk the FPR cliff |
| `--calibration-objective` | `reward` | What the grid search maximizes | **`reward` is correct**: calibration is monotone so AP is fixed; only recall/FPR move, and the reward trades them optimally. `ap_first`/`recall` are for experiments |
| `--calibration-target-fpr` | 0.04 | Conformal FPR target seeding the logit shift | Aims the boundary so ~4% of humans sit above 0.5 — margin below the 0.10 cliff |
| `--calibration-max-fpr` | 0.05 | Hard rejection ceiling | Any config with validation FPR ≥ this is discarded; **this is your cliff insurance**. Lower it to 0.03 for more safety margin (costs some recall) |
| `--no-calibration-isotonic` | off | Disable the isotonic stage | Leave isotonic on; it makes scores better-behaved probabilities without changing rank |

The calibrator prints a before/after line at the end of training:

```
reward 0.00->0.77  fpr 0.30->0.04  recall 0.94->0.65  human_prob_max 0.91->0.48
```

If it stays identity (`Calibrator stayed identity...`), your model is **not
separated enough** for any safe boundary — fix the model (Section 6), not the
calibrator.

### 4.5 Inference / miner runtime (must mirror training)

| Env var | Recommended | Why |
|---|---|---|
| `P44_WINDOW_HANDS` | = training `--window-hands` (20) | Score the same window size the model learned |
| `P44_WINDOW_STRIDE` | = training `--window-stride` (10) | Consistent coverage |
| `P44_WINDOW_AGG` | `mean` | **FPR safety.** `max`/`topk_mean` let one bot-looking window flip a whole human chunk → false positives → the cliff. `mean` is the human-safe aggregator |
| `P44_PREDICTION_THRESHOLD` | leave default | Ignored when a calibrator is embedded (the boundary is already at 0.5) |

---

## 5. Best strategy, and *why* it is best given the scoring

**Recommendation: train hard for ranking, then calibrate hard for the cliff,
with `mean` aggregation.** Concretely:

1. **Maximize AP in the model** (Sections 4.2–4.3). AP is 65% of base_score and
   is the *only* part of the reward calibration cannot recover later. Use
   sliding windows + visible-action training so the AP you measure offline is
   the AP you get live (no train/serve skew). Keep `pos_weight`/`scale_pos_weight`
   on so bots are ranked, not ignored.
2. **Calibrate with `objective=reward`, `max_fpr=0.05`** (Section 4.4). Since
   calibration is monotone, this provably cannot hurt AP; it only places the
   boundary to extract the most recall while staying ~2× under the 0.10 cliff.
   This directly maximizes `base_score · (1-fpr)²`.
3. **Aggregate windows with `mean`** (Section 4.5). The reward punishes false
   positives quadratically and then cliffs them; `mean` is the aggregator least
   likely to push a human chunk over 0.5.

Why not the alternatives:

- *Chase recall (`P44_WINDOW_AGG=max`, low threshold):* raises bot_recall (worth
  0.35) but inflates FPR — and one human chunk over 10% FPR zeroes everything.
  Negative expected value.
- *Optimize the model for accuracy/F1 at 0.5:* couples training to one threshold
  and ignores that AP (ranking) is what pays. The selection metric here already
  sums ROC-AUC + PR-AUC + F1 to favor ranking.
- *Calibrate for `ap_first`/`recall`:* AP is invariant under calibration, so
  `ap_first` degenerates; `recall` ignores the penalty and drifts toward the
  cliff. `reward` is the only objective that reflects what you're paid.

**Safety vs recall dial:** if the live leaderboard shows your FPR creeping up
(distribution drift), lower `--calibration-max-fpr` to 0.03 and retrain the
calibrator only — no model retrain needed. If you're comfortably under the cliff
and want more recall, raise it toward 0.06 (never near 0.10).

---

## 6. Diagnostics — what to watch

During/after training, the reward lines (`model/scoring.py`) are the source of
truth — not loss:

- `validator_ap_score` — your ceiling. Low here = a model problem (more capacity,
  bigger windows, more epochs, check features). Calibration cannot fix it.
- `validator_fpr` (calibrated) — must be **< 0.05** on validation. If it can't get
  there, the model isn't separated enough.
- `human_prob_max` (calibrated) — how close your worst human sits to 0.5. Want it
  comfortably below 0.5.
- `score_gap_at_0_5 = bot_prob_min - human_prob_max` — positive and wide means a
  clean, drift-tolerant boundary.

A healthy artifact looks like: high AP (≥ ~0.9 on the benchmark), calibrated
FPR ≈ 0.03–0.05, recall as high as that FPR allows, and a positive score gap.
```

---

## 7. What was ported from the reference (competitor) model

Three of the reference model's strengths were folded in **without giving up our
reward-aware calibrator**, which remains the final, decisive stage:

1. **Stacked OOF ensemble head (`--head stacked`, default).** Instead of a single
   XGBoost head, the final ranker is now a diverse set of base learners —
   XGBoost + ExtraTrees + RandomForest, plus LightGBM and CatBoost when installed
   — combined by a logistic-regression meta-learner trained on **out-of-fold**
   base predictions (`model/stacked.py`). Ensemble diversity lifts **AP** (65% of
   the reward and the one quantity calibration cannot recover), and OOF stacking
   guards against base-learner overfit. `--head xgboost` restores the old single
   booster. Tunables: `--stack-folds`, `--stack-top-k`, `--stack-meta-c`,
   `--no-stack-lightgbm`, `--no-stack-catboost`.

2. **Richer engineered features** (`model/features.py`): bet-size distribution
   shape (`amount_iqr_bb_scaled`, `amount_p90_bb_scaled`) and betting-rhythm /
   turn-taking tells (`actor_switch_rate`, `max_actor_run_norm`,
   `max_action_type_run_norm`, `long_action_hand_rate`). These capture mechanical
   regularities the leaner feature set previously dropped.

3. **Richer action tokenization** (`model/action_vectorizer.py`, schema **v3**):
   two new categorical channels — `actor_role` (the acting seat's position
   relative to the button) and `street_position` (the action's ordinal index
   within its street). Bots are often position-agnostic and act in mechanical
   order; humans are strongly positional.

> **Retrain required.** The new channels bump the artifact schema to **v3**, so
> existing v2 `.pt` artifacts will no longer load — retrain with
> `python -m model.train_hierarchical` (the recommended recipe in Section 3 now
> uses the stacked head by default) and repoint `P44_MODEL_PATH` at the new file.

What we deliberately **did not** adopt: the reference model's metric-agnostic
quantile "score spreader" calibrator. Our `ScoreCalibrator` already targets the
exact validator reward under the FPR cliff, which is strictly better aligned to
what the subnet pays.
