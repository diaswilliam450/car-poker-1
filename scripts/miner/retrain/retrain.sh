#!/usr/bin/env bash
# Daily auto-retrain for a Poker44 v2 miner.
#
#   fetch new public-benchmark releases  ->  rebuild a date-fresh training set
#   ->  train a candidate  ->  gate on held-out reward + FPR  ->  atomic swap
#   ->  pm2 restart the miner so it reloads the new artifact.
#
# The miner loads its artifact once at startup (no hot reload), so promotion is
# a same-filesystem atomic rename + a pm2 restart. The previous artifact is kept
# as .prev for rollback. Everything is best-effort and idempotent: an
# unreachable API or a worse candidate leaves the live artifact untouched.
#
# Per-miner behaviour is driven by env vars (usually from the project .env):
#   POKER44_PM2_NAME          pm2 process to restart on promotion (required to reload)
#   P44_MODEL_PATH            served artifact path (default detection_model/artifacts/p44_v2_lgbm_canon.joblib)
#   RETRAIN_BENCH_BOT_AUG / RETRAIN_BENCH_HUM_AUG / RETRAIN_N_CORPUS_HUMAN /
#   RETRAIN_N_SYNTH_BOT / RETRAIN_SYNTH_ARCHETYPES / RETRAIN_TRAIN_SEED /
#   RETRAIN_MIN_REWARD / RETRAIN_MAX_FPR / RETRAIN_TARGET_FPR
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${RETRAIN_PYTHON:-$REPO_ROOT/miner_env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# Load project .env (for POKER44_PM2_NAME etc.) without leaking to git.
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi

DATA_DIR="$REPO_ROOT/detection_model/data"
PULLS_DIR="$DATA_DIR/pulls"
ART_DIR="$REPO_ROOT/detection_model/artifacts"
mkdir -p "$DATA_DIR" "$PULLS_DIR" "$ART_DIR"
LOG() { echo "[retrain $(printf '%(%Y-%m-%dT%H:%M:%S)T' -1)] $*"; }

MODEL_PATH="${P44_MODEL_PATH:-detection_model/artifacts/p44_v2_lgbm_canon.joblib}"
CANON="$REPO_ROOT/$MODEL_PATH"
CAND="$ART_DIR/candidate.joblib"
# date seed -> a genuinely fresh resample/synthesis each day
SEED="${RETRAIN_TRAIN_SEED:-$(printf '%(%Y%m%d)T' -1)}"
TRAIN_JSON="$DATA_DIR/train_${SEED}.json"

MIN_REWARD="${RETRAIN_MIN_REWARD:-0.55}"
MAX_FPR="${RETRAIN_MAX_FPR:-0.05}"
TARGET_FPR="${RETRAIN_TARGET_FPR:-0.04}"

LOG "repo=$REPO_ROOT model=$MODEL_PATH seed=$SEED"

# 1) fetch newly-released benchmark data (best-effort; never fatal)
"$PY" -m detection_model.tools.get_dump --out-dir "$PULLS_DIR" || LOG "get_dump non-fatal failure"

# 2) build a fresh, benchmark-grounded training set
"$PY" -m detection_model.tools.build_dataset \
  --out "$TRAIN_JSON" --seed "$SEED" --extra-dir "$PULLS_DIR" \
  --bench-bot-aug "${RETRAIN_BENCH_BOT_AUG:-120}" \
  --bench-hum-aug "${RETRAIN_BENCH_HUM_AUG:-120}" \
  --n-corpus-human "${RETRAIN_N_CORPUS_HUMAN:-100}" \
  --n-synth-bot "${RETRAIN_N_SYNTH_BOT:-0}" \
  --synth-archetypes "${RETRAIN_SYNTH_ARCHETYPES:-nit:1,aggro:1,station:1,cbot:1}" \
  || { LOG "build_dataset failed; abort (live model untouched)"; exit 0; }

# 3) train a candidate
( cd "$REPO_ROOT/detection_model" && "$PY" -m model_v2.train \
    --data "$TRAIN_JSON" --out "$CAND" --seed "$SEED" \
    --target-fpr "$TARGET_FPR" --max-fpr "$MAX_FPR" ) \
  || { LOG "training failed; abort (live model untouched)"; exit 0; }

# 4) gate: candidate must clear a reward floor and FPR ceiling on the real benchmark
GATE=$( cd "$REPO_ROOT/detection_model" && "$PY" -m model_v2.evaluate \
    --model "$CAND" --data "$REPO_ROOT/detection_model/data/eval_benchmark.json" 2>/dev/null \
    || "$PY" -m model_v2.evaluate --model "$CAND" \
       --data <("$PY" - <<'PY'
import gzip,json
d=json.load(gzip.open('detection_model/public_miner_benchmark.json.gz','rt'))
print(json.dumps(d['labeled_chunks']))
PY
) )
REWARD=$(echo "$GATE" | grep -oE '"reward":[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | head -1)
FPR=$(echo "$GATE"    | grep -oE '"fpr_at_0.5":[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | head -1)
REWARD="${REWARD:-0}"; FPR="${FPR:-1}"
LOG "candidate gate: reward=$REWARD fpr=$FPR (need reward>=$MIN_REWARD, fpr<=$MAX_FPR)"

PROMOTE=$("$PY" - "$REWARD" "$FPR" "$MIN_REWARD" "$MAX_FPR" <<'PY'
import sys
reward,fpr,minr,maxf=map(float,sys.argv[1:5])
print("yes" if (reward>=minr and fpr<=maxf) else "no")
PY
)

if [ "$PROMOTE" != "yes" ]; then
  LOG "candidate rejected; keeping current live artifact."
  exit 0
fi

# 5) atomic swap (same filesystem) + keep rollback copy
[ -f "$CANON" ] && cp -f "$CANON" "$CANON.prev"
mv -f "$CAND" "$CANON.tmp" && mv -f "$CANON.tmp" "$CANON"
LOG "promoted new artifact -> $CANON"

# 6) reload the miner (loads artifact + rebuilds manifest sha256 at startup)
if [ -n "${POKER44_PM2_NAME:-}" ] && command -v pm2 >/dev/null 2>&1; then
  pm2 restart "$POKER44_PM2_NAME" --update-env >/dev/null 2>&1 && LOG "pm2 restarted $POKER44_PM2_NAME"
else
  LOG "no POKER44_PM2_NAME/pm2; restart the miner manually to load the new artifact."
fi

# tidy old date-stamped training files (keep last 5)
ls -1t "$DATA_DIR"/train_*.json 2>/dev/null | tail -n +6 | xargs -r rm -f
LOG "done."
