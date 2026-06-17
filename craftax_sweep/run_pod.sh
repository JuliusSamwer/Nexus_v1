#!/usr/bin/env bash
# RunPod (4090 / 5090) driver for the Craftax-Classic capacity sweep.
#
# Trains BOTH arms (full WM, then tiny WM) to TOTAL env-steps, sequentially on one GPU
# so the card stays loaded end-to-end. Runs under nohup so it survives SSH disconnects;
# re-run this script after any interruption and --resume continues from latest.pkl.
#
#   bash scripts/run_pod.sh            # full then tiny, 10M each, gpc=1
#   TOTAL=5000000 bash scripts/run_pod.sh
#   ARMS="tiny" bash scripts/run_pod.sh   # just one arm
#
# Env knobs (override inline):  TOTAL, GPC, ARMS, CKPT_ROOT, REPO_DIR, REPO_URL
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/JuliusSamwer/Nexus_v1.git}"
REPO_DIR="${REPO_DIR:-/workspace/Nexus_v1}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/ckpts}"
TOTAL="${TOTAL:-10000000}"
GPC="${GPC:-1}"                       # KEEP IDENTICAL across arms
ARMS="${ARMS:-full tiny}"
# use more of the card; harmless if it doesn't need it
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "=== 1. repo ==="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only || echo "(pull skipped — local changes?)"
fi
cd "$REPO_DIR"
git --no-pager log --oneline -1

echo "=== 2. install (one consistent jax+craftax resolve) ==="
pip install -q -U "jax[cuda12]" flax optax craftax

echo "=== 3. gpu + jax sanity ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv || true
python - <<'PY'
import jax
print("jax", jax.__version__, "| devices:", jax.devices())
assert any("cuda" in d.platform.lower() or d.platform == "gpu" for d in jax.devices()), \
    "No GPU visible to JAX"
PY

mkdir -p "$CKPT_ROOT"
LOG="$CKPT_ROOT/train_$(date +%Y%m%d_%H%M%S).log"

echo "=== 4. launch arms [$ARMS] -> $LOG (nohup) ==="
nohup bash -c "
  set -e
  cd '$REPO_DIR'
  for arm in $ARMS; do
    echo \"########## ARM=\$arm  total=$TOTAL  gpc=$GPC  \$(date) ##########\"
    python craftax_sweep/train_craftax_sweep.py \
      --arm \$arm --total $TOTAL --gpc $GPC \
      --ckpt-root '$CKPT_ROOT' --resume
  done
  echo \"########## ALL ARMS DONE \$(date) ##########\"
" > "$LOG" 2>&1 &

PID=$!
echo "launched pid $PID"
echo
echo "monitor:    tail -f $LOG"
echo "gpu watch:  watch -n2 nvidia-smi"
echo "metrics:    tail -f $CKPT_ROOT/full/metrics.jsonl   (and .../tiny/)"
echo "evals:      tail -f $CKPT_ROOT/full/eval.jsonl"
echo "checkpoints land in $CKPT_ROOT/{full,tiny}/  (step_*.pkl milestones + latest.pkl)"
