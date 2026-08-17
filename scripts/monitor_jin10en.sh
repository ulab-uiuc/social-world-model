#!/bin/bash
# Watch the jin10en training job: once best-model is saved, kill the (hang-prone)
# teardown to free GPUs and trigger the afterany eval; then wait for the eval
# result and score it. Prints the final corr/MAE.
set -uo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model
TAG=jin10en
TJ=$(cat logs/.jin10en_train_jobid)
LOG="logs/slurm-worldmodel-${TJ}.out"
RES="results/eval_polymarket/${TAG}.jsonl"

echo "[monitor] watching train job $TJ ..."
# 1) wait until training saved best-model (or job disappears)
for i in $(seq 1 360); do   # up to ~6h at 60s
  if grep -q "Best model saved" "$LOG" 2>/dev/null; then echo "[monitor] best-model saved"; break; fi
  squeue -h -j "$TJ" >/dev/null 2>&1 || { echo "[monitor] train job gone"; break; }
  sleep 60
done

# 2) if still running (teardown hang), kill to free GPU + trigger afterany eval
if squeue -h -j "$TJ" -o "%T" 2>/dev/null | grep -q RUNNING; then
  echo "[monitor] killing hung teardown $TJ"
  scancel "$TJ"
fi

# 3) wait for eval result
echo "[monitor] waiting for eval result $RES ..."
for i in $(seq 1 60); do   # up to ~1h
  n=$(wc -l < "$RES" 2>/dev/null || echo 0)
  [ "$n" -ge 700 ] && { echo "[monitor] eval done ($n rows)"; break; }
  sleep 60
done

# 4) score
echo "[monitor] ===== SCORE ====="
python3 scripts/score_evals.py 2>/dev/null
