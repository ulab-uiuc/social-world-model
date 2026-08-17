#!/bin/bash
# Concurrent watcher: for EACH training job, once it saves best-model, kill its
# hang-prone teardown to free GPUs + fire its afterany evals. Then wait for the
# final matrix result files and score. (v2: watches jobs in parallel subshells.)
set -uo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model

watch_and_kill() {  # jobid
  local j=$1 log="logs/slurm-worldmodel-$1.out"
  for i in $(seq 1 360); do
    grep -q "Best model saved" "$log" 2>/dev/null && break
    squeue -h -j "$j" >/dev/null 2>&1 || return 0
    sleep 60
  done
  if squeue -h -j "$j" -o "%T" 2>/dev/null | grep -q RUNNING; then
    echo "[monitor] killing hung teardown $j"; scancel "$j"
  fi
}

watch_and_kill "$(cat logs/.jin10en_train_jobid)" &
watch_and_kill "$(cat logs/.swmb_comb_s1_train_jobid)" &
wait
echo "[monitor] both trainings done + teardowns freed"

# key matrix files (good baseline = swmb_comb_s1)
files="jin10en jin10en_on_swmb swmb_comb_s1 swmb_comb_s1_on_jin10"
for i in $(seq 1 90); do
  ok=1
  for f in $files; do
    n=$(wc -l < "results/eval_polymarket/$f.jsonl" 2>/dev/null || echo 0)
    [ "$n" -ge 600 ] || ok=0
  done
  [ "$ok" = 1 ] && { echo "[monitor] all results ready"; break; }
  sleep 60
done

echo "[monitor] ===== FULL SCORE ====="
python3 scripts/score_evals.py 2>/dev/null
