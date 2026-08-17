#!/bin/bash
# Run the clean-sweep under a strict 8-GPU budget: 2 lanes, each runs
# train->eval->next-train serially; the next train in a lane waits on the
# PRIOR eval (afterany) so a lane's 4 GPUs are reused only after its 1-GPU
# eval finishes. Max concurrent = 2 trains (8 GPU); evals never push past 8.
set -euo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model

# tag  train_file  null_sub   (interleaved: even idx -> lane A, odd -> lane B)
EXPS=(
  "e_cdpm  data/train_e_cdpm.jsonl  1.0"
  "e_pmrep data/train_e_pmrep.jsonl 1.0"
  "e_cd    data/train_e_cd.jsonl    1.0"
  "e_cdpmn data/train_e_cdpmn.jsonl 0.3"
  "e_c2pm  data/train_e_c2pm.jsonl  1.0"
  "e_gen   data/train_e_gen.jsonl   1.0"
)

evaljob=()   # per-index eval job id, for chaining index i to i-2
i=0
for e in "${EXPS[@]}"; do
  read -r tag file nsub <<< "$e"
  dep=""
  if [ "$i" -ge 2 ]; then dep="--dependency=afterany:${evaljob[$((i-2))]}"; fi
  tj=$(sbatch --parsable $dep \
      --export=ALL,TAG=$tag,TRAIN_FILE=$file,NULL_SUB=$nsub,EVAL_STEPS=50,EPOCHS=6 \
      scripts/sbatch_train_generic.sh)
  ej=$(sbatch --parsable --dependency=afterany:$tj \
      --export=ALL,TAG=$tag scripts/sbatch_eval_generic.sh)
  evaljob[$i]=$ej
  lane=$(( i % 2 == 0 ? 0 : 1 ))
  echo "[$i lane$lane] $tag  train=$tj eval=$ej ${dep:+(after ${evaljob[$((i-2))]})}"
  i=$((i+1))
done
echo "submitted ${#EXPS[@]} experiments across 2 lanes (<=8 GPU)."
