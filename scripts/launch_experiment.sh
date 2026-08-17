#!/bin/bash
# Submit one experiment: train (generic) + eval (dependency afterok).
# Usage: launch_experiment.sh TAG TRAIN_FILE [NULL_SUB] [EVAL_STEPS] [EPOCHS]
set -euo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model
TAG="$1"; TRAIN_FILE="$2"; NULL_SUB="${3:-1.0}"; EVAL_STEPS="${4:-50}"; EPOCHS="${5:-6}"

TRAIN_JOB=$(sbatch --parsable \
  --export=ALL,TAG=$TAG,TRAIN_FILE=$TRAIN_FILE,NULL_SUB=$NULL_SUB,EVAL_STEPS=$EVAL_STEPS,EPOCHS=$EPOCHS \
  scripts/sbatch_train_generic.sh)
EVAL_JOB=$(sbatch --parsable --dependency=afterany:$TRAIN_JOB \
  --export=ALL,TAG=$TAG scripts/sbatch_eval_generic.sh)
echo "TAG=$TAG TRAIN_JOB=$TRAIN_JOB EVAL_JOB=$EVAL_JOB"
