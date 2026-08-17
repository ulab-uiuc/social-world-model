#!/bin/bash
# Auto-submit eval jobs if final-model is ready and eval hasn't been submitted yet
set -e
cd /storage/home/haofeiyu/social-world-model
mkdir -p logs results/eval_polymarket

for tag in pm clean v2 v3; do
  if [ -d "saves/world_model_qwen2p5_3b_${tag}/final-model" ] && \
     [ -f "saves/world_model_qwen2p5_3b_${tag}/final-model/model.safetensors.index.json" ] && \
     [ ! -f "results/eval_polymarket/${tag}.jsonl" ] && \
     [ ! -f "logs/.eval_${tag}_submitted" ]; then
    echo "submitting eval for $tag..."
    sbatch "scripts/sbatch_eval_${tag}.sh"
    touch "logs/.eval_${tag}_submitted"
  fi
done
