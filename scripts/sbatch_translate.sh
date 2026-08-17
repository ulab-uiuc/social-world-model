#!/bin/bash
#SBATCH --job-name=swm-translate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-translate-%j.out
#SBATCH --error=logs/slurm-translate-%j.err

set -euo pipefail
REPO=/storage/home/haofeiyu/social-world-model
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export HF_HOME=/storage/home/haofeiyu/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/home/haofeiyu/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/haofeiyu/.conda/envs/swm-train

MODEL=/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-7B-Instruct

python scripts/translate_clean_dataset.py \
    --model-path "$MODEL" \
    --batch-size 32 \
    --max-new-tokens 512

# Rebuild the swm-bench training file from the translated set.
python scripts/build_clean_swmbench.py \
    --src data/polymarket_cat5_clean_en.jsonl \
    --dst data/polymarket_cat5_clean_en_swmbench.jsonl
