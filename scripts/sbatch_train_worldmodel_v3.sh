#!/bin/bash
#SBATCH --job-name=swm-wm-v3
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-worldmodel-%j.out
#SBATCH --error=logs/slurm-worldmodel-%j.err

# v3: the `clean` (jin10) dataset, but with its news TRANSLATED to English
# (scripts/translate_clean_dataset.py). Direct test of whether fixing the
# language mismatch alone lets the clean/attributed data work on the English
# swm-bench. Same recipe as the original clean run for a clean comparison.

set -euo pipefail

REPO=/storage/home/haofeiyu/social-world-model
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/home/haofeiyu/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/home/haofeiyu/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME"

source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/haofeiyu/.conda/envs/swm-train

MODEL_PATH=/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-3B-Instruct
DATA="$REPO/swm-bench/Qwen3.5-397B-attributed-data-polymarket-only"
OUT="$REPO/saves/world_model_qwen2p5_3b_v3"
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=4 scripts/train_multievent_world_model.py \
    --train-data-path "$REPO/data/polymarket_cat5_clean_en_swmbench.jsonl" \
    --valid-data-path "$DATA/valid_subset150.jsonl" \
    --output-dir "$OUT" \
    --model-name "$MODEL_PATH" \
    --max-seq-length 1024 \
    --max-news 30 \
    --epochs 6 \
    --train-batch-size 1 \
    --eval-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --head-lr-multiplier 5 \
    --warmup-steps 30 \
    --logging-steps 10 \
    --eval-steps 20 \
    --save-steps 20 \
    --bf16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp-transformer-layer-cls Qwen2DecoderLayer \
    --gradient-checkpointing \
    --null-rho0 1.0 \
    --odds-eps 1e-3 \
    --odds-temp 1.0 \
    --save-total-limit 3 \
    --no-mid-checkpoints \
    --seed 42
