#!/bin/bash
#SBATCH --job-name=swm-attributer
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-attributer-%j.out
#SBATCH --error=logs/slurm-attributer-%j.err

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

# activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/haofeiyu/.conda/envs/swm-train

MODEL_PATH=/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-3B-Instruct
DATA="$REPO/swm-bench/Qwen3.5-397B-attributed-data"
OUT="$REPO/saves/prior_attributer_qwen2p5_3b"
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=4 scripts/train_attributer.py \
    --train-data-path "$DATA/train.jsonl" \
    --valid-data-path "$DATA/valid_subset150.jsonl" \
    --output-dir "$OUT" \
    --model-name "$MODEL_PATH" \
    --max-seq-length 1024 \
    --epochs 3 \
    --train-batch-size 1 \
    --eval-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --warmup-steps 30 \
    --logging-steps 20 \
    --eval-steps 200 \
    --save-steps 200 \
    --fp16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp-transformer-layer-cls Qwen2DecoderLayer \
    --gradient-checkpointing \
    --save-total-limit 3 \
    --no-mid-checkpoints \
    --seed 42
