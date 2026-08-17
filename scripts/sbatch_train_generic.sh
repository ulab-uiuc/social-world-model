#!/bin/bash
#SBATCH --job-name=swm-wm
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-worldmodel-%j.out
#SBATCH --error=logs/slurm-worldmodel-%j.err

# Generic parameterized world-model trainer for the experiment sweep.
# Required env: TAG, TRAIN_FILE
# Optional env: NULL_SUB (default 1.0), EVAL_STEPS (default 50), EPOCHS (default 6)
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
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/haofeiyu/.conda/envs/swm-train

: "${TAG:?}"; : "${TRAIN_FILE:?}"
NULL_SUB="${NULL_SUB:-1.0}"; EVAL_STEPS="${EVAL_STEPS:-50}"; EPOCHS="${EPOCHS:-6}"
HEAD_LR="${HEAD_LR:-5}"; WARMUP="${WARMUP:-30}"; SEED="${SEED:-42}"
NPROC="${NPROC:-4}"
MODEL_PATH="${MODEL_PATH:-/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
DATA="$REPO/swm-bench/Qwen3.5-397B-attributed-data-polymarket-only"
VALID_FILE="${VALID_FILE:-$DATA/valid_subset150.jsonl}"
OUT="$REPO/saves/world_model_qwen2p5_3b_${TAG}"
mkdir -p "$OUT"
echo "[${TAG}] train=$TRAIN_FILE valid=$VALID_FILE null_sub=$NULL_SUB eval_steps=$EVAL_STEPS epochs=$EPOCHS head_lr=$HEAD_LR warmup=$WARMUP seed=$SEED"

torchrun --standalone --nproc_per_node="$NPROC" scripts/train_multievent_world_model.py \
    --train-data-path "$TRAIN_FILE" \
    --valid-data-path "$VALID_FILE" \
    --output-dir "$OUT" \
    --model-name "$MODEL_PATH" \
    --max-seq-length 1024 --max-news "${MAXNEWS:-30}" --epochs "$EPOCHS" \
    --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 --head-lr-multiplier "$HEAD_LR" --warmup-steps "$WARMUP" \
    --logging-steps 10 --eval-steps "$EVAL_STEPS" --save-steps "$EVAL_STEPS" \
    --bf16 --fsdp "full_shard auto_wrap" --fsdp-transformer-layer-cls Qwen2DecoderLayer \
    --gradient-checkpointing --null-rho0 1.0 --odds-eps 1e-3 --odds-temp 1.0 \
    --null-subsample-ratio "$NULL_SUB" \
    --save-total-limit 3 --no-mid-checkpoints --seed "${SEED:-42}"
