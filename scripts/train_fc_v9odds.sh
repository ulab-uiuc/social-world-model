#!/bin/bash
# v9 recipe + ODDS+NULL CATEGORICAL target, on the SEMDEDUP 397B train_clean.
# per-news, batch4, lr2e-5, head x5, warmup30, 6ep, max-news30, seq1024.
# Args: MODE GPUS NPROC PORT MODEL TAG SAVE EP
set -uo pipefail
ENVDIR=/home/haofeiy2/.conda/envs/agentgym
REPO=/home/haofeiy2/social-world-model
cd "$REPO/scripts"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_NCCL_TIMEOUT_MS=3600000

MODE=$1; GPUS=$2; NPROC=$3; PORT=$4; MODEL=$5; TAG=$6; SAVE=$7; EP=${8:-6}
DATA="$REPO/data/social-world-model-v6-qwen3.5-397B-clean-semdedup"
LOG="$REPO/logs"; mkdir -p "$LOG" "$SAVE"

COMMON=(
  --train-data-path "$DATA/train_clean.jsonl"
  --valid-data-path "$DATA/valid_subset150.jsonl"
  --output-dir "$SAVE/$TAG" --model-name "$MODEL"
  --train-batch-size 4 --gradient-accumulation-steps 1 --eval-batch-size 4
  --learning-rate 2e-5 --head-lr-multiplier 5 --warmup-steps 30 --bf16
  --null-rho0 1.0 --odds-eps 1e-3 --odds-temp 1.0
  --epochs "$EP" --max-news 30 --max-seq-length 1024
  --eval-steps 50 --logging-steps 10 --gradient-checkpointing
)

if [ "$MODE" = "fsdp" ]; then
  CUDA_VISIBLE_DEVICES=$GPUS "$ENVDIR/bin/torchrun" --nproc_per_node=$NPROC --master_port=$PORT \
    train_multievent_world_model.py "${COMMON[@]}" \
    --fsdp "full_shard auto_wrap" --fsdp-transformer-layer-cls Qwen3DecoderLayer \
    --no-mid-checkpoints \
    > "$LOG/$TAG.log" 2>&1
else
  CUDA_VISIBLE_DEVICES=$GPUS "$ENVDIR/bin/python" \
    train_multievent_world_model.py "${COMMON[@]}" --save-steps 50 \
    > "$LOG/$TAG.log" 2>&1
fi
echo "$TAG done (exit $?)"
