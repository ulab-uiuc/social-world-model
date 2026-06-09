#!/bin/bash
# Full fine-tune (FSDP) prior attributer. Data dir via DATADIR (default 397b-semdedup).
# Usage: train_attributer_fullft.sh <gpus> <port> <run_name> <model> <batch> <lr> [extra args...]
# Single-GPU works for 0.6B (full params+Adam in fp32 ~10GB). Multi-GPU adds FSDP.
set -u
GPUS=$1; PORT=$2; NAME=$3; MODEL=$4; BATCH=$5; LR=$6; shift 6
EXTRA="$@"
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts

export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

DATA=${DATADIR:-/home/haofeiy2/social-world-model/data/v6_qwen235b_clean_dedup}
TRAIN=$DATA/train.jsonl
VALID=$DATA/valid_subset150.jsonl
LOG=/home/haofeiy2/social-world-model/logs
OUT=/mnt/disk2_from_server2/haofeiy2/swm_attributer_ckpts/prior_attributer_${OUTPREFIX:-v6_235b_}$NAME
MSL=1024; MNB=30

# Multi-GPU: enable FSDP full_shard (the --fsdp value must stay one token-with-space,
# so pass via an array — unquoted expansion splits "full_shard auto_wrap" and breaks argparse).
FSDP_ARGS=()
if [ "$NPROC" -gt 1 ]; then
  FSDP_ARGS=(--fsdp "full_shard auto_wrap" --fsdp-transformer-layer-cls Qwen3DecoderLayer)
fi

echo "[fullft] name=$NAME model=$MODEL gpus=$GPUS nproc=$NPROC batch=$BATCH lr=$LR fsdp='${FSDP_ARGS[*]}' extra='$EXTRA'"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$PORT \
    train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir $OUT \
    --model-name $MODEL \
    --train-batch-size $BATCH --gradient-accumulation-steps ${GRAD_ACCUM:-1} --eval-batch-size $BATCH \
    --learning-rate $LR --bf16 --warmup-steps ${WARMUP:-20} \
    --epochs ${EPOCHS:-3} --max-news $MNB --max-seq-length $MSL \
    --eval-steps ${EVAL_STEPS:-50} --save-steps ${SAVE_STEPS:-300} --logging-steps 20 \
    --save-only-model --ddp-timeout ${DDP_TIMEOUT:-7200} --gradient-checkpointing \
    "${FSDP_ARGS[@]}" $EXTRA \
    > $LOG/attr_$NAME.log 2>&1
echo "[fullft] $NAME done -> $OUT  (log=$LOG/attr_$NAME.log)"
