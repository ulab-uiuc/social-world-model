#!/bin/bash
# Attributer training on v6_clean, 4-GPU DDP. Runs 0.6B then 4B sequentially.
GPUS=${1:-0,1,2,3}
PORT=${2:-29500}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts

export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

TRAIN=/home/haofeiy2/social-world-model/data/v6/v6_clean/train.jsonl
VALID=/home/haofeiy2/social-world-model/data/v6/v6_clean/valid_subset150.jsonl
LOG=/home/haofeiy2/social-world-model/logs
MNB=8
MSL=1024

# 0.6B: effective batch = 2 × 2 × 4 = 16 (matches old 2 × 8 single-GPU).
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=$PORT \
    train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir ../saves/prior_attributer_combined_06b_klfix_v2 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 2 --gradient-accumulation-steps 2 --eval-batch-size 2 \
    --learning-rate 5e-5 --lora-r 16 --lora-alpha 32 \
    --epochs 3 --max-news $MNB --max-seq-length $MSL \
    --eval-steps 200 --save-steps 200 --logging-steps 10 --gradient-checkpointing \
    > $LOG/attr_06b_klfix_v2.log 2>&1
echo "0.6B done  log=$LOG/attr_06b_klfix_v2.log"

# 4B: effective batch = 1 × 4 × 4 = 16 (matches old 1 × 16 single-GPU).
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=$((PORT+1)) \
    train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir ../saves/prior_attributer_combined_4b_klfix_v2 \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 1 --gradient-accumulation-steps 4 --eval-batch-size 1 \
    --learning-rate 2e-5 --lora-r 32 --lora-alpha 64 \
    --epochs 3 --max-news $MNB --max-seq-length $MSL \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    > $LOG/attr_4b_klfix_v2.log 2>&1
echo "4B done  log=$LOG/attr_4b_klfix_v2.log"
