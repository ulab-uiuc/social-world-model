#!/bin/bash
# v50: vanilla full-finetune baseline on v6_clean. Attributor + forecaster,
# 2-GPU DDP each, launched in parallel. No LoRA.
#
# Usage:
#   bash train_v50.sh                   # attr→1,2  fc→3,4
#   bash train_v50.sh 0,1 2,3           # custom GPU split
ATTR_GPUS=${1:-1,2}
FC_GPUS=${2:-3,4}
ATTR_PORT=${3:-29550}
FC_PORT=${4:-29551}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

TRAIN=/home/haofeiy2/social-world-model/data/v6/v6_clean/train.jsonl
VALID=/home/haofeiy2/social-world-model/data/v6/v6_clean/valid_subset150.jsonl
LOG=/home/haofeiy2/social-world-model/logs
mkdir -p $LOG

# Attributor: Qwen3-0.6B full FT on $ATTR_GPUS. Effective batch = 8 × 1 × 2 = 16.
CUDA_VISIBLE_DEVICES=$ATTR_GPUS torchrun --nproc_per_node=2 --master_port=$ATTR_PORT \
    train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir /mnt/data_from_server1/haofeiy2/swm_saves/attributer_v50_06b \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 --gradient-accumulation-steps 1 --eval-batch-size 8 \
    --learning-rate 2e-5 \
    --epochs 3 --max-news 8 --max-seq-length 1024 \
    --eval-steps 200 --save-steps 200 --logging-steps 10 --gradient-checkpointing \
    --null-subsample-ratio 0.5 \
    > $LOG/attr_v50_06b.log 2>&1 &
ATTR_PID=$!
echo "attributor v50 0.6B  PID=$ATTR_PID  GPUS=$ATTR_GPUS  log=$LOG/attr_v50_06b.log"

# Forecaster: Qwen3-0.6B full FT on $FC_GPUS. Effective batch = 4 × 2 × 2 = 16.
CUDA_VISIBLE_DEVICES=$FC_GPUS torchrun --nproc_per_node=2 --master_port=$FC_PORT \
    train_multievent_forecaster.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir /mnt/data_from_server1/haofeiy2/swm_saves/forecaster_v50_06b \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 4 --gradient-accumulation-steps 2 --eval-batch-size 4 \
    --learning-rate 2e-5 --head-lr-multiplier 5 \
    --epochs 3 --max-news 30 --max-seq-length 1024 \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    --null-subsample-ratio 0.5 \
    > $LOG/fc_v50_06b.log 2>&1 &
FC_PID=$!
echo "forecaster v50 0.6B  PID=$FC_PID  GPUS=$FC_GPUS  log=$LOG/fc_v50_06b.log"

echo "Tail logs:  tail -f $LOG/attr_v50_06b.log $LOG/fc_v50_06b.log"
echo "Kill both:  kill $ATTR_PID $FC_PID"
wait
