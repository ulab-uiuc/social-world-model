#!/bin/bash
# v50 4B: vanilla full-finetune baseline on v6_clean, Qwen3-4B.
# Attributor + forecaster, 2-GPU DDP each, launched in parallel.
#
# Usage:
#   bash train_v50_4b.sh                 # attr→6,7  fc→8,9
#   bash train_v50_4b.sh 0,1 2,3         # custom GPU split
ATTR_GPUS=${1:-6,7}
FC_GPUS=${2:-8,9}
ATTR_PORT=${3:-29560}
FC_PORT=${4:-29561}

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
mkdir -p $LOG /mnt/data_from_server2/haofeiy2/v50_saves/{attributer,forecaster}_v50_4b

# Attributor 4B: 2-GPU DDP. Effective batch = 1 × 8 × 2 = 16.
CUDA_VISIBLE_DEVICES=$ATTR_GPUS torchrun --nproc_per_node=2 --master_port=$ATTR_PORT \
    train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir /mnt/data_from_server2/haofeiy2/v50_saves/attributer_v50_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 1 --gradient-accumulation-steps 8 --eval-batch-size 1 \
    --learning-rate 2e-5 \
    --epochs 3 --max-news 8 --max-seq-length 1024 \
    --eval-steps 200 --save-steps 200 --logging-steps 10 --gradient-checkpointing \
    --null-subsample-ratio 0.5 \
    > $LOG/attr_v50_4b.log 2>&1 &
ATTR_PID=$!
echo "attributor v50 4B  PID=$ATTR_PID  GPUS=$ATTR_GPUS  log=$LOG/attr_v50_4b.log"

# Forecaster 4B: 2-GPU DDP. Effective batch = 1 × 8 × 2 = 16.
CUDA_VISIBLE_DEVICES=$FC_GPUS torchrun --nproc_per_node=2 --master_port=$FC_PORT \
    train_multievent_forecaster.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir /mnt/data_from_server2/haofeiy2/v50_saves/forecaster_v50_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 1 --gradient-accumulation-steps 8 --eval-batch-size 1 \
    --learning-rate 2e-5 --head-lr-multiplier 5 \
    --epochs 3 --max-news 15 --max-seq-length 1024 \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    --null-subsample-ratio 0.5 \
    > $LOG/fc_v50_4b.log 2>&1 &
FC_PID=$!
echo "forecaster v50 4B  PID=$FC_PID  GPUS=$FC_GPUS  log=$LOG/fc_v50_4b.log"

echo "Tail logs:  tail -f $LOG/attr_v50_4b.log $LOG/fc_v50_4b.log"
echo "Kill both:  kill $ATTR_PID $FC_PID"
wait
