#!/bin/bash
# v49: same recipe as v47 but Qwen3-8B, 4-GPU DDP.
# Effective batch = 1 × 4 × 4 = 16. If single forward over 30 events OOMs,
# drop --max-news to 20.
GPUS=${1:-0,1,2,3}
PORT=${2:-29549}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=$PORT \
    train_multievent_forecaster.py \
    --train-data-path /home/haofeiy2/social-world-model/data/v6/v6_clean/train.jsonl \
    --valid-data-path /home/haofeiy2/social-world-model/data/v6/v6_clean/valid_subset150.jsonl \
    --output-dir /mnt/data_from_server1/haofeiy2/swm_saves/forecaster_v49_predict_p_origstd_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 1 --gradient-accumulation-steps 4 --eval-batch-size 1 \
    --learning-rate 5e-5 --lora-r 16 --lora-alpha 32 --head-lr-multiplier 5 \
    --epochs 3 --max-news 30 --max-seq-length 1024 \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    --null-subsample-ratio 0.5 \
    --window-std-threshold 0.02 \
    > /home/haofeiy2/social-world-model/logs/fc_v49_predict_p_origstd_8b.log 2>&1
echo "v49 (8B) done"
