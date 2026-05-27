#!/bin/bash
# v47: v44 config but with ORIG attribution + window_std cleanup.
# Multi-GPU DDP — pass GPU list, defaults to all 4 ours (0,1,2,3).
# Effective batch: per-gpu 2 × 4 gpus × grad_accum 2 = 16 (same as single-GPU v44).
GPUS=${1:-0,1,2,3}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29502 \
    train_multievent_forecaster.py \
    --train-data-path /home/haofeiy2/social-world-model/data/vllm_attributed/combined_train_vllm_attributed.jsonl \
    --valid-data-path /home/haofeiy2/social-world-model/data/vllm_attributed/combined_valid_subset150.jsonl \
    --output-dir /mnt/data_from_server1/haofeiy2/swm_saves/forecaster_v47_predict_p_origstd \
    --model-name Qwen/Qwen3-0.6B \
    --cache-dir /mnt/data_from_server1/haofeiy2/social-world-model/cache \
    --train-batch-size 2 --gradient-accumulation-steps 2 --eval-batch-size 2 \
    --learning-rate 5e-5 --lora-r 16 --lora-alpha 32 --head-lr-multiplier 5 \
    --epochs 3 --max-news-per-bp 30 --max-seq-length 1024 \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    --predict-absolute-price \
    --null-subsample-ratio 0.0 \
    --window-std-threshold 0.02 \
    > /home/haofeiy2/social-world-model/logs/fc_v47_predict_p_origstd.log 2>&1
echo "v47 done"
