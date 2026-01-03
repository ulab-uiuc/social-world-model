#!/bin/bash
# Train MultiEventForecaster using precomputed attributions

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

cd /data/haofeiy2/social-world-model/scripts

# Kalshi data with attributions
CUDA_VISIBLE_DEVICES=4 python train_multievent_forecaster.py \
    --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 1 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --gradient-checkpointing \
    --learning-rate 5e-5

CUDA_VISIBLE_DEVICES=3 python train_multievent_forecaster.py \
    --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --gradient-checkpointing

CUDA_VISIBLE_DEVICES=2 python train_multievent_forecaster.py \
    --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 1 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --gradient-checkpointing

# Sanity check
# CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
#     --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
#     --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
#     --output-dir ../saves/multievent_forecaster_sanity \
#     --train-batch-size 4 \
#     --eval-batch-size 4 \
#     --epochs 3 \
#     --sanity-check
