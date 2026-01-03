#!/bin/bash
# Train PriorAttributer using KL divergence from precomputed posterior attributions
#
# Data format: Each breakpoint should have 'news' and 'attributions' fields
# attributions format: [{"news_idx": 0, "score": 0.5}, ...]

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

# Kalshi data
CUDA_VISIBLE_DEVICES=5 python train_attributer.py \
    --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_kalshi \
    --model-name Qwen/Qwen3-0.6B \
    --eval-steps 500 \
    --train-batch-size 1 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --max-news-per-bp 30


CUDA_VISIBLE_DEVICES=6 python train_attributer.py \
    --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_kalshi_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 4 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --gradient-checkpointing \
    --max-news-per-bp 30

CUDA_VISIBLE_DEVICES=7 python train_attributer.py \
    --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_kalshi_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 1 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --gradient-checkpointing \
    --max-news-per-bp 30

# Sanity check with Kalshi
# CUDA_VISIBLE_DEVICES=0 python train_attributer.py \
#     --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
#     --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
#     --output-dir ../saves/prior_attributer_kalshi_sanity \
#     --train-batch-size 8 \
#     --eval-batch-size 8 \
#     --epochs 3 \
#     --sanity-check

