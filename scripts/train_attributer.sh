#!/bin/bash
# Train PriorAttributer using KL divergence from precomputed posterior attributions
#
# Data format: Each breakpoint should have 'news' and 'attributions' fields
# attributions format: [{"news_idx": 0, "score": 0.5}, ...]
#
# Supports both single-GPU and multi-GPU (DDP) training:
#   Single GPU: python train_attributer.py ...
#   Multi-GPU:  torchrun --nproc_per_node=N train_attributer.py ...

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

cd /home/haofeiy2/social-world-model/scripts

# ============================================================
# Kalshi data (DDP multi-GPU training)
# ============================================================
# Note: For DDP, effective batch size = train_batch_size * gradient_accumulation_steps * num_gpus

# 0.6B: 2 GPUs, effective batch size = 4 * 2 * 2 = 16
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_kalshi_06b \
    --model-name Qwen/Qwen3-0.6B \
    --eval-steps 500 \
    --save-steps 50 \
    --train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-news-per-bp 30 &

# 4B: 2 GPUs, effective batch size = 2 * 4 * 2 = 16
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_kalshi_4b \
    --model-name Qwen/Qwen3-4B \
    --save-steps 50 \
    --train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 2e-5 \
    --lora-r 32 \
    --gradient-checkpointing \
    --max-news-per-bp 30 &

# 8B: 2 GPUs, effective batch size = 1 * 8 * 2 = 16
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_kalshi_8b \
    --model-name Qwen/Qwen3-8B \
    --save-steps 50 \
    --train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 1e-5 \
    --lora-r 64 \
    --gradient-checkpointing \
    --max-news-per-bp 30 &

wait
echo "Kalshi attributer training complete!"

# ============================================================
# Polymarket data (DDP multi-GPU training)
# NOTE: Run step4_compute_posterior_attributions.py first to generate attributed data
# ============================================================

# Polymarket 0.6B: 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_polymarket_06b \
    --model-name Qwen/Qwen3-0.6B \
    --eval-steps 50 \
    --save-steps 50 \
    --train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-news-per-bp 30 &

# Polymarket 4B: 2 GPUs
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_polymarket_4b \
    --model-name Qwen/Qwen3-4B \
    --save-steps 50 \
    --train-batch-size 2 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 2e-5 \
    --lora-r 32 \
    --gradient-checkpointing \
    --max-news-per-bp 30 &

# Polymarket 8B: 2 GPUs
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/prior_attributer_polymarket_8b \
    --model-name Qwen/Qwen3-8B \
    --save-steps 50 \
    --eval-steps 50 \
    --train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --learning-rate 1e-5 \
    --lora-r 64 \
    --gradient-checkpointing \
    --max-news-per-bp 30 &

wait
echo "Polymarket attributer training complete!"

# ============================================================
# Single GPU examples (for reference)
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train_attributer.py \
#     --train-data-path ../data/attributed/train.jsonl \
#     --valid-data-path ../data/attributed/valid.jsonl \
#     --output-dir ../saves/prior_attributer \
#     --model-name Qwen/Qwen3-0.6B \
#     --train-batch-size 8 \
#     --epochs 10

