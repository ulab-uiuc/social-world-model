#!/bin/bash
# Train MultiEventForecaster using precomputed attributions
#
# Supports both single-GPU and multi-GPU (DDP) training:
#   Single GPU: python train_multievent_forecaster.py ...
#   Multi-GPU:  torchrun --nproc_per_node=N train_multievent_forecaster.py ...

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

cd /data/haofeiy2/social-world-model/scripts

# ============================================================
# Overfit tests (single GPU)
# ============================================================

# Forcaster overfit 测试
CUDA_VISIBLE_DEVICES=5 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/forecaster_overfit \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 16 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --overfit \
    --overfit-samples 4 \
    --lora-r 16

# Attributer overfit 测试
CUDA_VISIBLE_DEVICES=5 python train_attributer.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/attributer_overfit \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 16 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --overfit \
    --overfit-samples 4 \
    --lora-r 16

# ============================================================
# Kalshi data with attributions (DDP multi-GPU training)
# ============================================================
# Note: For DDP, effective batch size = train_batch_size * gradient_accumulation_steps * num_gpus
# Adjust gradient_accumulation_steps accordingly when using more GPUs

# 0.6B: 2 GPUs, effective batch size = 8 * 2 * 2 = 32
CUDA_VISIBLE_DEVICES=3,9 torchrun --nproc_per_node=2 train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 &

# 4B: 2 GPUs, effective batch size = 4 * 2 * 2 = 16
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 &

# 8B: 2 GPUs, effective batch size = 2 * 8 * 2 = 32
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 2 \
    --gradient-accumulation-steps 8 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 64 &

wait
echo "Kalshi training complete!"

# ============================================================
# Polymarket data with attributions (DDP multi-GPU training)
# NOTE: Run step4_compute_posterior_attributions.py first to generate attributed data
# ============================================================

# Polymarket 0.6B: 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_polymarket \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --eval-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 &

# Polymarket 4B: 2 GPUs
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_polymarket_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 32 &

# Polymarket 8B: 2 GPUs
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_polymarket_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 64 &

wait
echo "Polymarket training complete!"

# Sanity check
# CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
#     --train-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
#     --valid-data-path ../data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
#     --output-dir ../saves/multievent_forecaster_sanity \
#     --train-batch-size 4 \
#     --eval-batch-size 4 \
#     --epochs 3 \
#     --sanity-check
