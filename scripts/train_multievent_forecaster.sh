#!/bin/bash
# Train MultiEventForecaster using precomputed attributions

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

cd /data/haofeiy2/social-world-model/scripts

# Kalshi data with attributions
# 0.6B: smaller LoRA rank (r=16) is fine
# Effective batch size = train_batch_size * gradient_accumulation_steps = 1 * 8 = 8
CUDA_VISIBLE_DEVICES=1 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 16 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --r 16 &

# 4B: increase LoRA rank to leverage model capacity
# Effective batch size = 1 * 8 = 8
CUDA_VISIBLE_DEVICES=3 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 2e-5 \
    --r 32 &

# 8B: larger LoRA rank + lower LR for stability
# Effective batch size = 1 * 16 = 16 (larger for more stable gradients)
CUDA_VISIBLE_DEVICES=2 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 1e-5 \
    --r 64 &

wait
echo "Kalshi training complete!"

# ============================================================
# Polymarket data with attributions
# NOTE: Run step4_compute_posterior_attributions.py first to generate attributed data
# ============================================================

# Polymarket 0.6B
CUDA_VISIBLE_DEVICES=9 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_polymarket \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --eval-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --r 16 &

# Polymarket 4B
CUDA_VISIBLE_DEVICES=3 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_polymarket_4b \
    --model-name Qwen/Qwen3-4B \
    --train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --eval-batch-size 2 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 2e-5 \
    --r 32 &

# Polymarket 8B
CUDA_VISIBLE_DEVICES=2 python train_multievent_forecaster.py \
    --train-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_polymarket_8b \
    --model-name Qwen/Qwen3-8B \
    --train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --eval-batch-size 1 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 1e-5 \
    --r 64 &

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
