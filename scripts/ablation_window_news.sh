#!/bin/bash
# Ablation study: Window size and Max news per breakpoint (Training)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

cd /home/haofeiy2/social-world-model/scripts

# Data paths
KALSHI_TRAIN="/mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_train_2025-11-01.jsonl"
KALSHI_TEST="/mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl"

# ============================================================
# Ablation 1: Window Size
# Default: max_history_len=16, max_news_per_bp=30
# Test: max_history_len = 2, 4, 8
# ============================================================

echo "=== Ablation: Window Size ==="

# Baseline: window=16 (default)
echo "Training window=16 (baseline)..."
# window=2
echo "Training window=2..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_window2 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 2 \
    --max-news-per-bp 30

# window=4
echo "Training window=4..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_window4 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 4 \
    --max-news-per-bp 30

# window=8
echo "Training window=8..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_window8 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 8 \
    --max-news-per-bp 30

echo "Window size ablation complete!"

# ============================================================
# Ablation 2: Max News per Breakpoint
# Default: max_news_per_bp=30, max_history_len=16
# Test: max_news_per_bp = 1, 5, 10, 20
# ============================================================

echo "=== Ablation: Max News ==="

# Baseline: news=30 (already trained above as window16)

# news=1
echo "Training news=1..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_news1 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 16 \
    --max-news-per-bp 1

# news=5
echo "Training news=5..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_news5 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 16 \
    --max-news-per-bp 5

# news=10
echo "Training news=10..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_news10 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 16 \
    --max-news-per-bp 10

# news=20
echo "Training news=20..."
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path $KALSHI_TRAIN \
    --valid-data-path $KALSHI_TEST \
    --output-dir ../saves/ablation_news20 \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --eval-batch-size 4 \
    --epochs 10 \
    --logging-steps 10 \
    --save-steps 50 \
    --gradient-checkpointing \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --max-history-len 16 \
    --max-news-per-bp 20

echo "Max news ablation complete!"
echo "All ablation training complete!"
echo "Models saved to ../saves/ablation_*/"
