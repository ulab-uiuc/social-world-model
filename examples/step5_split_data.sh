#!/bin/bash
# Split with_news data into train/test (test also by category)

PYTHON="/data/haofeiy2/anaconda3/envs/social-wm/bin/python"
cd "$(dirname "$0")/.."

CUTOFF="2025-11-01"

# Kalshi
$PYTHON examples/step5_split_data.py \
    --input_file data/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl \
    --output_dir data/splitted_kalshi_v2_0102 \
    --cutoff_date $CUTOFF

# Polymarket
$PYTHON examples/step5_split_data.py \
    --input_file data/processed_polymarket_v2_0102/polymarket_data_processed_with_news.jsonl \
    --output_dir data/splitted_polymarket_v2_0102 \
    --cutoff_date $CUTOFF
