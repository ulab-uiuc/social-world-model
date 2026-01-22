#!/bin/bash
# Split attributed data into train/test (test also by category)
# Input: flat format with attributions

cd "$(dirname "$0")/.."

DATA_DIR="/mnt/data_from_server1/haofeiy2/social-world-model/data"
CUTOFF="2025-11-01"

# Kalshi
python examples/step5_split_data.py \
    --input_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_with_news_attributed.jsonl" \
    --output_dir "${DATA_DIR}/splitted_kalshi_v2_0102" \
    --cutoff_date $CUTOFF

# Polymarket
python examples/step5_split_data.py \
    --input_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_with_news_attributed.jsonl" \
    --output_dir "${DATA_DIR}/splitted_polymarket_v2_0102" \
    --cutoff_date $CUTOFF
