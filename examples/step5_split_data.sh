#!/bin/bash
# Step 5: Split data by time into train/test sets
# Each market's time series is split at the cutoff date

# Kalshi - cutoff at 2024-11-01
python step5_split_data.py \
    --input_file ../data/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl \
    --cutoff_date 2024-11-01

# Polymarket - cutoff at 2024-10-01
python step5_split_data.py \
    --input_file ../data/processed_polymarket_v2_0102/polymarket_data_processed_with_news.jsonl \
    --cutoff_date 2024-10-01

