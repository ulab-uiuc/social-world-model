#!/bin/bash
# Merge normal_points (negative samples) and breakpoints (positive samples) into flat format
# Each line in output is a single sample point

DATA_DIR="/mnt/data_from_server1/haofeiy2/social-world-model/data"

# Kalshi
echo "Merging Kalshi data..."
python step3c_merge_points.py \
    --breakpoints_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_breakpoint_with_news.jsonl" \
    --normal_points_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news.jsonl" \
    --flat_output_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl"

echo ""

# Polymarket
echo "Merging Polymarket data..."
python step3c_merge_points.py \
    --breakpoints_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_breakpoint_with_news.jsonl" \
    --normal_points_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_normalpoint_with_news.jsonl" \
    --flat_output_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_with_news.jsonl"

echo ""
echo "All merging completed!"
