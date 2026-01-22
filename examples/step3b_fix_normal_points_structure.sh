#!/bin/bash
# Fix normal_points structure to match breakpoints structure
# Keeps existing news, just adds missing fields (before, after, window_history, etc.)
# Filters out points with insufficient history to ensure window_history length = 17

DATA_DIR="/mnt/data_from_server1/haofeiy2/social-world-model/data"

# Kalshi
echo "Fixing Kalshi normal points..."
python step3b_fix_normal_points_structure.py \
    --normal_points_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news.jsonl" \
    --processed_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed.jsonl" \
    --output_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news_fixed.jsonl" \
    --window_size 15 \
    --min_history 15

# Backup and replace
if [ -f "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news_fixed.jsonl" ]; then
    echo "Replacing original file..."
    mv "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news.jsonl" \
       "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news.jsonl.bak"
    mv "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news_fixed.jsonl" \
       "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news.jsonl"
fi

echo ""

# Polymarket (if exists)
if [ -f "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news.jsonl" ]; then
    echo "Fixing Polymarket normal points..."
    python step3b_fix_normal_points_structure.py \
        --normal_points_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news.jsonl" \
        --processed_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed.jsonl" \
        --output_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news_fixed.jsonl" \
        --window_size 15 \
        --min_history 15
    
    if [ -f "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news_fixed.jsonl" ]; then
        echo "Replacing original file..."
        mv "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news.jsonl" \
           "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news.jsonl.bak"
        mv "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news_fixed.jsonl" \
           "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news.jsonl"
    fi
fi

echo ""
echo "Done! Original files backed up with .bak extension."
