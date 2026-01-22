#!/bin/bash
# Crawl news for normal (non-breakpoint) days as negative samples
# window_size=15 ensures point_idx >= 15, resulting in window_history length = 17 (same as breakpoints)

DATA_DIR="/mnt/data_from_server1/haofeiy2/social-world-model/data"

# Kalshi
echo "Crawling Kalshi normal point news..."
python step3b_crawl_normal_news.py \
    --use_llm_keywords \
    --use_gnews \
    --samples_per_market 3 \
    --min_days_from_breakpoint 3 \
    --price_change_threshold 0.05 \
    --llm_model gpt-4o-mini \
    --skip_existing \
    --input_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed.jsonl" \
    --output_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_normalpoint_with_news.jsonl"

# Polymarket
echo "Crawling Polymarket normal point news..."
python step3b_crawl_normal_news.py \
    --use_llm_keywords \
    --use_gnews \
    --samples_per_market 3 \
    --min_days_from_breakpoint 3 \
    --price_change_threshold 0.05 \
    --llm_model gpt-4o-mini \
    --input_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed.jsonl" \
    --output_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_normalpoint_with_news.jsonl"

echo ""
echo "Done! Now run step3b_fix_normal_points_structure.sh to ensure window_history = 17"
