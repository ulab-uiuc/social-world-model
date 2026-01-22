#!/bin/bash
# Step 4: Compute posterior attributions for breakpoint news
#
# Prerequisites:
#   1. Run step2 converter to generate breakpoints with window_history
#   2. Run step3 crawl_breakpoint_news to add news to each breakpoint
#
# Features:
#   - Incremental writing (saves progress after each market)
#   - Resume support with --skip_existing
#   - Auto-generates output filename with _attributed suffix

# Kalshi
python step4_compute_posterior_attributions.py \
    --input_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl \
    --output_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_kalshi_v2_0102/kalshi_data_processed_with_news_attributed.jsonl \
    --model gpt-4o-mini \
    --max_news 100 \
    --batch_size 10 \
    --skip_existing

# Polymarket
python step4_compute_posterior_attributions.py \
    --input_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_polymarket_v2_0102/polymarket_data_processed_with_news.jsonl \
    --output_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_polymarket_v2_0102/polymarket_data_processed_with_news_attributed.jsonl \
    --model gpt-4o-mini \
    --max_news 100 \
    --batch_size 10 \
    --skip_existing
