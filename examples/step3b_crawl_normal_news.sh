#!/bin/bash
# Crawl news for normal (non-breakpoint) days as negative samples

# Kalshi
python step3b_crawl_normal_news.py \
    --use_llm_keywords \
    --use_gnews \
    --samples_per_market 3 \
    --min_days_from_breakpoint 3 \
    --llm_model gpt-4o-mini \
    --skip_existing \
    --input_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_kalshi_v2_0102/kalshi_data_processed.jsonl \
    --output_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_kalshi_v2_0102/kalshi_normalpoint_with_news.jsonl

# Polymarket
python step3b_crawl_normal_news.py \
    --use_llm_keywords \
    --use_gnews \
    --samples_per_market 3 \
    --min_days_from_breakpoint 3 \
    --llm_model gpt-4o-mini \
    --skip_existing \
    --input_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_polymarket_v2_0102/polymarket_data_processed.jsonl \
    --output_file /mnt/data_from_server1/haofeiy2/social-world-model/data/processed_polymarket_v2_0102/polymarket_normalpoint_with_news.jsonl
