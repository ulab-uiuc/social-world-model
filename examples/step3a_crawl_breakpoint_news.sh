python step3_crawl_breakpoint_news.py \
--use_llm_keywords \
--use_gnews \
--z_score_threshold 4.0 \
--llm_model gpt-4o-mini \
--skip_existing \
--input_file ../data/processed_kalshi_v2_0102/kalshi_data_processed.jsonl \
--output_file ../data/processed_kalshi_v2_0102/kalshi_data_processed_breakpoint_with_news.jsonl

python step3_crawl_breakpoint_news.py \
--use_llm_keywords \
--use_gnews \
--z_score_threshold 4.0 \
--llm_model gpt-4o-mini \
--skip_existing \
--input_file ../data/processed_polymarket_v2_0102/polymarket_data_processed.jsonl \
--output_file ../data/processed_kalshi_v2_0102/polymarket_data_processed_breakpoint_with_news.jsonl