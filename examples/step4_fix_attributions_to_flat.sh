#!/bin/bash
# Add attributions to merged flat file
# - Breakpoints: use existing attributions
# - Normal points: set all to 0
# Output: new file with _attribution suffix

DATA_DIR="/mnt/data_from_server1/haofeiy2/social-world-model/data"

# Kalshi
echo "Adding attributions to Kalshi data..."
python step4_fix_attributions_to_flat.py \
    --merged_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl" \
    --breakpoints_attributed_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_breakpoint_with_news_attributed.jsonl" \
    --output_file "${DATA_DIR}/processed_kalshi_v2_0102/kalshi_data_processed_with_news_attributed.jsonl"

echo ""

# Polymarket (if exists)
if [ -f "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_with_news.jsonl" ]; then
    echo "Adding attributions to Polymarket data..."
    python step4_fix_attributions_to_flat.py \
        --merged_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_with_news.jsonl" \
        --breakpoints_attributed_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_breakpoint_with_news_attributed.jsonl" \
        --output_file "${DATA_DIR}/processed_polymarket_v2_0102/polymarket_data_processed_with_news_attributed.jsonl"
fi

echo ""
echo "Done!"
echo "Output files:"
echo "  - kalshi_data_processed_with_news_attribution.jsonl"
echo "  - polymarket_data_processed_with_news_attribution.jsonl (if exists)"
