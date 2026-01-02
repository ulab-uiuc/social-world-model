#!/bin/bash
#
# Precompute posterior attributions for breakpoint news
#
# Prerequisites:
#   1. Run converter to generate breakpoints with window_history
#   2. Run crawl_breakpoint_news.py to add news to each breakpoint
#
# Example:
#   bash precompute_posterior_attributions.sh

set -e

cd "$(dirname "$0")"

# Configuration
INPUT_FILE="${INPUT_FILE:-../data/with_news/train.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-../data/attributed/train.jsonl}"
MODEL="${MODEL:-gpt-4o-mini}"
MAX_NEWS="${MAX_NEWS:-10}"

echo "================================"
echo "Precompute Posterior Attributions"
echo "================================"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model:  $MODEL"
echo ""

python precompute_posterior_attributions.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model "$MODEL" \
    --max_news "$MAX_NEWS" \
    --skip_existing

echo ""
echo "Done! Attributed data saved to $OUTPUT_FILE"
