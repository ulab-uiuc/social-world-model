"""Generate PriorAttributer attributions for training/test data.

This script runs a trained PriorAttributer on data that has news but may not
have attributions (or has GPT attributions we want to replace). The output is
a JSONL file with PriorAttributer-generated attributions.

Usage:
    python generate_prior_attributions.py \
        --data-path /path/to/data.jsonl \
        --attributer-path /path/to/prior_attributer/checkpoint \
        --model-name Qwen/Qwen3-0.6B \
        --output-path /path/to/output.jsonl
"""
import argparse
import json

import jsonlines
from tqdm import tqdm

from swm.attributer import BasicPriorAttributer
from swm.utils.utils import load_flat_samples_as_markets, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Generate PriorAttributer attributions')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--attributer-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data as markets
    print(f"Loading data from {args.data_path}...")
    markets = load_flat_samples_as_markets(args.data_path)
    print(f"Loaded {len(markets)} markets")

    # Load attributer
    print(f"Loading attributer from {args.attributer_path}...")
    attributer = BasicPriorAttributer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
    )
    attributer.load(args.attributer_path)

    # Generate attributions for each breakpoint
    total_bps = 0
    for market in tqdm(markets, desc='Generating attributions'):
        if not market.daily_breakpoints:
            continue
        for bp in market.daily_breakpoints:
            news_list = bp.get('news', [])
            if not news_list or len(news_list) < 2:
                continue
            attributions = attributer.attribute_breakpoint(market, bp)
            if attributions:
                bp['attributions'] = [
                    {'news_idx': attr['news_idx'], 'score': attr['score']}
                    for attr in attributions
                ]
                total_bps += 1

    print(f"Generated attributions for {total_bps} breakpoints")

    # Write back as flat samples (one per breakpoint, same format as input)
    print(f"Writing output to {args.output_path}...")
    with jsonlines.open(args.output_path, mode='w') as writer:
        for market in markets:
            if not market.daily_breakpoints:
                continue
            for bp in market.daily_breakpoints:
                sample = {
                    'event_id': market.event_id,
                    'market_id': market.market_id,
                    'question': market.question,
                    'description': market.description,
                    'category': getattr(market, 'category', None),
                    'sample_type': bp.get('sample_type', 'breakpoint'),
                    'before': bp.get('before', {}),
                    'after': bp.get('after', {}),
                    'window_history': bp.get('window_history', []),
                    'news': bp.get('news', []),
                    'attributions': bp.get('attributions', []),
                }
                writer.write(sample)

    print(f"Done! Output: {args.output_path}")


if __name__ == '__main__':
    main()
