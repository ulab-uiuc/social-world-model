"""
Precompute POSTERIOR attributions for market data.

This script uses PosteriorAttributer (GPT-4 + news) to determine which news
items caused each price change. The attributions are saved to the market data
for later use in training.

Pipeline:
    Step 1: precompute_posterior_attributions.py  <-- YOU ARE HERE
            (GPT-4 + news → market.attributions)
    Step 2: train_multievent_forecaster.py
            (Uses precomputed attributions to train forecaster)
    Step 3: train_attributer.py (optional)
            (Train PriorAttributer to predict attributions without news)

Usage:
    python precompute_posterior_attributions.py \\
        --input-data-path ../data/processed/train.jsonl \\
        --output-data-path ../data/attributed/train.jsonl \\
        --corpus-news-path ../data/news/daily_news.jsonl \\
        --attributer-model gpt-4o
"""
import argparse
import json
from pathlib import Path

import jsonlines
from tqdm import tqdm

from swm.utils.posterior_attributer import BasicPosteriorAttributer
from swm.utils.utils import load_dailynews_data, load_polymarket_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute attributions for market data'
    )
    parser.add_argument('--input-data-path', type=str, required=True,
                        help='Path to input market data (jsonl)')
    parser.add_argument('--output-data-path', type=str, required=True,
                        help='Path to output market data with attributions (jsonl)')
    parser.add_argument('--corpus-news-path', type=str, required=True,
                        help='Path to news corpus (jsonl)')
    parser.add_argument('--attributer-model', type=str, default='gpt-4o',
                        help='Model name for attribution (e.g., gpt-4o)')
    parser.add_argument('--max-news-items', type=int, default=10,
                        help='Maximum number of news items to consider')
    parser.add_argument('--cache-dir', type=str, default='./cache/attributions',
                        help='Cache directory for attributions')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for time series')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of markets to process (for testing)')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load data
    print(f"Loading market data from {args.input_data_path}...")
    markets = load_polymarket_data(args.input_data_path)
    if args.limit:
        markets = markets[:args.limit]
    print(f"Loaded {len(markets)} markets")
    
    print(f"Loading news corpus from {args.corpus_news_path}...")
    corpus_news = load_dailynews_data(args.corpus_news_path)
    print(f"Loaded {len(corpus_news)} news items")
    
    # Initialize attributer
    attributer = BasicPosteriorAttributer(
        corpus_news=corpus_news,
        model_name=args.attributer_model,
        max_news_items=args.max_news_items,
        cache_dir=args.cache_dir,
    )
    
    # Process each market
    output_path = Path(args.output_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_markets = []
    for market in tqdm(markets, desc='Computing attributions'):
        # Skip markets without time series
        if not market.daily_time_series or 'Yes' not in market.daily_time_series:
            processed_markets.append(market)
            continue
        
        series = market.daily_time_series['Yes']
        if len(series) <= args.window_size:
            processed_markets.append(market)
            continue
        
        # Compute attributions for each valid timestamp
        attributions = {}
        for start_idx in range(len(series) - args.window_size):
            target = series[start_idx + args.window_size]
            target_ts = str(target['t'])
            
            try:
                events = attributer.attribute(target['t'], market)
                if events:
                    attributions[target_ts] = [
                        {
                            'news': e['news'].model_dump() if hasattr(e['news'], 'model_dump') else e['news'],
                            'score': e['score']
                        }
                        for e in events
                    ]
            except Exception as e:
                print(f"Error computing attribution for {market.market_id} at {target_ts}: {e}")
                continue
        
        # Add attributions to market
        market.attributions = attributions
        processed_markets.append(market)
    
    # Save results
    print(f"Saving {len(processed_markets)} markets to {args.output_data_path}...")
    with jsonlines.open(output_path, mode='w') as writer:
        for market in processed_markets:
            writer.write(market.model_dump())
    
    # Print statistics
    markets_with_attr = sum(1 for m in processed_markets if m.attributions)
    total_attributions = sum(len(m.attributions or {}) for m in processed_markets)
    print(f"\nStatistics:")
    print(f"  Markets with attributions: {markets_with_attr}/{len(processed_markets)}")
    print(f"  Total attribution timestamps: {total_attributions}")


if __name__ == '__main__':
    main()

