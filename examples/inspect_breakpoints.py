#!/usr/bin/env python3
"""
Inspect breakpoints in processed market data with news.

Usage:
    python inspect_breakpoints.py --input_file ../data/kalshi_with_news.jsonl
    python inspect_breakpoints.py --input_file ../data/kalshi_with_news.jsonl --market_id 12345
    python inspect_breakpoints.py --input_file ../data/kalshi_with_news.jsonl --show_news --limit 3
"""
import argparse
import json
from datetime import datetime

import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect breakpoints in market data')
    parser.add_argument(
        '--input_file',
        default='../data/processed_kalshi_v2_0102/kalshi_data_processed.jsonl',
        help='Input JSONL file with processed market data',
    )
    parser.add_argument(
        '--market_id',
        type=str,
        default=None,
        help='Filter by specific market ID',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of markets to show (default: 5)',
    )
    parser.add_argument(
        '--show_news',
        action='store_true',
        help='Show news articles for each breakpoint',
    )
    parser.add_argument(
        '--min_news',
        type=int,
        default=0,
        help='Only show breakpoints with at least N news articles',
    )
    parser.add_argument(
        '--stats_only',
        action='store_true',
        help='Only show statistics, no individual breakpoints',
    )
    return parser.parse_args()


def format_timestamp(ts):
    """Convert timestamp to readable date string."""
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')


def print_breakpoint(bp, idx, show_news=False):
    breakpoint()
    """Print a single breakpoint."""
    before = bp.get('before', {})
    after = bp.get('after', {})
    news = bp.get('news', [])
    window_history = bp.get('window_history', [])
    
    print(f"\n  Breakpoint {idx + 1}:")
    print(f"    Before: {format_timestamp(before.get('t'))} | Price: {before.get('p', 'N/A'):.3f}")
    print(f"    After:  {format_timestamp(after.get('t'))} | Price: {after.get('p', 'N/A'):.3f}")
    print(f"    Change: {bp.get('change', 0):.3f} | Z-score: {bp.get('z_score', 0):.2f}")
    print(f"    Window: {format_timestamp(bp.get('window_start'))} to {format_timestamp(bp.get('window_end'))}")
    print(f"    Window history points: {len(window_history)}")
    print(f"    News articles: {len(news)}")
    
    if show_news and news:
        print("    --- News ---")
        for i, article in enumerate(news[:5]):  # Show max 5 articles
            title = article.get('title', 'No title')[:80]
            source = article.get('source', 'Unknown')
            pub_date = article.get('published_at', 'Unknown date')
            print(f"      [{i+1}] {title}...")
            print(f"          Source: {source} | Date: {pub_date}")
        if len(news) > 5:
            print(f"      ... and {len(news) - 5} more articles")


def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    with jsonlines.open(args.input_file, 'r') as reader:
        markets = list(reader)
    print(f"Loaded {len(markets)} markets\n")
    
    # Compute statistics
    total_breakpoints = 0
    breakpoints_with_news = 0
    breakpoints_empty_news = 0
    breakpoints_no_field = 0
    total_news_articles = 0
    
    for market in markets:
        for bp in market.get('daily_breakpoints', []):
            total_breakpoints += 1
            if 'news' in bp:
                if bp['news']:
                    breakpoints_with_news += 1
                    total_news_articles += len(bp['news'])
                else:
                    breakpoints_empty_news += 1
            else:
                breakpoints_no_field += 1
    
    # Print statistics
    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total markets: {len(markets)}")
    print(f"Total breakpoints: {total_breakpoints}")
    print(f"  With news: {breakpoints_with_news} ({100*breakpoints_with_news/total_breakpoints:.1f}%)")
    print(f"  Empty news []: {breakpoints_empty_news} ({100*breakpoints_empty_news/total_breakpoints:.1f}%)")
    print(f"  No news field: {breakpoints_no_field} ({100*breakpoints_no_field/total_breakpoints:.1f}%)")
    print(f"Total news articles: {total_news_articles}")
    if breakpoints_with_news > 0:
        print(f"Avg articles per breakpoint (with news): {total_news_articles/breakpoints_with_news:.1f}")
    print("=" * 60)
    
    if args.stats_only:
        return
    
    # Filter markets
    if args.market_id:
        markets = [m for m in markets if str(m.get('market_id')) == args.market_id]
        if not markets:
            print(f"\nNo market found with ID: {args.market_id}")
            return
    
    # Show individual breakpoints
    shown = 0
    for market in markets:
        if shown >= args.limit:
            break
        
        breakpoints = market.get('daily_breakpoints', [])
        if not breakpoints:
            continue
        
        # Filter by min_news if specified
        if args.min_news > 0:
            breakpoints = [bp for bp in breakpoints if len(bp.get('news', [])) >= args.min_news]
            if not breakpoints:
                continue
        
        print(f"\n{'='*60}")
        print(f"Market ID: {market.get('market_id')}")
        print(f"Question: {market.get('question', market.get('title', 'N/A'))[:100]}...")
        print(f"Category: {market.get('category', 'N/A')}")
        print(f"Breakpoints: {len(market.get('daily_breakpoints', []))}")
        
        for i, bp in enumerate(breakpoints[:3]):  # Show max 3 breakpoints per market
            print_breakpoint(bp, i, show_news=args.show_news)
        
        if len(breakpoints) > 3:
            print(f"\n  ... and {len(breakpoints) - 3} more breakpoints")
        
        shown += 1
    
    print(f"\n{'='*60}")
    print(f"Showed {shown} markets")


if __name__ == '__main__':
    main()

