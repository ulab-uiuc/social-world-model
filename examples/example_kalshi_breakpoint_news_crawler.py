#!/usr/bin/env python3
"""Crawl Google News for Kalshi breakpoints (-2 days to +1 day around each breakpoint)."""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import jsonlines

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.data import KalshiData
from swm.utils.crawler import GoogleNewsCrawler
from swm.utils.utils import extract_search_keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crawl Google News around Kalshi breakpoints'
    )
    parser.add_argument(
        '--input_file',
        default='../data/processed_kalshi/kalshi_data_processed_Economics.jsonl',
        help='Input JSONL file with processed Kalshi data',
    )
    parser.add_argument(
        '--output_dir',
        default='../data/kalshi_breakpoint_news',
        help='Output directory for news files',
    )
    parser.add_argument(
        '--days_before',
        type=int,
        default=2,
        help='Days before breakpoint to crawl (default: 2)',
    )
    parser.add_argument(
        '--days_after',
        type=int,
        default=0,
        help='Days after breakpoint to crawl (default: 1)',
    )
    parser.add_argument(
        '--max_pages',
        type=int,
        default=5,
        help='Max pages to crawl per query (default: 3)',
    )
    parser.add_argument(
        '--min_delay',
        type=float,
        default=2.0,
        help='Minimum delay between requests (default: 2.0)',
    )
    parser.add_argument(
        '--max_delay',
        type=float,
        default=4.0,
        help='Maximum delay between requests (default: 4.0)',
    )
    parser.add_argument(
        '--z_score_threshold',
        type=float,
        default=2.0,
        help='Minimum z_score to crawl news for (default: 2.0)',
    )
    parser.add_argument(
        '--use_llm_keywords',
        action='store_true',
        help='Use GPT to extract search keywords from questions',
    )
    parser.add_argument(
        '--llm_model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model for keyword extraction (default: gpt-4o-mini)',
    )
    return parser.parse_args()


def extract_breakpoint_queries(data: KalshiData, z_score_threshold: float = 2.0):
    """Extract search queries and date ranges from breakpoints."""
    if not data.daily_breakpoints:
        return []

    queries = []
    for bp in data.daily_breakpoints:
        # Filter by z_score threshold
        z_score = bp.get('z_score', 0)
        if z_score < z_score_threshold:
            continue

        # Get both before and after timestamps
        before_ts = bp.get('before', {}).get('t')
        after_ts = bp.get('after', {}).get('t')
        if before_ts is None or after_ts is None:
            continue

        # Use the market question as the search query
        query = data.question or data.title
        if not query:
            continue

        queries.append({
            'market_id': data.market_id,
            'query': query,
            'before_ts': before_ts,
            'after_ts': after_ts,
            'before_date': datetime.fromtimestamp(before_ts).strftime('%Y-%m-%d'),
            'after_date': datetime.fromtimestamp(after_ts).strftime('%Y-%m-%d'),
            'change': bp.get('change'),
            'z_score': z_score,
        })

    return queries


def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Kalshi data
    with jsonlines.open(args.input_file, 'r') as reader:
        dataset = [KalshiData.from_dict(data) for data in reader]

    print(f'Loaded {len(dataset)} records from {args.input_file}')

    # Extract all breakpoint queries
    all_queries = []
    for data in dataset:
        queries = extract_breakpoint_queries(data, args.z_score_threshold)
        all_queries.extend(queries)

    print(f'Found {len(all_queries)} breakpoints to crawl')

    if not all_queries:
        print('No breakpoints found, exiting.')
        return

    # Initialize crawler
    crawler = GoogleNewsCrawler(min_delay=args.min_delay, max_delay=args.max_delay)

    # Crawl news for each breakpoint
    for i, q in enumerate(all_queries):
        # start_date = before_ts - days_before
        # end_date = after_ts + days_after
        before_date = datetime.fromtimestamp(q['before_ts'])
        after_date = datetime.fromtimestamp(q['after_ts'])
        start_date = (before_date - timedelta(days=args.days_before)).strftime('%Y-%m-%d')
        end_date = (after_date + timedelta(days=args.days_after)).strftime('%Y-%m-%d')

        output_file = os.path.join(
            args.output_dir,
            f"{q['market_id']}_{q['before_date']}_to_{q['after_date']}.jsonl"
        )

        # Extract keywords using LLM if enabled
        search_query = q['query']
        if args.use_llm_keywords:
            keywords = extract_search_keywords(q['query'], model=args.llm_model)
            if keywords:
                search_query = keywords

        print(f"\n[{i + 1}/{len(all_queries)}] Crawling news for: {q['market_id']}")
        print(f"  Original: {q['query'][:60]}...")
        print(f"  Search:   {search_query}")
        print(f"  Breakpoint: {q['before_date']} -> {q['after_date']} (z_score={q['z_score']:.2f}, change={q['change']:.4f})")
        print(f"  Date range: {start_date} to {end_date}")

        try:
            crawler.crawl(
                query=search_query,
                start_date=start_date,
                end_date=end_date,
                output_file=output_file,
                max_pages=args.max_pages,
                fetch_full_content=True,
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f'\nDone! News saved to {args.output_dir}')


if __name__ == '__main__':
    main()

