#!/usr/bin/env python3
"""
Crawl Google News for normal (non-breakpoint) days as negative samples.

This script randomly samples days where there was NO significant price change
and crawls news for those days. These serve as negative samples for training:
- Positive samples: breakpoints with news (price changed after news)
- Negative samples: normal days with news (price did NOT change after news)

Usage:
    python step3b_crawl_normal_news.py \
        --input_file ../data/processed_kalshi_v2_0102/kalshi_data_processed.jsonl \
        --output_file ../data/processed_kalshi_v2_0102/kalshi_normal_points_with_news.jsonl \
        --use_gnews

Output format (separate file for negative samples):
{
    "market_id": "...",
    "question": "...",
    "normal_points": [
        {
            "date": "2024-01-15",
            "timestamp": 1705276800,
            "price": 0.65,
            "price_change": 0.01,  # small change (non-breakpoint)
            "news": [
                {"title": "...", "description": "...", "url": "...", "published_at": "..."},
                ...
            ]
        },
        ...
    ]
}
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import jsonlines
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.crawler import GNewsCrawler, GoogleNewsCrawler
from swm.utils.utils import extract_search_keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crawl Google News for normal (non-breakpoint) days as negative samples'
    )
    parser.add_argument(
        '--input_file',
        default='../data/processed_kalshi_v2_0102/kalshi_data_processed.jsonl',
        help='Input JSONL file with processed market data',
    )
    parser.add_argument(
        '--output_file',
        default='../data/processed_kalshi_v2_0102/kalshi_normal_points_with_news.jsonl',
        help='Output JSONL file with normal points and news',
    )
    parser.add_argument(
        '--samples_per_market',
        type=int,
        default=3,
        help='Number of normal points to sample per market (default: 3)',
    )
    parser.add_argument(
        '--days_before',
        type=int,
        default=2,
        help='Days before the point to crawl news (default: 2)',
    )
    parser.add_argument(
        '--days_after',
        type=int,
        default=0,
        help='Days after the point to crawl news (default: 0)',
    )
    parser.add_argument(
        '--max_results',
        type=int,
        default=1000,
        help='Max news articles per point (default: 1000)',
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
        '--price_change_threshold',
        type=float,
        default=0.05,
        help='Maximum absolute price change for normal points (default: 0.05, i.e. 5%%)',
    )
    parser.add_argument(
        '--min_days_from_breakpoint',
        type=int,
        default=3,
        help='Minimum days away from any breakpoint (default: 3)',
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
    parser.add_argument(
        '--use_gnews',
        action='store_true',
        help='Use GNews API instead of Google News scraping (requires GNEWS_API_KEY)',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip markets that already exist in output file',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    return parser.parse_args()


def get_breakpoint_timestamps(market: Dict) -> Set[int]:
    """Get timestamps of all breakpoints (in days since epoch)."""
    breakpoint_days = set()
    for bp in market.get('daily_breakpoints', []):
        before_ts = bp.get('before', {}).get('t')
        after_ts = bp.get('after', {}).get('t')
        if before_ts:
            # Convert to day (ignore time)
            day = int(before_ts // 86400)
            breakpoint_days.add(day)
        if after_ts:
            day = int(after_ts // 86400)
            breakpoint_days.add(day)
    return breakpoint_days


def sample_normal_points(
    market: Dict,
    num_samples: int,
    price_change_threshold: float,
    min_days_from_breakpoint: int,
    window_size: int = 15,
) -> List[Dict]:
    """
    Sample normal points from daily_time_series that are:
    1. Small price change (below threshold)
    2. At least min_days_from_breakpoint away from any breakpoint
    3. Have enough history (at least window_size points)

    Returns points with same structure as breakpoints:
    - before, after, change, z_score, window_start, window_end, window_history
    """
    daily_ts = market.get('daily_time_series', [])
    if not daily_ts or len(daily_ts) < 5:
        return []

    # Get breakpoint days to avoid
    breakpoint_days = get_breakpoint_timestamps(market)

    # Expand breakpoint days to include buffer zone
    excluded_days = set()
    for bp_day in breakpoint_days:
        for offset in range(-min_days_from_breakpoint, min_days_from_breakpoint + 1):
            excluded_days.add(bp_day + offset)

    # Find candidate normal points
    candidates = []
    for i, point in enumerate(daily_ts):
        ts = point.get('t')
        if ts is None:
            continue

        day = int(ts // 86400)

        # Skip if too close to breakpoint
        if day in excluded_days:
            continue

        # Skip first few days (need some history for window)
        if i < window_size:
            continue

        # Calculate price change from previous day
        prev_point = daily_ts[i - 1]
        prev_ts = prev_point.get('t')
        prev_price = prev_point.get('p', prev_point.get('yes_price', 0))
        curr_price = point.get('p', point.get('yes_price', 0))

        if prev_price is None or curr_price is None or prev_ts is None:
            continue

        price_change = curr_price - prev_price
        abs_change = abs(price_change)

        # We want points with small price changes (normal days)
        if abs_change < price_change_threshold:
            # Build window_history (same structure as breakpoints)
            # Include from start_idx to after point (i+1), so range is [start_idx, i+2)
            window_start_idx = max(0, i - window_size)
            window_end_idx = i + 2  # Include both before (i) and after (i+1) points
            window_history = [
                {
                    't': daily_ts[j].get('t'),
                    'p': daily_ts[j].get('p', daily_ts[j].get('yes_price')),
                }
                for j in range(window_start_idx, min(window_end_idx, len(daily_ts)))
            ]

            # Calculate a simple z_score based on rolling std
            if len(window_history) > 2:
                prices = [p['p'] for p in window_history if p['p'] is not None]
                if len(prices) > 2:
                    import statistics

                    try:
                        std = statistics.stdev(prices[:-1])  # Exclude current point
                        z_score = abs_change / std if std > 0 else 0
                    except Exception:
                        z_score = 0
                else:
                    z_score = 0
            else:
                z_score = 0

            candidates.append(
                {
                    # Same structure as breakpoints
                    'before': {'t': prev_ts, 'p': prev_price},
                    'after': {'t': ts, 'p': curr_price},
                    'change': price_change,
                    'z_score': z_score,
                    'window_start': window_history[0]['t']
                    if window_history
                    else prev_ts,
                    'window_end': ts,
                    'window_history': window_history,
                    # Keep index for internal use
                    '_index': i,
                }
            )

    # Randomly sample from candidates
    if len(candidates) <= num_samples:
        sampled = candidates
    else:
        sampled = random.sample(candidates, num_samples)

    # Remove internal fields
    for point in sampled:
        point.pop('_index', None)

    return sampled


def crawl_news_for_point(
    crawler,
    query: str,
    timestamp: float,
    days_before: int,
    days_after: int,
    max_results: int,
    use_gnews: bool,
) -> List[Dict]:
    """Crawl news for a single normal point."""
    point_date = datetime.fromtimestamp(timestamp)
    start_date = (point_date - timedelta(days=days_before)).strftime('%Y-%m-%d')
    end_date = (point_date + timedelta(days=days_after)).strftime('%Y-%m-%d')

    try:
        articles = crawler.fetch(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )

        # Normalize article format
        news_list = []
        for article in articles:
            news_list.append(
                {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url') or article.get('link', ''),
                    'published_at': article.get('published_at')
                    or article.get('publishedAt')
                    or article.get('date', ''),
                    'source': article.get('source', {}).get('name', '')
                    if isinstance(article.get('source'), dict)
                    else article.get('source', ''),
                }
            )
        return news_list
    except Exception as e:
        print(f'  Error crawling: {e}')
        return []


def load_processed_market_ids(output_file: str) -> Set[str]:
    """Load market IDs that have already been processed."""
    processed_ids = set()
    if Path(output_file).exists():
        try:
            with jsonlines.open(output_file, 'r') as reader:
                for market in reader:
                    market_id = market.get('market_id')
                    if market_id is not None:
                        processed_ids.add(str(market_id))
        except Exception as e:
            print(f'Warning: Error reading existing output file: {e}')
    return processed_ids


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Prepare output file
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load already processed market IDs for resume support
    processed_ids = set()
    if args.skip_existing and Path(args.output_file).exists():
        processed_ids = load_processed_market_ids(args.output_file)
        print(f'Found {len(processed_ids)} already processed markets in output file')

    # Load market data
    print(f'Loading data from {args.input_file}...')
    with jsonlines.open(args.input_file, 'r') as reader:
        markets = list(reader)
    print(f'Loaded {len(markets)} markets')

    # Count markets with enough data
    eligible_markets = sum(
        1 for m in markets if len(m.get('daily_time_series', [])) >= 10
    )
    print(f'Markets with enough time series data: {eligible_markets}')

    # Initialize crawler
    if args.use_gnews:
        api_key = os.environ.get('GNEWS_API_KEY')
        if not api_key:
            print('Error: GNEWS_API_KEY environment variable not set')
            print('Get your free API key at: https://gnews.io/')
            return
        crawler = GNewsCrawler(api_key=api_key)
        print('Using GNews API')
    else:
        crawler = GoogleNewsCrawler(
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
        print('Using Google News scraping')

    # Process each market
    crawled_count = 0
    skipped_markets = 0
    markets_with_samples = 0

    # Open output file in append mode
    write_mode = 'a' if processed_ids else 'w'

    with jsonlines.open(args.output_file, mode=write_mode) as writer:
        for market in tqdm(markets, desc='Processing markets'):
            market_id = str(market.get('market_id', ''))

            # Skip if already processed
            if market_id in processed_ids:
                skipped_markets += 1
                continue

            question = market.get('question') or market.get('title', '')
            if not question:
                continue

            # Sample normal points
            normal_points = sample_normal_points(
                market=market,
                num_samples=args.samples_per_market,
                price_change_threshold=args.price_change_threshold,
                min_days_from_breakpoint=args.min_days_from_breakpoint,
            )

            if not normal_points:
                continue

            # Extract keywords if enabled
            search_query = question
            if args.use_llm_keywords:
                keywords = extract_search_keywords(question, model=args.llm_model)
                if keywords:
                    search_query = keywords

            # Crawl news for each normal point
            for point in normal_points:
                # Get timestamp from 'after' field (new format)
                point_ts = point['after']['t']
                news = crawl_news_for_point(
                    crawler=crawler,
                    query=search_query,
                    timestamp=point_ts,
                    days_before=args.days_before,
                    days_after=args.days_after,
                    max_results=args.max_results,
                    use_gnews=args.use_gnews,
                )
                point['news'] = news
                crawled_count += 1

                if news:
                    point_date = datetime.fromtimestamp(point_ts).strftime('%Y-%m-%d')
                    tqdm.write(
                        f'Found {len(news)} articles for normal point on {point_date}'
                    )

            # Write result
            result = {
                'market_id': market_id,
                'event_id': market.get('event_id', ''),
                'question': question,
                'categories': market.get('categories', []),
                'normal_points': normal_points,
            }
            writer.write(result)
            writer._fp.flush()  # Flush to disk immediately
            markets_with_samples += 1

    print('\nDone!')
    print(f'  Markets with samples: {markets_with_samples}')
    print(f'  Total points crawled: {crawled_count}')
    print(f'  Skipped markets: {skipped_markets} (already in output)')
    print(f'  Output: {args.output_file}')


if __name__ == '__main__':
    main()
