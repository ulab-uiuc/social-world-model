#!/usr/bin/env python3
"""
Crawl Google News for breakpoints and embed news directly into daily_breakpoints.

Output: Updated market data with news embedded in each breakpoint:
{
    "daily_breakpoints": [
        {
            "before": {"t": ..., "p": ...},
            "after": {"t": ..., "p": ...},
            "window_history": [...],
            "news": [                     # <-- Added by this script
                {"title": "...", "description": "...", "url": "...", "published_at": "..."},
                ...
            ]
        }
    ]
}
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import jsonlines
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.crawler import GNewsCrawler, GoogleNewsCrawler
from swm.utils.utils import extract_search_keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crawl Google News and embed into breakpoints'
    )
    parser.add_argument(
        '--input_file',
        required=True,
        help='Input JSONL file with processed market data',
    )
    parser.add_argument(
        '--output_file',
        required=True,
        help='Output JSONL file with news embedded in breakpoints',
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
        default=-1,
        help='Days after breakpoint to crawl (default: -1, meaning day before after_date)',
    )
    parser.add_argument(
        '--max_results',
        type=int,
        default=20,
        help='Max news articles per breakpoint (default: 20)',
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
    parser.add_argument(
        '--use_gnews',
        action='store_true',
        help='Use GNews API instead of Google News scraping (requires GNEWS_API_KEY)',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip breakpoints that already have news',
    )
    return parser.parse_args()


def crawl_news_for_breakpoint(
    crawler,
    query: str,
    before_ts: float,
    after_ts: float,
    days_before: int,
    days_after: int,
    max_results: int,
    use_gnews: bool,
) -> list:
    """Crawl news for a single breakpoint and return as list of dicts."""
    before_date = datetime.fromtimestamp(before_ts)
    after_date = datetime.fromtimestamp(after_ts)
    start_date = (before_date - timedelta(days=days_before)).strftime('%Y-%m-%d')
    end_date = (after_date + timedelta(days=days_after)).strftime('%Y-%m-%d')
    
    try:
        if use_gnews:
            articles = crawler.fetch(
                query=query,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
            )
        else:
            articles = crawler.fetch(
                query=query,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
            )
        
        # Normalize article format
        news_list = []
        for article in articles:
            news_list.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'published_at': article.get('published_at') or article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else article.get('source', ''),
            })
        return news_list
    except Exception as e:
        print(f"  Error crawling: {e}")
        return []


def main():
    args = parse_args()

    # Load market data
    print(f'Loading data from {args.input_file}...')
    with jsonlines.open(args.input_file, 'r') as reader:
        markets = list(reader)
    print(f'Loaded {len(markets)} markets')

    # Count breakpoints
    total_breakpoints = sum(
        len(m.get('daily_breakpoints', []))
        for m in markets
    )
    valid_breakpoints = sum(
        len([bp for bp in m.get('daily_breakpoints', []) if bp.get('z_score', 0) >= args.z_score_threshold])
        for m in markets
    )
    print(f'Total breakpoints: {total_breakpoints}, above threshold: {valid_breakpoints}')

    # Initialize crawler
    if args.use_gnews:
        api_key = os.environ.get("GNEWS_API_KEY")
        if not api_key:
            print("Error: GNEWS_API_KEY environment variable not set")
            print("Get your free API key at: https://gnews.io/")
            return
        crawler = GNewsCrawler(api_key=api_key)
        print("Using GNews API")
    else:
        crawler = GoogleNewsCrawler(
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
        print("Using Google News scraping")

    # Process each market
    crawled_count = 0
    skipped_count = 0
    
    for market in tqdm(markets, desc='Processing markets'):
        breakpoints = market.get('daily_breakpoints', [])
        if not breakpoints:
            continue
        
        question = market.get('question') or market.get('title', '')
        if not question:
            continue
        
        # Extract keywords if enabled
        search_query = question
        if args.use_llm_keywords:
            keywords = extract_search_keywords(question, model=args.llm_model)
            if keywords:
                search_query = keywords
        
        # Process each breakpoint
        for bp in breakpoints:
            z_score = bp.get('z_score', 0)
            if z_score < args.z_score_threshold:
                continue
            
            # Skip if already has news
            if args.skip_existing and bp.get('news'):
                skipped_count += 1
                continue
            
            before_ts = bp.get('before', {}).get('t')
            after_ts = bp.get('after', {}).get('t')
            if before_ts is None or after_ts is None:
                continue
            
            # Crawl news
            news = crawl_news_for_breakpoint(
                crawler=crawler,
                query=search_query,
                before_ts=before_ts,
                after_ts=after_ts,
                days_before=args.days_before,
                days_after=args.days_after,
                max_results=args.max_results,
                use_gnews=args.use_gnews,
            )
            
            # Embed news into breakpoint
            bp['news'] = news
            crawled_count += 1
            
            if crawled_count % 10 == 0:
                tqdm.write(f'  Crawled {crawled_count} breakpoints, found {len(news)} articles for latest')

    # Save output
    print(f'\nSaving to {args.output_file}...')
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(markets)

    print(f'\nDone!')
    print(f'  Crawled: {crawled_count} breakpoints')
    print(f'  Skipped: {skipped_count} (already had news)')
    print(f'  Output: {args.output_file}')


if __name__ == '__main__':
    main()
