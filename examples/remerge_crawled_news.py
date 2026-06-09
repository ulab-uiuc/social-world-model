#!/usr/bin/env python3
"""
Re-merge crawled news files into daily_breakpoints with correct field mapping.

The crawled news files use:
- 'link' instead of 'url'
- 'date' instead of 'published_at'

This script re-reads the news files and correctly maps the fields.

Usage:
    python remerge_crawled_news.py \
        --input_file ../data/processed_kalshi_v2_0102/kalshi_data_processed.jsonl \
        --news_dir ../data_backup/kalshi_breakpoint_news_v2_0102 \
        --output_file ../data/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl
"""

import argparse
from datetime import datetime
from pathlib import Path

import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Re-merge crawled news files into breakpoints with correct field mapping'
    )
    parser.add_argument(
        '--input_file',
        required=True,
        help='Input JSONL file with processed market data',
    )
    parser.add_argument(
        '--news_dir',
        required=True,
        help='Directory containing crawled news files',
    )
    parser.add_argument(
        '--output_file',
        default=None,
        help='Output JSONL file. Default: input file with _with_news suffix',
    )
    return parser.parse_args()


def load_news_files(news_dir: str) -> dict:
    """
    Load all news files and index them by (market_id, before_date, after_date).

    Expected filename format: {market_id}_{before_date}_to_{after_date}.jsonl
    """
    news_index = {}
    news_path = Path(news_dir)

    if not news_path.exists():
        print(f'News directory not found: {news_dir}')
        return news_index

    news_files = list(news_path.glob('*.jsonl'))
    print(f'Found {len(news_files)} news files in {news_dir}')

    for f in tqdm(news_files, desc='Loading news files'):
        try:
            # Parse filename: {market_id}_{before_date}_to_{after_date}.jsonl
            stem = f.stem  # filename without extension
            parts = stem.rsplit('_to_', 1)
            if len(parts) != 2:
                continue

            prefix = parts[0]
            after_date = parts[1]

            # Find the date part in prefix (last occurrence of YYYY-MM-DD pattern)
            prefix_parts = prefix.rsplit('_', 1)
            if len(prefix_parts) != 2:
                continue

            market_id = prefix_parts[0]
            before_date = prefix_parts[1]

            # Load news articles with correct field mapping
            with jsonlines.open(f, 'r') as reader:
                articles = list(reader)

            # Normalize article format - map 'link' to 'url' and 'date' to 'published_at'
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
                        else str(article.get('source', '')),
                    }
                )

            key = (market_id, before_date, after_date)
            news_index[key] = news_list

        except Exception as e:
            print(f'Error loading {f}: {e}')
            continue

    print(f'Loaded news for {len(news_index)} breakpoints')
    return news_index


def main():
    args = parse_args()

    # Auto-generate output filename if not specified
    if args.output_file is None:
        input_path = Path(args.input_file)
        output_name = input_path.stem + '_with_news' + input_path.suffix
        args.output_file = str(input_path.parent / output_name)

    print(f'Output file: {args.output_file}')

    # Load news files
    news_index = load_news_files(args.news_dir)
    if not news_index:
        print('No news files found, exiting.')
        return

    # Load market data
    print(f'\nLoading market data from {args.input_file}...')
    with jsonlines.open(args.input_file, 'r') as reader:
        markets = list(reader)
    print(f'Loaded {len(markets)} markets')

    # Merge news into breakpoints
    merged_count = 0
    empty_count = 0

    for market in tqdm(markets, desc='Merging news'):
        market_id = str(market.get('market_id', ''))
        breakpoints = market.get('daily_breakpoints', [])

        if not breakpoints:
            continue

        for bp in breakpoints:
            before_ts = bp.get('before', {}).get('t')
            after_ts = bp.get('after', {}).get('t')

            if before_ts is None or after_ts is None:
                bp['news'] = []
                empty_count += 1
                continue

            # Convert timestamps to dates
            before_date = datetime.fromtimestamp(before_ts).strftime('%Y-%m-%d')
            after_date = datetime.fromtimestamp(after_ts).strftime('%Y-%m-%d')

            # Look up news
            key = (market_id, before_date, after_date)
            news = news_index.get(key)

            if news:
                bp['news'] = news
                merged_count += 1
            else:
                # Add empty news list for breakpoints without crawled news
                bp['news'] = []
                empty_count += 1

    # Save output
    print(f'\nSaving to {args.output_file}...')
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(markets)

    print('\nDone!')
    print(f'  Merged: {merged_count} breakpoints (with news)')
    print(f'  Empty: {empty_count} breakpoints (no news found)')
    print(f'  Output: {args.output_file}')


if __name__ == '__main__':
    main()
