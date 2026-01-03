#!/usr/bin/env python3
"""
Step 5: Split market data by time - each market's time series is split at cutoff point.

Usage:
    python step5_split_data.py --input_file xxx_with_news.jsonl --cutoff_date 2025-11-01

Output:
    xxx_with_news_train_2025-11-01.jsonl
    xxx_with_news_test_2025-11-01.jsonl
    xxx_with_news_test_2025-11-01_by_category/
        politics.jsonl
        crypto.jsonl
        ...
"""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import jsonlines

from swm.utils.splitter import split_dataset_by_time


def parse_args():
    parser = argparse.ArgumentParser(description='Split market data by time')
    parser.add_argument('--input_file', required=True, help='Input JSONL file (with_news)')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: same as input)')
    parser.add_argument('--cutoff_date', required=True, help='Cutoff date (YYYY-MM-DD)')
    return parser.parse_args()


class SimpleMarketData:
    """Simple wrapper for market data."""
    def __init__(self, data: dict):
        self._data = data
    
    def model_dump(self) -> dict:
        return self._data


def get_categories(data: dict) -> list:
    """Extract all categories from market data."""
    if data.get('categories') and len(data['categories']) > 0:
        return data['categories']
    if data.get('category'):
        return [data['category']]
    return ['unknown']


def sanitize_filename(name: str) -> str:
    """Sanitize category name for filename."""
    name = re.sub(r'[^\w\-]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name.lower() or 'unknown'


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_ts = datetime.strptime(args.cutoff_date, '%Y-%m-%d').timestamp()

    # Output files
    base_name = input_path.stem
    train_file = output_dir / f'{base_name}_train_{args.cutoff_date}.jsonl'
    test_file = output_dir / f'{base_name}_test_{args.cutoff_date}.jsonl'

    # Load
    print(f'Loading {args.input_file}...')
    with jsonlines.open(args.input_file) as reader:
        dataset = [SimpleMarketData(data) for data in reader]
    print(f'Loaded {len(dataset)} markets')

    # Split
    train_data, test_data = split_dataset_by_time(dataset, cutoff_ts)

    # Write train (all)
    with jsonlines.open(train_file, 'w') as w:
        for item in train_data:
            w.write(item)
    print(f'Train: {train_file} ({len(train_data)})')

    # Write test (all)
    with jsonlines.open(test_file, 'w') as w:
        for item in test_data:
            w.write(item)
    print(f'Test:  {test_file} ({len(test_data)})')

    # Group train by category (each item goes to ALL its categories)
    train_by_cat = defaultdict(list)
    for item in train_data:
        for cat in get_categories(item):
            train_by_cat[cat].append(item)

    # Write train by category
    print(f'\nTrain by category (markets can appear in multiple categories):')
    for cat, items in sorted(train_by_cat.items(), key=lambda x: -len(x[1])):
        cat_file = output_dir / f'{base_name}_train_{args.cutoff_date}_{sanitize_filename(cat)}.jsonl'
        with jsonlines.open(cat_file, 'w') as w:
            for item in items:
                w.write(item)
        print(f'  {cat}: {len(items)} -> {cat_file.name}')

    # Group test by category (each item goes to ALL its categories)
    test_by_cat = defaultdict(list)
    for item in test_data:
        for cat in get_categories(item):
            test_by_cat[cat].append(item)

    # Write test by category
    print(f'\nTest by category (markets can appear in multiple categories):')
    for cat, items in sorted(test_by_cat.items(), key=lambda x: -len(x[1])):
        cat_file = output_dir / f'{base_name}_test_{args.cutoff_date}_{sanitize_filename(cat)}.jsonl'
        with jsonlines.open(cat_file, 'w') as w:
            for item in items:
                w.write(item)
        print(f'  {cat}: {len(items)} -> {cat_file.name}')


if __name__ == '__main__':
    main()

