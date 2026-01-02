#!/usr/bin/env python3
"""Split Polymarket data by time - each market's time series is split at cutoff point."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import jsonlines

from swm.data import PolyMarketData
from swm.utils.splitter import get_split_stats, split_dataset_by_time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split Polymarket data by time - splits each market\'s time series at cutoff'
    )
    parser.add_argument(
        '--input_file',
        default='../data/processed_polymarket_v2_0102/polymarket_data_processed.jsonl',
        help='Input JSONL file with processed Polymarket data',
    )
    parser.add_argument(
        '--output_dir',
        default='../data/splitted_polymarket',
        help='Output directory for split files',
    )
    parser.add_argument(
        '--cutoff_date',
        type=str,
        required=True,
        help='Cutoff date in YYYY-MM-DD format (e.g., 2024-10-01)',
    )
    return parser.parse_args()


def date_to_timestamp(date_str: str) -> float:
    """Convert YYYY-MM-DD to Unix timestamp (midnight UTC)."""
    return datetime.strptime(date_str, '%Y-%m-%d').timestamp()


def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cutoff_ts = date_to_timestamp(args.cutoff_date)
    print(f'Cutoff: {args.cutoff_date} (timestamp: {cutoff_ts})')

    base_name = Path(args.input_file).stem
    train_file = output_path / f'{base_name}_train_{args.cutoff_date}.jsonl'
    test_file = output_path / f'{base_name}_test_{args.cutoff_date}.jsonl'

    # Load data
    print(f'Loading data from {args.input_file}...')
    with jsonlines.open(args.input_file, 'r') as reader:
        dataset = [PolyMarketData.from_dict(data) for data in reader]
    print(f'Loaded {len(dataset)} markets')

    # Split by time
    print(f'Splitting at {args.cutoff_date}...')
    train_data, test_data = split_dataset_by_time(dataset, cutoff_ts)

    # Write output files
    with jsonlines.open(train_file, 'w') as writer:
        for item in train_data:
            writer.write(item)
    print(f'Train: {train_file} ({len(train_data)} markets)')

    with jsonlines.open(test_file, 'w') as writer:
        for item in test_data:
            writer.write(item)
    print(f'Test:  {test_file} ({len(test_data)} markets)')

    # Print stats
    stats = get_split_stats(train_data, test_data, cutoff_ts)
    print(f'\nSplit Statistics:')
    for key, value in stats.items():
        print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
