#!/usr/bin/env python3
"""Merge polymarket_data_raw.jsonl and polymarket_event_with_history_data.jsonl into one file."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Merge Polymarket raw data files')
    parser.add_argument(
        '--raw_file',
        type=str,
        default='../data/raw_polymarket/polymarket_data_raw.jsonl',
        help='Path to polymarket_data_raw.jsonl',
    )
    parser.add_argument(
        '--history_file',
        type=str,
        default='../data/raw_polymarket/polymarket_event_with_history_data.jsonl',
        help='Path to polymarket_event_with_history_data.jsonl',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='../data/raw_polymarket/polymarket_merged.jsonl',
        help='Path to output merged file',
    )
    parser.add_argument(
        '--prefer_history',
        action='store_true',
        default=True,
        help='For duplicates, prefer data from history file (has more complete data)',
    )
    return parser.parse_args()


def load_jsonl(file_path: str) -> dict:
    """Load JSONL file and return dict keyed by ID."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            record_id = record.get('id')
            if record_id:
                data[record_id] = record
    return data


def main():
    args = parse_args()

    print(f'Loading {args.raw_file}...')
    raw_data = load_jsonl(args.raw_file)
    print(f'  Loaded {len(raw_data)} records')

    print(f'Loading {args.history_file}...')
    history_data = load_jsonl(args.history_file)
    print(f'  Loaded {len(history_data)} records')

    # Find overlaps
    raw_ids = set(raw_data.keys())
    history_ids = set(history_data.keys())
    overlap_ids = raw_ids & history_ids
    
    print(f'\nOverlap: {len(overlap_ids)} records')
    print(f'Only in raw: {len(raw_ids - history_ids)} records')
    print(f'Only in history: {len(history_ids - raw_ids)} records')

    # Merge: start with raw, then add/override with history
    if args.prefer_history:
        merged = {**raw_data, **history_data}
        print(f'\nMerging (preferring history for duplicates)...')
    else:
        merged = {**history_data, **raw_data}
        print(f'\nMerging (preferring raw for duplicates)...')

    print(f'Total merged: {len(merged)} records')

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for record in merged.values():
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'\n✅ Saved merged data to {args.output_file}')


if __name__ == '__main__':
    main()

