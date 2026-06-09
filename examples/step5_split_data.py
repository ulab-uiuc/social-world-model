#!/usr/bin/env python3
"""
Step 5: Split flat sample data by time.

Each line in input is a single sample (breakpoint or normal_point).
Split based on the sample's timestamp.

Usage:
    python step5_split_data.py --input_file xxx_with_news.jsonl --cutoff_date 2025-11-01

Output:
    xxx_with_news_train_2025-11-01.jsonl
    xxx_with_news_test_2025-11-01.jsonl
    xxx_with_news_train_2025-11-01_politics.jsonl
    xxx_with_news_test_2025-11-01_politics.jsonl
    ...
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description='Split flat sample data by time')
    parser.add_argument(
        '--input_file', required=True, help='Input JSONL file (flat format)'
    )
    parser.add_argument(
        '--output_dir', default=None, help='Output directory (default: same as input)'
    )
    parser.add_argument('--cutoff_date', required=True, help='Cutoff date (YYYY-MM-DD)')
    return parser.parse_args()


def get_sample_timestamp(sample: dict) -> float:
    """Get timestamp from a sample (breakpoint or normal_point).

    Both types now have the same structure with 'after' field.
    """
    # Use the 'after' timestamp (when the change/observation happened)
    after = sample.get('after', {})
    return after.get('t', 0)


def get_categories(sample: dict) -> list:
    """Extract all categories from sample."""
    if sample.get('categories') and len(sample['categories']) > 0:
        return sample['categories']
    if sample.get('category'):
        return [sample['category']]
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

    # Load and split
    print(f'Loading {args.input_file}...')
    train_samples = []
    test_samples = []

    with jsonlines.open(args.input_file) as reader:
        for sample in reader:
            ts = get_sample_timestamp(sample)
            if ts < cutoff_ts:
                train_samples.append(sample)
            else:
                test_samples.append(sample)

    total = len(train_samples) + len(test_samples)
    print(f'Loaded {total} samples')

    # Write train (all)
    with jsonlines.open(train_file, 'w') as w:
        for sample in train_samples:
            w.write(sample)
    print(f'Train: {train_file} ({len(train_samples)} samples)')

    # Write test (all)
    with jsonlines.open(test_file, 'w') as w:
        for sample in test_samples:
            w.write(sample)
    print(f'Test:  {test_file} ({len(test_samples)} samples)')

    # Count by sample type
    train_breakpoints = sum(
        1 for s in train_samples if s.get('sample_type') == 'breakpoint'
    )
    train_normal = sum(
        1 for s in train_samples if s.get('sample_type') == 'normal_point'
    )
    test_breakpoints = sum(
        1 for s in test_samples if s.get('sample_type') == 'breakpoint'
    )
    test_normal = sum(1 for s in test_samples if s.get('sample_type') == 'normal_point')

    print('\nSample type breakdown:')
    print(f'  Train: {train_breakpoints} breakpoints, {train_normal} normal_points')
    print(f'  Test:  {test_breakpoints} breakpoints, {test_normal} normal_points')

    # Group train by category
    train_by_cat = defaultdict(list)
    for sample in train_samples:
        for cat in get_categories(sample):
            train_by_cat[cat].append(sample)

    # Write train by category
    print('\nTrain by category:')
    for cat, samples in sorted(train_by_cat.items(), key=lambda x: -len(x[1])):
        cat_file = (
            output_dir
            / f'{base_name}_train_{args.cutoff_date}_{sanitize_filename(cat)}.jsonl'
        )
        with jsonlines.open(cat_file, 'w') as w:
            for sample in samples:
                w.write(sample)
        bp_count = sum(1 for s in samples if s.get('sample_type') == 'breakpoint')
        np_count = sum(1 for s in samples if s.get('sample_type') == 'normal_point')
        print(
            f'  {cat}: {len(samples)} ({bp_count} bp, {np_count} np) -> {cat_file.name}'
        )

    # Group test by category
    test_by_cat = defaultdict(list)
    for sample in test_samples:
        for cat in get_categories(sample):
            test_by_cat[cat].append(sample)

    # Write test by category
    print('\nTest by category:')
    for cat, samples in sorted(test_by_cat.items(), key=lambda x: -len(x[1])):
        cat_file = (
            output_dir
            / f'{base_name}_test_{args.cutoff_date}_{sanitize_filename(cat)}.jsonl'
        )
        with jsonlines.open(cat_file, 'w') as w:
            for sample in samples:
                w.write(sample)
        bp_count = sum(1 for s in samples if s.get('sample_type') == 'breakpoint')
        np_count = sum(1 for s in samples if s.get('sample_type') == 'normal_point')
        print(
            f'  {cat}: {len(samples)} ({bp_count} bp, {np_count} np) -> {cat_file.name}'
        )


if __name__ == '__main__':
    main()
