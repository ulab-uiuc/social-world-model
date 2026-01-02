import argparse
import os
from pathlib import Path

import jsonlines

from swm.data import KalshiData
from swm.utils.splitter import get_split_stats, split_by_time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split Kalshi data into train/dev/test based on time cutoffs'
    )
    parser.add_argument(
        '--input_file',
        required=True,
        help='Input JSONL file with processed Kalshi data',
    )
    parser.add_argument(
        '--output_dir',
        default='../data/splitted_kalshi',
        help='Output directory for split files',
    )
    parser.add_argument(
        '--train_cutoff',
        type=float,
        required=True,
        help='Unix timestamp - data with end_ts <= this goes to train',
    )
    parser.add_argument(
        '--dev_cutoff',
        type=float,
        required=True,
        help='Unix timestamp - data with end_ts <= this (and > train_cutoff) goes to dev',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(args.input_file).stem
    train_file = os.path.join(args.output_dir, f'{base_name}_train.jsonl')
    dev_file = os.path.join(args.output_dir, f'{base_name}_dev.jsonl')
    test_file = os.path.join(args.output_dir, f'{base_name}_test.jsonl')

    # Load data
    with jsonlines.open(args.input_file, 'r') as reader:
        dataset = [KalshiData.from_dict(data) for data in reader]

    print(f'Loaded {len(dataset)} records from {args.input_file}')

    # Split by time
    train_data, dev_data, test_data = split_by_time(
        dataset, args.train_cutoff, args.dev_cutoff
    )

    # Write output files
    for data, file_path in [
        (train_data, train_file),
        (dev_data, dev_file),
        (test_data, test_file),
    ]:
        with jsonlines.open(file_path, 'w') as writer:
            for item in data:
                writer.write(item.model_dump())

    print(f'\nCreated:')
    print(f'  Train: {train_file}')
    print(f'  Dev:   {dev_file}')
    print(f'  Test:  {test_file}')
    print(f'\nStats: {get_split_stats(train_data, dev_data, test_data)}')


if __name__ == '__main__':
    main()

