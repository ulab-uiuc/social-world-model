import argparse
import os
import random
from pathlib import Path

import jsonlines

from swm.data import PolyMarketData
from swm.utils.splitter import get_split_stats, split_polymarket_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process PolyMarket data files into train/dev/test splits'
    )
    parser.add_argument(
        '--input_files',
        nargs='+',
        default=[
            '../data/processed/polymarket_data_processed_Crypto.jsonl',
            '../data/processed/polymarket_data_processed_Sports.jsonl',
            '../data/processed/polymarket_data_processed_Election.jsonl',
            '../data/processed/polymarket_data_processed_Other.jsonl',
            '../data/processed/polymarket_data_processed_Politics.jsonl',
            '../data/processed/polymarket_data_processed.jsonl',
        ],
    )
    parser.add_argument(
        '--output_dir',
        default='../data/splitted',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )

    return parser.parse_args()


def process_file(input_file: str, output_dir: str) -> None:
    base_name = Path(input_file).stem
    train_file = os.path.join(output_dir, f'{base_name}_train.jsonl')
    dev_file = os.path.join(output_dir, f'{base_name}_dev.jsonl')
    test_file = os.path.join(output_dir, f'{base_name}_test.jsonl')

    with jsonlines.open(input_file, 'r') as reader:
        dataset = list(reader)

    processed_dataset = []
    for data in dataset:
        poly_market_data = PolyMarketData.from_dict(data)
        processed_dataset.append(poly_market_data)

    train_data, dev_data, test_data = split_polymarket_data(processed_dataset)

    for data, file_path in [
        (train_data, train_file),
        (dev_data, dev_file),
        (test_data, test_file),
    ]:
        with jsonlines.open(file_path, 'w') as writer:
            for item in data:
                writer.write(item.model_dump())

    print(f'Processed {input_file}')
    print(f'Created: {train_file}, {dev_file}, {test_file}')

    print(get_split_stats(train_data, dev_data, test_data))


def main():
    args = parse_args()

    random.seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for input_file in args.input_files:
        print(f'Processing {input_file}')
        try:
            process_file(input_file, args.output_dir)
        except Exception as e:
            print(f'Error processing {input_file}: {str(e)}')
            continue


if __name__ == '__main__':
    main()
