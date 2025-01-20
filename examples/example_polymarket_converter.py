import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm

from swm.utils.converter import PolyMarketDataConverter, TimeSeriesConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PolyMarket event data')
    parser.add_argument(
        '--input_file_path', type=str, default='../data/raw/polymarket_data_raw.jsonl'
    )
    parser.add_argument('--output_dir', type=str, default='../data/processed')
    parser.add_argument('--prob_threshold', type=float, default=0.25)
    parser.add_argument('--time_threshold', type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    with jsonlines.open(args.input_file_path) as f:
        dataset = list(f)

    config = TimeSeriesConfig(
        prob_threshold=args.prob_threshold,
        time_threshold=args.time_threshold,
    )
    converter = PolyMarketDataConverter(config)

    processed_dataset = []
    for event_data in tqdm(dataset):
        processed_data = converter.convert(event_data)
        processed_dataset += processed_data

    with jsonlines.open(
        Path(args.output_dir) / 'polymarket_data_processed.jsonl', 'w'
    ) as writer:
        for data in processed_dataset:
            writer.write(data.model_dump())

    processed_dataset_with_categories = {}
    for data in processed_dataset:
        categories = data.categories
        for category in categories:
            if category not in processed_dataset_with_categories:
                processed_dataset_with_categories[category] = []
            processed_dataset_with_categories[category].append(data)

    for category, data in processed_dataset_with_categories.items():
        print(f'Processing category: {category}, num events: {len(data)}')
        with jsonlines.open(
            Path(args.output_dir) / f'polymarket_data_processed_{category}.jsonl', 'w'
        ) as writer:
            for d in data:
                writer.write(d.model_dump())


if __name__ == '__main__':
    main()
