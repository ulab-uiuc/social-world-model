import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm

from swm.utils.converter import KalshiDataConverter, TimeSeriesConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Kalshi market data')
    parser.add_argument(
        '--input_file_path',
        type=str,
        default='../data/raw_kalshi/kalshi_data_raw.jsonl',
    )
    parser.add_argument(
        '--output_dir', type=str, default='../data/processed_kalshi'
    )
    parser.add_argument('--z_score_threshold', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(args.input_file_path) as f:
        dataset = list(f)

    config = TimeSeriesConfig(
        z_score_threshold=args.z_score_threshold,
    )
    converter = KalshiDataConverter(config)

    processed_dataset = []
    for market_data in tqdm(dataset, desc='Converting Kalshi markets'):
        processed_data = converter.convert(market_data)
        processed_dataset += processed_data

    print(f'Processed {len(processed_dataset)} markets from {len(dataset)} entries')

    # Save all processed data
    with jsonlines.open(
        output_dir / 'kalshi_data_processed.jsonl', 'w'
    ) as writer:
        for data in processed_dataset:
            writer.write(data.model_dump())

    # Group by category
    processed_dataset_with_categories = {}
    for data in processed_dataset:
        categories = data.categories or ['Other']
        for category in categories:
            if category not in processed_dataset_with_categories:
                processed_dataset_with_categories[category] = []
            processed_dataset_with_categories[category].append(data)

    # Save per-category files
    for category, data in processed_dataset_with_categories.items():
        print(f'Processing category: {category}, num markets: {len(data)}')
        with jsonlines.open(
            output_dir / f'kalshi_data_processed_{category}.jsonl', 'w'
        ) as writer:
            for d in data:
                writer.write(d.model_dump())


if __name__ == '__main__':
    main()

