import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm

from swm.utils.converter import (
    PolymarketCategory,
    PolyMarketDataConverter,
    TimeSeriesConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PolyMarket event data')
    parser.add_argument(
        '--input_file_path',
        type=str,
        default='../data/raw_polymarket/polymarket_merged.jsonl',
    )
    parser.add_argument(
        '--output_dir', type=str, default='../data/processed_polymarket_v2_0102_test'
    )
    parser.add_argument('--z_score_threshold', type=float, default=2.0)
    parser.add_argument(
        '--rolling_window',
        type=int,
        default=15,
        help='Window size for rolling robust z-score calculation',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with jsonlines.open(args.input_file_path) as f:
        dataset = list(f)

    config = TimeSeriesConfig(
        z_score_threshold=args.z_score_threshold,
        rolling_window=args.rolling_window,
    )
    converter = PolyMarketDataConverter(config)

    # Valid categories: Crypto, Election, Politics
    {cat.value for cat in PolymarketCategory}

    processed_dataset = []
    seen_market_ids = set()
    for event_data in tqdm(dataset):
        processed_data = converter.convert(event_data)
        # Only keep data with valid categories, deduplicate by market_id
        for data in processed_data:
            if data.categories and data.market_id not in seen_market_ids:
                processed_dataset.append(data)
                seen_market_ids.add(data.market_id)

    print(f'Processed {len(processed_dataset)} unique markets with valid categories')

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all processed data (combined file)
    with jsonlines.open(output_dir / 'polymarket_data_processed.jsonl', 'w') as writer:
        for data in processed_dataset:
            writer.write(data.model_dump())
    print('Saved combined file: polymarket_data_processed.jsonl')

    # Group by category
    processed_dataset_with_categories = {}
    for data in processed_dataset:
        for category in data.categories:
            cat_value = category.value if hasattr(category, 'value') else category
            if cat_value not in processed_dataset_with_categories:
                processed_dataset_with_categories[cat_value] = []
            processed_dataset_with_categories[cat_value].append(data)

    # Save per-category files
    for category, data in processed_dataset_with_categories.items():
        print(f'Processing category: {category}, num events: {len(data)}')
        with jsonlines.open(
            output_dir / f'polymarket_data_processed_{category}.jsonl', 'w'
        ) as writer:
            for d in data:
                writer.write(d.model_dump())


if __name__ == '__main__':
    main()
