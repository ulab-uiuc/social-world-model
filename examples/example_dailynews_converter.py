import argparse
from pathlib import Path
from typing import List

import jsonlines
from tqdm import tqdm

from swm.data import DailyNewsData
from swm.utils.converter import DailyNewsConverter


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DailyNews event data')
    parser.add_argument(
        '--input_file_path',
        type=str,
        default='../data/raw_dailynews/daily_news_2024-01-01_2025-01-02.jsonl',
        help='Path to the input raw DailyNews JSONL file.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/processed_dailynews',
        help='Directory to save the processed DailyNews data.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_file_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(input_path) as reader:
        raw_dataset = list(reader)

    converter = DailyNewsConverter()

    processed_dataset: List[DailyNewsData] = []
    for event_data in tqdm(raw_dataset, desc='Converting DailyNews data'):
        processed_data = converter.convert(event_data)
        if processed_data:
            processed_dataset.append(processed_data)

    processed_output_path = output_dir / 'dailynews_data_processed.jsonl'
    with jsonlines.open(processed_output_path, mode='w') as writer:
        for data in tqdm(processed_dataset, desc='Saving processed DailyNews data'):
            writer.write(data.model_dump())


if __name__ == '__main__':
    main()
