import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Input and output file paths
TRAIN_PATH = 'swm_bench_train_new.jsonl'
TEST_PATH = 'swm_bench_test_and_dev_new.jsonl'


def load_jsonl(path: str) -> List[Dict]:
    """
    Load data from a .jsonl file
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def get_market_start_end_times(data: List[Dict]) -> List[Tuple[float, float]]:
    """
    Extract start and end timestamps for each market
    """
    time_ranges = []
    for item in data:
        start_ts = item.get('start_ts')
        end_ts = item.get('end_ts')
        if start_ts is not None and end_ts is not None:
            time_ranges.append((start_ts, end_ts))
    return time_ranges


def check_overlap(
    train_ranges: List[Tuple[float, float]], test_ranges: List[Tuple[float, float]]
) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Check for time overlaps between train and test data
    """
    overlap_count = 0
    overlapping_markets = []

    for train_start, train_end in train_ranges:
        for test_start, test_end in test_ranges:
            # Overlap condition: (start1 <= end2) and (start2 <= end1)
            if max(train_start, test_start) < min(train_end, test_end):
                overlap_count += 1
                overlapping_markets.append(
                    (max(train_start, test_start), min(train_end, test_end))
                )

    return overlap_count, overlapping_markets


def visualize_time_ranges(
    train_ranges: List[Tuple[float, float]], test_ranges: List[Tuple[float, float]]
):
    """
    Visualize time ranges of train and test sets
    """
    plt.figure(figsize=(12, 8))

    for i, (start, end) in enumerate(train_ranges):
        plt.plot(
            [start, end],
            [i, i],
            color='blue',
            marker='|',
            linestyle='-',
            linewidth=2,
            label='Train' if i == 0 else '_nolegend_',
        )

    # Offset test ranges slightly for better visibility if overlapping
    for i, (start, end) in enumerate(test_ranges):
        plt.plot(
            [start, end],
            [i + len(train_ranges), i + len(train_ranges)],
            color='red',
            marker='|',
            linestyle='-',
            linewidth=2,
            label='Test' if i == 0 else '_nolegend_',
        )

    plt.xlabel('Timestamp')
    plt.ylabel('Market Index')
    plt.title('Time Range Overlap Analysis of Train and Test Sets')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    print('Loading training data...')
    train_data = load_jsonl(TRAIN_PATH)
    print(f'Loaded {len(train_data)} training records.')

    print('Loading test data...')
    test_data = load_jsonl(TEST_PATH)
    print(f'Loaded {len(test_data)} test records.')

    train_time_ranges = get_market_start_end_times(train_data)
    test_time_ranges = get_market_start_end_times(test_data)

    print('Checking for time overlaps between train and test sets...')
    overlap_count, overlapping_markets = check_overlap(
        train_time_ranges, test_time_ranges
    )

    if overlap_count > 0:
        print(f'Found {overlap_count} overlapping time ranges.')
        print('Example overlapping periods (start_ts, end_ts):')
        for i, (start, end) in enumerate(
            overlapping_markets[:5]
        ):  # Display up to 5 examples
            print(f'  - {start} - {end}')
    else:
        print('No time overlaps found between train and test sets. Good separation!')


if __name__ == '__main__':
    main()
