import json
from typing import Dict, List, Tuple

INPUT_PATH = 'splitted_polymarket/swm-bench/swm_bench_all.jsonl'
TRAIN_PATH = 'swm_bench_train_new.jsonl'
TEST_PATH = 'swm_bench_test_and_dev_new.jsonl'
SPLIT_RATIO = 0.8


def load_data(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], path: str):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def extract_key_timestamps(data: List[Dict], window_size: int = 5) -> List[float]:
    """
    Extract key timestamps that meet the price change conditions
    """
    key_timestamps = []

    for market in data:
        series = market.get('daily_time_series', {}).get('Yes', [])
        if len(series) <= window_size:
            continue

        for start_idx in range(len(series) - window_size):
            window = series[start_idx : start_idx + window_size]
            target = series[start_idx + window_size]
            if target['p'] - window[-1]['p'] >= 0.25:
                key_timestamps.append(target['t'])

    return sorted(key_timestamps)


def get_market_key_timestamps(market: Dict, window_size: int = 5) -> List[float]:
    """
    Get a list of key timestamps for a single market
    """
    key_timestamps = []
    series = market.get('daily_time_series', {}).get('Yes', [])

    if len(series) <= window_size:
        return key_timestamps

    for start_idx in range(len(series) - window_size):
        window = series[start_idx : start_idx + window_size]
        target = series[start_idx + window_size]
        if target['p'] - window[-1]['p'] >= 0.25:
            key_timestamps.append(target['t'])

    return key_timestamps


def split_data_by_key_timestamps_strict(
    data: List[Dict], train_ratio: float, window_size: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Strictly split data based on key timestamps, ensuring complete separation of training and test sets at key timestamps
    """
    print('Extracting all key timestamps...')
    all_key_timestamps = extract_key_timestamps(data, window_size)
    print(f'Found {len(all_key_timestamps)} key timestamps')

    if len(all_key_timestamps) == 0:
        print(
            'No key timestamps found, unable to perform key timestamp-based splitting'
        )
        return [], []

    split_timestamp_idx = int(len(all_key_timestamps) * train_ratio)
    if split_timestamp_idx >= len(all_key_timestamps):
        split_timestamp_idx = len(all_key_timestamps) - 1

    split_timestamp = all_key_timestamps[split_timestamp_idx]
    print(f'Selected split timestamp: {split_timestamp}')

    train_data = []
    test_data = []

    for item in data:
        market_key_ts = get_market_key_timestamps(item, window_size)

        if len(market_key_ts) == 0:
            if item['start_ts'] <= split_timestamp:
                train_data.append(item)
            else:
                test_data.append(item)
        else:
            max_key_ts = max(market_key_ts)
            if max_key_ts <= split_timestamp:
                train_data.append(item)
            else:
                test_data.append(item)

    return train_data, test_data


def split_data_by_key_timestamps_perfect(
    data: List[Dict], train_ratio: float, window_size: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perfectly split data based on key timestamps, ensuring complete separation of training and test sets at key timestamps
    """
    print('Extracting all key timestamps...')
    all_key_timestamps = extract_key_timestamps(data, window_size)
    print(f'Found {len(all_key_timestamps)} key timestamps')

    if len(all_key_timestamps) == 0:
        print(
            'No key timestamps found, unable to perform key timestamp-based splitting'
        )
        return [], []

    # Find split point: select the key timestamp at the train_ratio as the split boundary
    split_timestamp_idx = int(len(all_key_timestamps) * train_ratio)
    if split_timestamp_idx >= len(all_key_timestamps):
        split_timestamp_idx = len(all_key_timestamps) - 1

    split_timestamp = all_key_timestamps[split_timestamp_idx]
    print(f'Selected split timestamp: {split_timestamp}')

    # Split data based on split timestamp, ensuring perfect separation
    train_data = []
    test_data = []

    for item in data:
        # Get key timestamps for this market
        market_key_ts = get_market_key_timestamps(item, window_size)

        if len(market_key_ts) == 0:
            # If this market has no key timestamps, assign based on start_ts
            if item['start_ts'] <= split_timestamp:
                train_data.append(item)
            else:
                test_data.append(item)
        else:
            # If this market has key timestamps, check all key timestamps
            # Only assign to training set if all key timestamps are less than or equal to the split timestamp
            all_before_split = all(ts <= split_timestamp for ts in market_key_ts)
            all_after_split = all(ts > split_timestamp for ts in market_key_ts)

            if all_before_split:
                train_data.append(item)
            elif all_after_split:
                test_data.append(item)
            else:
                # If key timestamps span the split point, special handling is needed
                # Check the position of most key timestamps
                before_count = sum(1 for ts in market_key_ts if ts <= split_timestamp)
                after_count = len(market_key_ts) - before_count

                if before_count >= after_count:
                    train_data.append(item)
                else:
                    test_data.append(item)

    return train_data, test_data


def verify_key_timestamp_separation(
    train_data: List[Dict], test_data: List[Dict], window_size: int = 5
) -> bool:
    """
    Verify if training and test sets are completely separated at key timestamps
    """
    train_key_ts = extract_key_timestamps(train_data, window_size)
    test_key_ts = extract_key_timestamps(test_data, window_size)

    if len(train_key_ts) == 0 or len(test_key_ts) == 0:
        print('Training or test set has no key timestamps')
        return True

    max_train_key_ts = max(train_key_ts)
    min_test_key_ts = min(test_key_ts)

    if max_train_key_ts < min_test_key_ts:
        print('Key timestamps are completely separated')
        print(f'   - Max train key timestamp: {max_train_key_ts}')
        print(f'   - Min test key timestamp: {min_test_key_ts}')
        return True
    else:
        print('Key timestamps overlap')
        print(f'   - Max train key timestamp: {max_train_key_ts}')
        print(f'   - Min test key timestamp: {min_test_key_ts}')

        # Display overlapping timestamps
        overlap_ts = [ts for ts in train_key_ts if ts >= min_test_key_ts]
        if overlap_ts:
            print(f'   - Overlapping key timestamps count: {len(overlap_ts)}')
            print(f'   - Example overlapping timestamps: {overlap_ts[:5]}')

        return False


def main():
    data = load_data(INPUT_PATH)
    print(f'Total data records: {len(data)}')

    # Display time range
    start_times = [item['start_ts'] for item in data]
    end_times = [item['end_ts'] for item in data]
    print(f'Time range: {min(start_times)} - {max(end_times)}')

    # Attempt perfect split
    print('Attempting perfect split...')
    train, test = split_data_by_key_timestamps_perfect(data, SPLIT_RATIO)

    if len(train) == 0 or len(test) == 0:
        print('Perfect split failed')
        return

    print('Data split successfully:')
    print(f'   - Train: {len(train)} records ({len(train) / len(data) * 100:.1f}%)')
    print(f'   - Test : {len(test)} records ({len(test) / len(data) * 100:.1f}%)')

    # Verify key timestamp separation
    is_separated = verify_key_timestamp_separation(train, test)

    if not is_separated:
        print('Attempting strict split...')
        train, test = split_data_by_key_timestamps_strict(data, SPLIT_RATIO)
        is_separated = verify_key_timestamp_separation(train, test)

        if not is_separated:
            print('Warning: Unable to achieve complete separation of key timestamps')

    # Display time range after splitting
    if len(train) > 0:
        train_start = min(item['start_ts'] for item in train)
        train_end = max(item['end_ts'] for item in train)
        print(f'   - Train time range: {train_start} - {train_end}')

    if len(test) > 0:
        test_start = min(item['start_ts'] for item in test)
        test_end = max(item['end_ts'] for item in test)
        print(f'   - Test time range: {test_start} - {test_end}')

    save_jsonl(train, TRAIN_PATH)
    save_jsonl(test, TEST_PATH)
    print(f'Saved to: {TRAIN_PATH}, {TEST_PATH}')


if __name__ == '__main__':
    main()
