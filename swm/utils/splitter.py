import random
from collections import defaultdict
from typing import Dict, List, Tuple

from ..data import PolyMarketData


def split_polymarket_data(
    data_list: List[PolyMarketData],
) -> Tuple[List[PolyMarketData], List[PolyMarketData], List[PolyMarketData]]:
    # Group by event_id
    event_groups = defaultdict(list)
    for record in data_list:
        event_groups[record.event_id].append(record)

    train_data = []
    dev_data = []
    test_data = []

    for event_id, records in event_groups.items():
        no_outcome = [r for r in records if r.outcome is None]
        with_outcome = [r for r in records if r.outcome is not None]

        test_data.extend(no_outcome)

        if with_outcome:
            random.shuffle(with_outcome)
            total = len(with_outcome)
            train_size = int(0.8 * total)
            dev_size = int(0.1 * total)

            train_data.extend(with_outcome[:train_size])
            dev_data.extend(with_outcome[train_size : train_size + dev_size])
            test_data.extend(with_outcome[train_size + dev_size :])

    return train_data, dev_data, test_data


def get_split_stats(
    train: List[PolyMarketData], dev: List[PolyMarketData], test: List[PolyMarketData]
) -> Dict:
    total = len(train) + len(dev) + len(test)
    return {
        'total_records': total,
        'train_count': len(train),
        'dev_count': len(dev),
        'test_count': len(test),
        'train_ratio': len(train) / total if total > 0 else 0,
        'dev_ratio': len(dev) / total if total > 0 else 0,
        'test_ratio': len(test) / total if total > 0 else 0,
        'unique_events_train': len(set(x.event_id for x in train)),
        'unique_events_dev': len(set(x.event_id for x in dev)),
        'unique_events_test': len(set(x.event_id for x in test)),
        'no_outcome_count_test': len([x for x in test if x.outcome is None]),
    }


# Example usage:
# data_list = [your_polymarket_data_instances]
# train_data, dev_data, test_data = split_polymarket_data(data_list)
# stats = get_split_stats(train_data, dev_data, test_data)
# print(stats)
