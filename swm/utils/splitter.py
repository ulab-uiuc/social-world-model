import random
from typing import Dict, List, Tuple

from ..data import PolyMarketData


def split_polymarket_data(
    data_list: List[PolyMarketData],
) -> Tuple[List[PolyMarketData], List[PolyMarketData], List[PolyMarketData]]:
    event_groups: Dict[str, List[PolyMarketData]] = {}
    for record in data_list:
        if record.outcome is not None:  # Only keep records with outcomes
            if record.event_id not in event_groups:
                event_groups[record.event_id] = []
            event_groups[record.event_id].append(record)

    event_list = [(event_id, records) for event_id, records in event_groups.items()]
    random.shuffle(event_list)

    total_size = len(data_list)
    target_test_size = int(0.1 * total_size)

    test_data = []
    train_data = []
    dev_data = []

    current_test_size = 0

    for event_id, records in event_list:
        if current_test_size < target_test_size:
            test_data.extend(records)
            current_test_size += len(records)
        else:
            break

    remaining_events = event_list[len(test_data) :]

    train_split = int(0.89 * len(remaining_events))

    for event_id, records in remaining_events[:train_split]:
        train_data.extend(records)

    for event_id, records in remaining_events[train_split:]:
        dev_data.extend(records)

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
