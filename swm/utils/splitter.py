import random
from typing import Dict, List, Tuple

from ..data import PolyMarketData


def split_polymarket_data(
    data_list: List[PolyMarketData],
) -> Tuple[List[PolyMarketData], List[PolyMarketData], List[PolyMarketData]]:
    event_groups: Dict[str, List[PolyMarketData]] = {}
    for record in data_list:
        if record.event_id not in event_groups:
            event_groups[record.event_id] = []
        event_groups[record.event_id].append(record)

    event_list = list(event_groups.items())
    random.shuffle(event_list)

    events_with_outcomes = []
    events_without_outcomes = []
    for event_id, records in event_list:
        if any(r.outcome is not None for r in records):
            events_with_outcomes.append((event_id, records))
        else:
            events_without_outcomes.append((event_id, records))

    test_split = min(len(events_with_outcomes), int(0.1 * len(event_list)))
    test_data = [r for _, records in events_with_outcomes[:test_split] for r in records]

    remaining_events = events_with_outcomes[test_split:] + events_without_outcomes
    random.shuffle(remaining_events)

    train_split = int(0.89 * len(remaining_events))
    train_data = [r for _, records in remaining_events[:train_split] for r in records]
    dev_data = [r for _, records in remaining_events[train_split:] for r in records]

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
