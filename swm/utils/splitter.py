import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

from ..data import PolyMarketData


def split_polymarket_data(
    data_list: List[PolyMarketData],
) -> Tuple[List[PolyMarketData], List[PolyMarketData], List[PolyMarketData]]:
    event_groups: DefaultDict[str, list] = defaultdict(list)
    for record in data_list:
        event_groups[record.event_id].append(record)

    all_with_outcomes = []
    for records in event_groups.values():
        with_outcome = [r for r in records if r.outcome is not None]
        if with_outcome:
            all_with_outcomes.extend(with_outcome)

    total_size = len(data_list)
    test_size = int(0.1 * total_size)

    if len(all_with_outcomes) < test_size:
        test_size = len(all_with_outcomes)

    random.shuffle(all_with_outcomes)

    test_data = all_with_outcomes[:test_size]

    remaining = all_with_outcomes[test_size:]
    train_size = int(0.89 * len(remaining))

    train_data = remaining[:train_size]
    dev_data = remaining[train_size:]

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
