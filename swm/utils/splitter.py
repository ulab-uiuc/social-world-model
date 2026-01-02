from typing import Dict, List, Tuple, Union

from ..data import KalshiData, PolyMarketData

MarketData = Union[PolyMarketData, KalshiData]


def split_by_time(
    data_list: List[MarketData],
    train_cutoff: float,
    dev_cutoff: float,
) -> Tuple[List[MarketData], List[MarketData], List[MarketData]]:
    """Split data by time based on end_ts.
    
    Args:
        data_list: List of market data records
        train_cutoff: Unix timestamp - data with end_ts <= train_cutoff goes to train
        dev_cutoff: Unix timestamp - data with train_cutoff < end_ts <= dev_cutoff goes to dev
                    data with end_ts > dev_cutoff goes to test
    
    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    train_data = []
    dev_data = []
    test_data = []
    
    for record in data_list:
        # Use end_ts for splitting, fall back to start_ts if end_ts is None
        ts = record.end_ts or record.start_ts
        
        if ts is None:
            # No timestamp, put in train by default
            train_data.append(record)
        elif ts <= train_cutoff:
            train_data.append(record)
        elif ts <= dev_cutoff:
            dev_data.append(record)
        else:
            test_data.append(record)
    
    return train_data, dev_data, test_data


def get_split_stats(
    train: List[MarketData], dev: List[MarketData], test: List[MarketData]
) -> Dict:
    total = len(train) + len(dev) + len(test)
    return {
        'total_records': total,
        'train_count': len(train),
        'dev_count': len(dev),
        'test_count': len(test),
        'train_ratio': round(len(train) / total, 4) if total > 0 else 0,
        'dev_ratio': round(len(dev) / total, 4) if total > 0 else 0,
        'test_ratio': round(len(test) / total, 4) if total > 0 else 0,
        'unique_events_train': len(set(x.event_id for x in train)),
        'unique_events_dev': len(set(x.event_id for x in dev)),
        'unique_events_test': len(set(x.event_id for x in test)),
    }
