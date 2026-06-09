from typing import Any, Dict, List, Optional, Protocol, Tuple


class MarketDataProtocol(Protocol):
    """Protocol for market data objects - must have model_dump() method."""

    def model_dump(self) -> Dict[str, Any]: ...


def split_time_series_by_cutoff(
    time_series: List[Dict[str, float]],
    cutoff_ts: float,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Split a time series at cutoff timestamp.

    Args:
        time_series: List of {'t': timestamp, 'p': price}
        cutoff_ts: Unix timestamp to split at

    Returns:
        (before_cutoff, after_cutoff) - two lists of time series points
    """
    before = [point for point in time_series if point['t'] <= cutoff_ts]
    after = [point for point in time_series if point['t'] > cutoff_ts]
    return before, after


def split_breakpoints_by_cutoff(
    breakpoints: List[Dict[str, Any]],
    cutoff_ts: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split breakpoints at cutoff timestamp (based on 'after' timestamp).

    Args:
        breakpoints: List of breakpoint dicts with 'before' and 'after' timestamps
        cutoff_ts: Unix timestamp to split at

    Returns:
        (before_cutoff, after_cutoff) - breakpoints before/after cutoff
    """
    before = []
    after = []
    for bp in breakpoints:
        # Use 'after' timestamp to determine which split the breakpoint belongs to
        bp_ts = bp.get('after', {}).get('t', 0)
        if bp_ts <= cutoff_ts:
            before.append(bp)
        else:
            after.append(bp)
    return before, after


def split_market_by_time(
    market: MarketDataProtocol,
    cutoff_ts: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Split a single market's time series data at cutoff timestamp.

    Splits:
    - time_series (full hourly data)
    - daily_time_series
    - daily_breakpoints

    Args:
        market: Market data record
        cutoff_ts: Unix timestamp to split at

    Returns:
        (train_part, test_part) - two dicts with split data, or None if empty
    """
    # Get base market data as dict
    market_dict = market.model_dump()

    # Split full time series
    ts_before, ts_after = split_time_series_by_cutoff(
        market_dict.get('time_series', []), cutoff_ts
    )

    # Split daily time series
    daily_ts_before, daily_ts_after = split_time_series_by_cutoff(
        market_dict.get('daily_time_series', []), cutoff_ts
    )

    # Split breakpoints
    bp_before, bp_after = split_breakpoints_by_cutoff(
        market_dict.get('daily_breakpoints', []), cutoff_ts
    )

    # Create train part (before cutoff)
    train_part = None
    if daily_ts_before:  # Only include if has data
        train_part = {
            **market_dict,
            'time_series': ts_before,
            'daily_time_series': daily_ts_before,
            'daily_breakpoints': bp_before,
            'split_cutoff_ts': cutoff_ts,
            'split_type': 'train',
        }

    # Create test part (after cutoff)
    test_part = None
    if daily_ts_after:  # Only include if has data
        test_part = {
            **market_dict,
            'time_series': ts_after,
            'daily_time_series': daily_ts_after,
            'daily_breakpoints': bp_after,
            'split_cutoff_ts': cutoff_ts,
            'split_type': 'test',
        }

    return train_part, test_part


def split_dataset_by_time(
    data_list: List[MarketDataProtocol],
    cutoff_ts: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split all markets' time series at cutoff timestamp.

    Each market's time series is split:
    - Points with t <= cutoff_ts go to train
    - Points with t > cutoff_ts go to test

    A market can appear in BOTH train and test if it spans the cutoff.

    Args:
        data_list: List of market data records
        cutoff_ts: Unix timestamp to split at

    Returns:
        (train_data, test_data) - lists of split market dicts
    """
    train_data = []
    test_data = []

    for market in data_list:
        train_part, test_part = split_market_by_time(market, cutoff_ts)

        # Include if has any data points
        if train_part:
            train_data.append(train_part)

        if test_part:
            test_data.append(test_part)

    return train_data, test_data


def get_split_stats(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    cutoff_ts: float,
) -> Dict:
    """Get statistics about the split."""
    from datetime import datetime

    # Count breakpoints
    train_bp_count = sum(len(x.get('daily_breakpoints', [])) for x in train)
    test_bp_count = sum(len(x.get('daily_breakpoints', [])) for x in test)

    # Get market IDs
    train_market_ids = set(x.get('market_id') for x in train)
    test_market_ids = set(x.get('market_id') for x in test)

    # Markets that span the cutoff (appear in both train and test)
    overlap_market_ids = train_market_ids & test_market_ids
    train_only_ids = train_market_ids - test_market_ids
    test_only_ids = test_market_ids - train_market_ids

    return {
        'cutoff_timestamp': cutoff_ts,
        'cutoff_date': datetime.fromtimestamp(cutoff_ts).strftime('%Y-%m-%d'),
        # Market counts
        'unique_markets_total': len(train_market_ids | test_market_ids),
        'markets_in_train': len(train_market_ids),
        'markets_in_test': len(test_market_ids),
        'markets_in_both': len(overlap_market_ids),  # Span cutoff
        'markets_train_only': len(train_only_ids),  # End before cutoff
        'markets_test_only': len(test_only_ids),  # Start after cutoff
        # Breakpoint counts
        'breakpoints_in_train': train_bp_count,
        'breakpoints_in_test': test_bp_count,
        # Data points
        'train_total_daily_points': sum(
            len(x.get('daily_time_series', [])) for x in train
        ),
        'test_total_daily_points': sum(
            len(x.get('daily_time_series', [])) for x in test
        ),
        'train_avg_daily_points': round(
            sum(len(x.get('daily_time_series', [])) for x in train) / len(train), 2
        )
        if train
        else 0,
        'test_avg_daily_points': round(
            sum(len(x.get('daily_time_series', [])) for x in test) / len(test), 2
        )
        if test
        else 0,
    }
