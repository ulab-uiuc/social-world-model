import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..data import Action, Consensus


def find_breakpoint_ts_pairs(
    time_series_data: List[Dict[str, float]],
    start_ts: float,
    end_ts: float,
    prob_threshold: float = 0.5,
    time_threshold: float = 0.05,
) -> List[Tuple[float, float]]:
    if prob_threshold > 1.0:
        return []

    max_time_diff = max((end_ts - start_ts) * time_threshold, 3600)

    # Convert time series to sorted list of unique (timestamp, price) pairs
    price_points = set()
    for point in time_series_data:
        price_points.add((point['t'], point['p']))

    price_points = sorted(price_points)

    # Find all pairs meeting both conditions
    valid_pairs = []
    for i in range(len(price_points)):
        for j in range(i + 1, len(price_points)):
            t1, p1 = price_points[i]
            t2, p2 = price_points[j]

            time_diff = t2 - t1
            if time_diff > max_time_diff or time_diff == 0:
                break

            if abs(p2 - p1) >= prob_threshold:
                valid_pairs.append((t1, t2))
    return valid_pairs


def extract_winning_outcome(
    outcomes: List[str], outcome_prices: Dict[str, float]
) -> Optional[str]:
    """Extract the winning outcome based on highest price."""
    if '0' in outcome_prices and '1' in outcome_prices:
        max_price_index = outcome_prices.index(max(outcome_prices))
        return outcomes[max_price_index]
    return None


def extract_time_series(
    market: Dict[str, Any], outcomes: List[str]
) -> Optional[Dict[str, Dict[int, float]]]:
    """
    Extract time series data from market history.
    Returns None if no valid history data is available.
    """
    clob_token_ids = json.loads(market.get('clobTokenIds', None))
    if not clob_token_ids or not market.get('history'):
        return None

    time_series = {}
    for idx, clob_token_id in enumerate(clob_token_ids):
        outcome = outcomes[idx]
        raw_data = market['history'][str(clob_token_id)]
        time_series[outcome] = raw_data
    return time_series


def parse_timestamp(datetime_str: str) -> Optional[float]:
    """
    Convert ISO 8601 datetime string to UNIX timestamp.
    Returns None if parsing fails.
    """
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',  # With milliseconds
        '%Y-%m-%dT%H:%M:%SZ',  # Without milliseconds
    ]

    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt).timestamp()
        except ValueError:
            continue
    return None


def validate_market_data(market: Dict[str, Any]) -> bool:
    """Validate required fields in market data."""
    required_fields = [
        'outcomes',
        'outcomePrices',
        'startDate',
        'endDate',
        'volume',
        'resolutionSource',
        'clobTokenIds',
    ]
    return all(field in market for field in required_fields)


def convert_polymarket_event_into_consensuses(event: Dict[str, Any]) -> List[Consensus]:
    """
    Convert Polymarket event data into a list of Consensus objects.
    Returns empty list if required data is missing or processing fails.
    """
    try:
        if not {'markets', 'tags'}.issubset(event.keys()):
            return []

        tags = [tag['label'] for tag in event['tags']]
        tag_ids = [tag['id'] for tag in event['tags']]
        consensuses = []

        for market in event['markets']:
            if not validate_market_data(market):
                continue

            outcomes = json.loads(market['outcomes'])
            outcome_prices = json.loads(market['outcomePrices'])
            start_ts = parse_timestamp(market['startDate'])
            end_ts = parse_timestamp(market['endDate'])
            time_series = extract_time_series(market, outcomes)

            # Find significant probability changes
            breakpoint_ts_pairs = {}
            if time_series and start_ts and end_ts:
                for outcome, time_series_data in time_series.items():
                    breakpoint_ts_pairs_data = find_breakpoint_ts_pairs(
                        time_series_data, start_ts, end_ts
                    )
                    breakpoint_ts_pairs[outcome] = breakpoint_ts_pairs_data

            consensus = Consensus(
                event_id=event['id'],
                market_id=market['id'],
                question=market['question'],
                discrption=market['description'],
                volumn=market['volume'],
                resolution_source=market['resolutionSource'],
                outcome=extract_winning_outcome(outcomes, outcome_prices),
                time_series=time_series,
                tags=tags,
                tag_ids=tag_ids,
                start_ts=start_ts,
                end_ts=end_ts,
                breakpoint_ts_pairs=breakpoint_ts_pairs,  # Add this to your Consensus model
            )
            consensuses.append(consensus)

        return consensuses

    except Exception as e:
        print(f"Error processing event {event.get('id', 'unknown')}: {str(e)}")
        return []


def find_action_in_states(consensuses: List[Consensus]) -> Optional[List[Action]]:
    """
    Find an Action object in a list of Consensus objects.
    Returns None if no matching action is found.
    """
    actions = []
    for consensus in tqdm(consensuses):
        time_series_data = consensus.time_series
        if time_series_data is None:
            continue
        for time_series in time_series_data.values():
            last_time_number = 5
            last_time_series = time_series[-last_time_number:]
            min_p = min([point['p'] for point in last_time_series])
            max_p = max([point['p'] for point in last_time_series])
            if (
                max_p - min_p > 0.3
                and (min_p < 0.05 or max_p > 0.95)
                and len(time_series) > 100
                and 'Sports' not in consensus.tags
            ):
                action = Action(
                    event_id=consensus.event_id,
                    market_id=consensus.market_id,
                    question=consensus.question,
                    discrption=consensus.discrption,
                    volumn=consensus.volumn,
                    resolution_source=consensus.resolution_source,
                    outcome=consensus.outcome,
                    tags=consensus.tags,
                    tag_ids=consensus.tag_ids,
                    end_ts=consensus.end_ts,
                )
                actions.append(action)
                print(action.question)
                break
    return actions


def find_state_action_pairs(
    actions: List[Action],
    consensuses: List[Consensus],
) -> List[Tuple[List[Consensus], Action]]:
    """
    Find a pair of Consensus objects representing a state-action pair.
    Returns a tuple of (state, action) Consensus objects.
    """
    state_action_pairs = []
    for action in actions:
        end_ts = action.end_ts
        if 'Sports' in action.tags:
            continue
        for consensus in consensuses:
            if 'Sports' in consensus.tags:
                continue
            breakpoint_ts_pairs = consensus.breakpoint_ts_pairs
            for outcome, breakpoint_ts_pair in breakpoint_ts_pairs.items():
                for ts_pair in breakpoint_ts_pair:
                    a, b = ts_pair[0], ts_pair[1]
                    if a < end_ts and b > end_ts and abs(a - b) <= 10800:
                        state_action_pairs.append((consensus, action))
                        print(
                            'Action: {}, State: {}, Outcome: {}'.format(
                                action.question, consensus.question, outcome
                            )
                        )
                        break
    return state_action_pairs
