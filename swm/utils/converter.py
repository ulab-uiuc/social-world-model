import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..data import PolyMarketData


class Category(str, Enum):
    POLITICS = 'Politics'
    SPORTS = 'Sports'
    CRYPTO = 'Crypto'
    ELECTION = 'Election'
    OTHER = 'Other'


@dataclass
class TimeSeriesConfig:
    prob_threshold: float = 0.5
    time_threshold: float = 0.05
    min_time_diff: float = 3600  # 1 hour in seconds


class PolyMarketDataConverter:
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.datetime_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
        ]

    def find_breakpoints(
        self,
        time_series_data: List[Dict[str, float]],
        start_ts: float,
        end_ts: float,
    ) -> List[Tuple[float, float, float]]:
        if self.config.prob_threshold > 1.0:
            return []

        max_time_diff = max(
            (end_ts - start_ts) * self.config.time_threshold, self.config.min_time_diff
        )

        price_points = {(point['t'], point['p']) for point in time_series_data}
        price_points = sorted(price_points)

        valid_pairs = []
        for i, (t1, p1) in enumerate(price_points):
            for t2, p2 in price_points[i + 1 :]:
                time_diff = t2 - t1
                if time_diff > max_time_diff or time_diff == 0:
                    break

                if abs(p2 - p1) >= self.config.prob_threshold:
                    valid_pairs.append((t1, t2, abs(p2 - p1)))

        return valid_pairs

    def find_categories(self, tags: List[str]) -> List[Category]:
        matches = set()
        for tag in tags:
            tag_lower = tag.lower()
            for category in Category:
                if category != Category.OTHER and category.value.lower() in tag_lower:
                    matches.add(category)

        return list(matches) if matches else [Category.OTHER]

    def parse_winning_outcome(
        self, outcomes: List[str], outcome_prices: List[str]
    ) -> Optional[str]:
        if '0' in outcome_prices and '1' in outcome_prices:
            max_price_index = outcome_prices.index(max(outcome_prices))
            return outcomes[max_price_index]
        return None

    def parse_time_series(
        self, market: Dict[str, Any], outcomes: List[str]
    ) -> Dict[str, Dict[int, float]]:
        clob_token_ids = json.loads(market['clobTokenIds'])
        return {
            outcomes[idx]: market['history'][str(token_id)]
            for idx, token_id in enumerate(clob_token_ids)
        }

    def parse_timestamp(self, datetime_str: str) -> float:
        for fmt in self.datetime_formats:
            try:
                return datetime.strptime(datetime_str, fmt).timestamp()
            except ValueError:
                continue
        raise ValueError(f'Unable to parse timestamp: {datetime_str}')

    def process_market(
        self,
        market: Dict[str, Any],
        event: Dict[str, Any],
    ) -> Optional[PolyMarketData]:
        try:
            tags = [tag['label'] for tag in event['tags']]
            tag_ids = [tag['id'] for tag in event['tags']]
            category = self.find_categories(tags)

            outcome_options = json.loads(market['outcomes'])
            outcome_prices = json.loads(market['outcomePrices'])

            outcome = self.parse_winning_outcome(outcome_options, outcome_prices)
            start_date = (
                market['startDate'] if 'startDate' in market else event['startDate']
            )
            end_date = market['endDate'] if 'endDate' in market else event['endDate']
            start_ts = self.parse_timestamp(start_date)
            end_ts = self.parse_timestamp(end_date)
            time_series = self.parse_time_series(market, outcome_options)
            volumn = market.get('volume', None)
            resolution_source = market.get('resolutionSource', None)
            description = market.get('description', None)

            breakpoint_ts_pairs = {
                outcome: self.find_breakpoints(data, start_ts, end_ts)
                for outcome, data in time_series.items()
            }

            return PolyMarketData(
                event_id=event['id'],
                market_id=market['id'],
                question=market['question'],
                description=description,
                resolution_source=resolution_source,
                volume=volumn,
                outcome=outcome,
                time_series=time_series,
                tags=tags,
                tag_ids=tag_ids,
                category=category,
                start_ts=start_ts,
                end_ts=end_ts,
                breakpoint_ts_pairs=breakpoint_ts_pairs,
            )
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            print(f"Error processing market {market.get('id', 'unknown')}: {str(e)}")
            return None

    def convert(self, event: Dict[str, Any]) -> List[PolyMarketData]:
        try:
            markets_data = []
            for market in event['markets']:
                market_data = self.process_market(market, event)
                if market_data:
                    markets_data.append(market_data)
            return markets_data
        except KeyError as e:
            print(f"Error processing event {event.get('id', 'unknown')}: {str(e)}")
            return []
