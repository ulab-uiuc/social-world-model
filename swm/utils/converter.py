import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError
import requests

from ..data import DailyNewsData, KalshiData, PolyMarketData
from .utils import filter_midnight_points


class Category(str, Enum):
    POLITICS = 'Politics'
    SPORTS = 'Sports'
    CRYPTO = 'Crypto'
    ELECTION = 'Election'
    OTHER = 'Other'


@dataclass
class TimeSeriesConfig:
    z_score_threshold: float = 2.0  # Z-score threshold for anomaly detection


class PolyMarketDataConverter:
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.datetime_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
        ]

    def find_breakpoints(
        self,
        daily_series: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in daily time series using Z-score based detection.
        
        Algorithm:
        1. Calculate mean and std of all consecutive price changes
        2. Compute Z-score for each change
        3. Detect changes where Z-score exceeds threshold
        
        Args:
            daily_series: List of {'t': timestamp, 'p': price} sorted by time
            
        Returns:
            List of breakpoint dicts with full point information
        """
        if not daily_series or len(daily_series) < 2:
            return []
        
        breakpoints = []
        z_threshold = self.config.z_score_threshold
        
        # Sort by timestamp
        sorted_series = sorted(daily_series, key=lambda x: x['t'])
        
        # Calculate all consecutive changes
        changes = []
        for i in range(len(sorted_series) - 1):
            p1 = sorted_series[i]['p']
            p2 = sorted_series[i + 1]['p']
            changes.append(abs(p2 - p1))
        
        # Calculate statistics for Z-score
        if len(changes) >= 3:
            import statistics
            mean_change = statistics.mean(changes)
            std_change = statistics.stdev(changes) if len(changes) > 1 else 1.0
            std_change = max(std_change, 0.01)  # Avoid division by zero
        else:
            # Not enough data for meaningful statistics
            return []
        
        # Detect anomalies based on Z-score
        for i in range(len(sorted_series) - 1):
            t1, p1 = sorted_series[i]['t'], sorted_series[i]['p']
            t2, p2 = sorted_series[i + 1]['t'], sorted_series[i + 1]['p']
            
            change = abs(p2 - p1)
            z_score = (change - mean_change) / std_change
            
            if z_score > z_threshold:
                breakpoints.append({
                    'before': {'t': float(t1), 'p': round(p1, 4)},
                    'after': {'t': float(t2), 'p': round(p2, 4)},
                    'change': round(change, 4),
                    'z_score': round(z_score, 2),
                })
        
        return breakpoints

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
    ) -> List[Dict[str, float]]:
        """Parse hourly time series, only keeping the first outcome (Yes)."""
        clob_token_ids = json.loads(market['clobTokenIds'])
        if clob_token_ids:
            first_token_id = clob_token_ids[0]
            return market['history'].get(str(first_token_id), [])
        return []

    def parse_daily_time_series(
        self, market: Dict[str, Any], outcomes: List[str]
    ) -> List[Dict[str, float]]:
        """Parse daily time series, only keeping the first outcome (Yes)."""
        clob_token_ids = json.loads(market['clobTokenIds'])
        if clob_token_ids:
            first_token_id = clob_token_ids[0]
            hourly_data = market['history'].get(str(first_token_id), [])
            return filter_midnight_points(hourly_data)
        return []

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
            categories = self.find_categories(tags)

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
            daily_time_series = self.parse_daily_time_series(market, outcome_options)
            volumn = market.get('volume', None)
            resolution_source = market.get('resolutionSource', None)
            description = market.get('description', None)

            # Find daily breakpoints using daily time series for anomaly detection
            daily_breakpoints = self.find_breakpoints(daily_time_series)

            return PolyMarketData(
                event_id=event['id'],
                market_id=market['id'],
                question=market['question'],
                description=description,
                resolution_source=resolution_source,
                volume=volumn,
                outcome=outcome,
                time_series=time_series,
                daily_time_series=daily_time_series,
                tags=tags,
                tag_ids=tag_ids,
                categories=categories,
                start_ts=start_ts,
                end_ts=end_ts,
                daily_breakpoints=daily_breakpoints,
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


class KalshiDataConverter:
    """Converter for Kalshi prediction market data to standardized format."""

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()

        url = "https://api.elections.kalshi.com/trade-api/v2/series"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        self.series_ticker_dict = {series['ticker']: series['category'] for series in data['series']}

    def find_breakpoints(
        self,
        daily_series: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in daily time series using Z-score based detection.
        
        Algorithm:
        1. Calculate mean and std of all consecutive price changes
        2. Compute Z-score for each change
        3. Detect changes where Z-score exceeds threshold
        
        Args:
            daily_series: List of {'t': timestamp, 'p': price} sorted by time
            
        Returns:
            List of breakpoint dicts with full point information
        """
        if not daily_series or len(daily_series) < 2:
            return []
        
        breakpoints = []
        z_threshold = self.config.z_score_threshold
        
        # Sort by timestamp
        sorted_series = sorted(daily_series, key=lambda x: x['t'])
        
        # Calculate all consecutive changes
        changes = []
        for i in range(len(sorted_series) - 1):
            p1 = sorted_series[i]['p']
            p2 = sorted_series[i + 1]['p']
            changes.append(abs(p2 - p1))
        
        # Calculate statistics for Z-score
        if len(changes) >= 3:
            import statistics
            mean_change = statistics.mean(changes)
            std_change = statistics.stdev(changes) if len(changes) > 1 else 1.0
            std_change = max(std_change, 0.01)  # Avoid division by zero
        else:
            # Not enough data for meaningful statistics
            return []
        
        # Detect anomalies based on Z-score
        for i in range(len(sorted_series) - 1):
            t1, p1 = sorted_series[i]['t'], sorted_series[i]['p']
            t2, p2 = sorted_series[i + 1]['t'], sorted_series[i + 1]['p']
            
            change = abs(p2 - p1)
            z_score = (change - mean_change) / std_change
            
            if z_score > z_threshold:
                breakpoints.append({
                    'before': {'t': float(t1), 'p': round(p1, 4)},
                    'after': {'t': float(t2), 'p': round(p2, 4)},
                    'change': round(change, 4),
                    'z_score': round(z_score, 2),
                })
        
        return breakpoints

    def find_categories(self, event_id: str, question: str) -> List[str]:
        """Get category from Kalshi series ticker."""
        series_ticker = event_id.split('-')[0]
        category_str = self.series_ticker_dict.get(series_ticker, 'Other')
        return [category_str]

    def parse_time_series(
        self, time_series: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Return the time series directly (represents Yes probability)."""
        return time_series

    def parse_daily_time_series(
        self, time_series: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Filter to daily (midnight) points."""
        return filter_midnight_points(time_series)

    def process_market(self, market_data: Dict[str, Any]) -> Optional[KalshiData]:
        """Process a single Kalshi market entry."""
        try:
            event_id = market_data.get('event_id', '')
            market_id = market_data.get('market_id', '')
            question = market_data.get('question', '')
            time_series = market_data.get('time_series', [])

            if not time_series:
                return None

            categories = self.find_categories(event_id, question)
            start_ts = market_data.get('start_ts', 0)
            end_ts = market_data.get('end_ts', 0)

            time_series = self.parse_time_series(time_series)
            daily_time_series = self.parse_daily_time_series(time_series)

            # Find daily breakpoints using daily time series for anomaly detection
            daily_breakpoints = self.find_breakpoints(daily_time_series)

            # Map outcome to Yes/No format
            outcome = market_data.get('outcome')
            if outcome == 'yes':
                outcome = 'Yes'
            elif outcome == 'no':
                outcome = 'No'

            return KalshiData(
                event_id=event_id,
                market_id=market_id,
                question=question,
                description=market_data.get('yes_sub_title'),
                outcome=outcome,
                time_series=time_series,
                daily_time_series=daily_time_series,
                categories=[c for c in categories],
                start_ts=start_ts,
                end_ts=end_ts,
                daily_breakpoints=daily_breakpoints,
                # Kalshi-specific fields
                event_ticker=event_id,
                market_ticker=market_id,
                title=question,
                subtitle=market_data.get('yes_sub_title'),
            )
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing Kalshi market {market_data.get('market_id', 'unknown')}: {e}")
            return None

    def convert(self, market_data: Dict[str, Any]) -> List[KalshiData]:
        """Convert a single Kalshi market entry to list of KalshiData."""
        result = self.process_market(market_data)
        return [result] if result else []


class DailyNewsConverter:
    def __init__(self):
        self.datetime_format = '%Y-%m-%d'

    def convert(self, data: Dict) -> Optional[DailyNewsData]:
        try:
            date = self.parse_date(data.get('published_at'))
            daily_news = DailyNewsData(
                uuid=data.get('uuid'),
                title=data.get('title'),
                url=data.get('url'),
                snippet=data.get('snippet'),
                description=data.get('description'),
                date=date,
            )
            return daily_news
        except ValidationError as ve:
            print(f"Validation error for article UUID {data.get('uuid')}: {ve}")
            return None
        except Exception as e:
            print(
                f"Unexpected error during conversion for article UUID {data.get('uuid')}: {e}"
            )
            return None

    def parse_date(self, date_str: str) -> Optional[str]:
        return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ').strftime(
            self.datetime_format
        )
