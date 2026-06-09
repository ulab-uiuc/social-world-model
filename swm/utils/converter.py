import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from pydantic import ValidationError

from ..data import DailyNewsData, MarketData
from .utils import filter_midnight_points


class PolymarketCategory(str, Enum):
    """Categories for Polymarket data."""

    POLITICS = 'Politics'
    CRYPTO = 'Crypto'
    ELECTION = 'Election'


class KalshiCategory(str, Enum):
    """Categories for Kalshi data (excluding Sports and Climate)."""

    COMPANIES = 'Companies'
    CRYPTO = 'Crypto'
    ECONOMICS = 'Economics'
    EDUCATION = 'Education'
    ELECTIONS = 'Elections'
    ENTERTAINMENT = 'Entertainment'
    FINANCIALS = 'Financials'
    HEALTH = 'Health'
    MENTIONS = 'Mentions'
    OTHER = 'Other'
    POLITICS = 'Politics'
    SCIENCE_AND_TECH = 'Science and Technology'
    SOCIAL = 'Social'
    TRANSPORTATION = 'Transportation'
    WORLD = 'World'


@dataclass
class TimeSeriesConfig:
    z_score_threshold: float = 2.0  # Z-score threshold for anomaly detection
    rolling_window: int = 30  # Window size for rolling robust z-score calculation


class PolyMarketDataConverter:
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.datetime_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
        ]

    def _compute_robust_z_score(
        self,
        changes: List[float],
        current_change: float,
    ) -> float:
        """Compute robust z-score using median and MAD (Median Absolute Deviation).

        Args:
            changes: List of historical changes (local window)
            current_change: The change to compute z-score for

        Returns:
            Robust z-score value
        """
        import statistics

        if len(changes) < 10:
            return 0.0

        median = statistics.median(changes)
        # MAD = median(|x_i - median|)
        mad = statistics.median([abs(c - median) for c in changes])
        # Scale MAD to approximate std (for normal distribution, MAD ≈ 0.6745 * std)
        scaled_mad = mad * 1.4826
        scaled_mad = max(scaled_mad, 0.01)  # Avoid division by zero

        return (current_change - median) / scaled_mad

    def find_breakpoints(
        self,
        daily_series: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in daily time series using rolling robust Z-score.

        Algorithm:
        1. For each point, use a rolling window of previous changes
        2. Compute robust Z-score using median and MAD (Median Absolute Deviation)
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
        window_size = self.config.rolling_window

        # Sort by timestamp
        sorted_series = sorted(daily_series, key=lambda x: x['t'])

        # Calculate all consecutive changes
        changes = []
        for i in range(len(sorted_series) - 1):
            p1 = sorted_series[i]['p']
            p2 = sorted_series[i + 1]['p']
            changes.append(abs(p2 - p1))

        # Need at least 10 historical changes for meaningful statistics
        min_history = 10

        # Detect anomalies using rolling robust z-score
        for i in range(len(changes)):
            if i < min_history:
                # Not enough history yet
                continue

            # Use local window: previous changes only (causal)
            start_idx = max(0, i - window_size)
            local_changes = changes[start_idx:i]

            current_change = changes[i]
            z_score = self._compute_robust_z_score(local_changes, current_change)

            if z_score > z_threshold:
                t1, p1 = sorted_series[i]['t'], sorted_series[i]['p']
                t2, p2 = sorted_series[i + 1]['t'], sorted_series[i + 1]['p']

                # Extract window history: points from start_idx to after point (inclusive)
                window_end_idx = i + 2  # Include both before and after points
                window_history = [
                    {
                        't': float(sorted_series[j]['t']),
                        'p': round(sorted_series[j]['p'], 4),
                    }
                    for j in range(start_idx, min(window_end_idx, len(sorted_series)))
                ]

                breakpoints.append(
                    {
                        'before': {'t': float(t1), 'p': round(p1, 4)},
                        'after': {'t': float(t2), 'p': round(p2, 4)},
                        'change': round(current_change, 4),
                        'z_score': round(z_score, 2),
                        'window_start': float(sorted_series[start_idx]['t']),
                        'window_end': float(t2),
                        'window_history': window_history,
                    }
                )

        return breakpoints

    def find_categories(self, tags: List[str]) -> List[PolymarketCategory]:
        """Find matching categories from tags.

        Only returns categories that match: Crypto, Election, Politics.
        Returns empty list if no match found.
        """
        matches = set()
        for tag in tags:
            tag_lower = tag.lower()
            for category in PolymarketCategory:
                if category.value.lower() in tag_lower:
                    matches.add(category)

        return list(matches)

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
        clob_token_ids_str = market.get('clobTokenIds')
        if not clob_token_ids_str:
            return []
        clob_token_ids = json.loads(clob_token_ids_str)
        if clob_token_ids:
            first_token_id = clob_token_ids[0]
            history = market.get('history', {})
            return history.get(str(first_token_id), [])
        return []

    def parse_daily_time_series(
        self, market: Dict[str, Any], outcomes: List[str]
    ) -> List[Dict[str, float]]:
        """Parse daily time series, only keeping the first outcome (Yes)."""
        clob_token_ids_str = market.get('clobTokenIds')
        if not clob_token_ids_str:
            return []
        clob_token_ids = json.loads(clob_token_ids_str)
        if clob_token_ids:
            first_token_id = clob_token_ids[0]
            history = market.get('history', {})
            hourly_data = history.get(str(first_token_id), [])
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
    ) -> Optional[MarketData]:
        try:
            event_tags = event.get('tags') or []
            tags = [tag['label'] for tag in event_tags]
            tag_ids = [tag['id'] for tag in event_tags]
            categories = self.find_categories(tags)

            outcome_options = json.loads(market['outcomes'])
            # outcomePrices may be missing for unsettled markets
            outcome_prices_str = market.get('outcomePrices')
            if outcome_prices_str:
                outcome_prices = json.loads(outcome_prices_str)
                outcome = self.parse_winning_outcome(outcome_options, outcome_prices)
            else:
                outcome = None
            start_date = market.get('startDate') or event.get('startDate')
            end_date = market.get('endDate') or event.get('endDate')

            # Skip markets without dates
            if not start_date or not end_date:
                return None

            start_ts = self.parse_timestamp(start_date)
            end_ts = self.parse_timestamp(end_date)
            time_series = self.parse_time_series(market, outcome_options)
            daily_time_series = self.parse_daily_time_series(market, outcome_options)
            volume = market.get('volume', None)
            resolution_source = market.get('resolutionSource', None)
            description = market.get('description', None)

            # Find daily breakpoints using daily time series for anomaly detection
            daily_breakpoints = self.find_breakpoints(daily_time_series)

            return MarketData(
                event_id=event['id'],
                market_id=market['id'],
                question=market['question'],
                description=description,
                resolution_source=resolution_source,
                volume=volume,
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

    def convert(self, event: Dict[str, Any]) -> List[MarketData]:
        markets = event.get('markets') or []
        if not markets:
            return []

        markets_data = []
        for market in markets:
            market_data = self.process_market(market, event)
            if market_data:
                markets_data.append(market_data)
        return markets_data


class KalshiDataConverter:
    """Converter for Kalshi prediction market data to standardized format."""

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()

        url = 'https://api.elections.kalshi.com/trade-api/v2/series'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        self.series_ticker_dict = {
            series['ticker']: series['category'] for series in data['series']
        }

    def _compute_robust_z_score(
        self,
        changes: List[float],
        current_change: float,
    ) -> float:
        """Compute robust z-score using median and MAD (Median Absolute Deviation).

        Args:
            changes: List of historical changes (local window)
            current_change: The change to compute z-score for

        Returns:
            Robust z-score value
        """
        import statistics

        if len(changes) < 10:
            return 0.0

        median = statistics.median(changes)
        # MAD = median(|x_i - median|)
        mad = statistics.median([abs(c - median) for c in changes])
        # Scale MAD to approximate std (for normal distribution, MAD ≈ 0.6745 * std)
        scaled_mad = mad * 1.4826
        scaled_mad = max(scaled_mad, 0.01)  # Avoid division by zero

        return (current_change - median) / scaled_mad

    def find_breakpoints(
        self,
        daily_series: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in daily time series using rolling robust Z-score.

        Algorithm:
        1. For each point, use a rolling window of previous changes
        2. Compute robust Z-score using median and MAD (Median Absolute Deviation)
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
        window_size = self.config.rolling_window

        # Sort by timestamp
        sorted_series = sorted(daily_series, key=lambda x: x['t'])

        # Calculate all consecutive changes
        changes = []
        for i in range(len(sorted_series) - 1):
            p1 = sorted_series[i]['p']
            p2 = sorted_series[i + 1]['p']
            changes.append(abs(p2 - p1))

        # Need at least 10 historical changes for meaningful statistics
        min_history = 10

        # Detect anomalies using rolling robust z-score
        for i in range(len(changes)):
            if i < min_history:
                # Not enough history yet
                continue

            # Use local window: previous changes only (causal)
            start_idx = max(0, i - window_size)
            local_changes = changes[start_idx:i]

            current_change = changes[i]
            z_score = self._compute_robust_z_score(local_changes, current_change)

            if z_score > z_threshold:
                t1, p1 = sorted_series[i]['t'], sorted_series[i]['p']
                t2, p2 = sorted_series[i + 1]['t'], sorted_series[i + 1]['p']

                # Extract window history: points from start_idx to after point (inclusive)
                window_end_idx = i + 2  # Include both before and after points
                window_history = [
                    {
                        't': float(sorted_series[j]['t']),
                        'p': round(sorted_series[j]['p'], 4),
                    }
                    for j in range(start_idx, min(window_end_idx, len(sorted_series)))
                ]

                breakpoints.append(
                    {
                        'before': {'t': float(t1), 'p': round(p1, 4)},
                        'after': {'t': float(t2), 'p': round(p2, 4)},
                        'change': round(current_change, 4),
                        'z_score': round(z_score, 2),
                        'window_start': float(sorted_series[start_idx]['t']),
                        'window_end': float(t2),
                        'window_history': window_history,
                    }
                )

        return breakpoints

    def find_categories(self, event_id: str, question: str) -> List[KalshiCategory]:
        """Get category from Kalshi series ticker.

        Only returns categories that match the allowed list.
        Returns empty list if no match found.
        """
        series_ticker = event_id.split('-')[0]
        category_str = self.series_ticker_dict.get(series_ticker, '')

        # Map API category strings to KalshiCategory enum
        # Exclude Sports and Climate and Weather
        category_mapping = {
            'companies': KalshiCategory.COMPANIES,
            'crypto': KalshiCategory.CRYPTO,
            'economics': KalshiCategory.ECONOMICS,
            'education': KalshiCategory.EDUCATION,
            'elections': KalshiCategory.ELECTIONS,
            'entertainment': KalshiCategory.ENTERTAINMENT,
            'financials': KalshiCategory.FINANCIALS,
            'health': KalshiCategory.HEALTH,
            'mentions': KalshiCategory.MENTIONS,
            'other': KalshiCategory.OTHER,
            'politics': KalshiCategory.POLITICS,
            'science and technology': KalshiCategory.SCIENCE_AND_TECH,
            'social': KalshiCategory.SOCIAL,
            'transportation': KalshiCategory.TRANSPORTATION,
            'world': KalshiCategory.WORLD,
        }

        category_lower = category_str.lower()
        if category_lower in category_mapping:
            return [category_mapping[category_lower]]
        return []

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

    def process_market(self, market_data: Dict[str, Any]) -> Optional[MarketData]:
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

            return MarketData(
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
            print(
                f"Error processing Kalshi market {market_data.get('market_id', 'unknown')}: {e}"
            )
            return None

    def convert(self, market_data: Dict[str, Any]) -> List[MarketData]:
        """Convert a single Kalshi market entry to list of MarketData."""
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
