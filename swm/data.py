from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class AttributedEvent(BaseModel):
    """A single attributed event (news + score)."""
    news: Dict[str, Any]  # News data (title, description, etc.)
    score: float  # Attribution score (0-1)


class AttributionResult(BaseModel):
    """Attribution results for a specific market at a specific timestamp."""
    market_id: str
    timestamp: float
    events: List[AttributedEvent] = Field(default_factory=list)


class PolyMarketData(BaseModel):
    event_id: str
    market_id: str
    question: str
    resolution_source: Optional[str] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    outcome: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    start_ts: Optional[float] = Field(default=None)
    end_ts: Optional[float] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    tag_ids: Optional[List[str]] = Field(default=None)
    categories: Optional[List[str]] = Field(default=None)
    # Time series as simple list (represents Yes probability)
    daily_time_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    time_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    daily_breakpoints: Optional[List[Dict[str, Any]]] = Field(default=None)
    window_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    # Precomputed attributions: {timestamp -> list of {news, score}}
    attributions: Optional[Dict[str, List[Dict[str, Any]]]] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolyMarketData':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'


class KalshiData(BaseModel):
    """
    Data model for Kalshi prediction market data.
    Compatible with PolyMarket data structure.
    """
    # Core fields
    event_id: str
    market_id: str
    question: str
    resolution_source: Optional[str] = Field(default=None)
    volumn: Optional[float] = Field(default=None)
    outcome: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    start_ts: Optional[float] = Field(default=None)
    end_ts: Optional[float] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    tag_ids: Optional[List[str]] = Field(default=None)
    categories: Optional[List[str]] = Field(default=None)
    
    # Time series as simple list (represents Yes probability)
    daily_time_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    time_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    daily_breakpoints: Optional[List[Dict[str, Any]]] = Field(default=None)
    
    # Kalshi-specific fields (Legacy/Extra)
    event_ticker: Optional[str] = Field(default=None)
    market_ticker: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    subtitle: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    series_ticker: Optional[str] = Field(default=None)
    strike_type: Optional[str] = Field(default=None)
    floor_strike: Optional[float] = Field(default=None)
    cap_strike: Optional[float] = Field(default=None)
    open_time: Optional[float] = Field(default=None)
    close_time: Optional[float] = Field(default=None)
    expiration_time: Optional[float] = Field(default=None)
    settlement_value: Optional[float] = Field(default=None)
    result: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    open_interest: Optional[float] = Field(default=None)
    liquidity: Optional[float] = Field(default=None)
    yes_bid: Optional[float] = Field(default=None)
    yes_ask: Optional[float] = Field(default=None)
    no_bid: Optional[float] = Field(default=None)
    no_ask: Optional[float] = Field(default=None)
    last_price: Optional[float] = Field(default=None)
    settlement_source: Optional[str] = Field(default=None)
    snapshot_history: Optional[List[Dict[str, Union[int, float, str]]]] = Field(
        default=None
    )
    rules: Optional[str] = Field(default=None)
    
    # Raw data
    kalshi_raw: Optional[Dict] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict) -> 'KalshiData':
        """
        Create KalshiData instance from dictionary.
        Handles legacy Kalshi fields mapping to PolyMarket format if needed.
        """
        # Convert timestamp strings to floats if needed
        for ts_field in ['open_time', 'close_time', 'expiration_time']:
            if ts_field in data and isinstance(data[ts_field], str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(data[ts_field].replace('Z', '+00:00'))
                    data[ts_field] = dt.timestamp()
                except (ValueError, AttributeError):
                    pass

        # Map legacy/raw Kalshi fields to PolyMarket format if they are missing
        if 'event_id' not in data and 'event_ticker' in data:
            data['event_id'] = data['event_ticker']
        if 'market_id' not in data and 'market_ticker' in data:
            data['market_id'] = data['market_ticker']
        if 'question' not in data and 'title' in data:
            data['question'] = data['title']
        if 'start_ts' not in data and 'open_time' in data:
            data['start_ts'] = data['open_time']
        if 'end_ts' not in data and 'close_time' in data:
            data['end_ts'] = data['close_time']
        if 'volumn' not in data and 'volume' in data:
            data['volumn'] = data['volume']
        if 'outcome' not in data and 'result' in data:
            data['outcome'] = data['result']
            
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'


class DailyNewsData(BaseModel):
    uuid: str
    title: str
    date: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DailyNewsData':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'
