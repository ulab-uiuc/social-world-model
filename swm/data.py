"""
Unified data models for prediction market data.
Works for both Polymarket and Kalshi data.
"""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """
    Unified data model for prediction market data.
    Compatible with Polymarket, Kalshi, and other platforms.
    """
    # Core required fields
    event_id: str
    market_id: str
    question: str
    
    # Common optional fields
    resolution_source: Optional[str] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    outcome: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    start_ts: Optional[float] = Field(default=None)
    end_ts: Optional[float] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    tag_ids: Optional[List[str]] = Field(default=None)
    categories: Optional[List[str]] = Field(default=None)
    
    # Time series data
    time_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    daily_time_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    window_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)
    
    # Breakpoints with all preprocessing done
    # Each breakpoint contains: before, after, change, z_score, window_history, news, attributions
    daily_breakpoints: Optional[List[Dict[str, Any]]] = Field(default=None)
    
    # Platform identifier (optional)
    platform: Optional[str] = Field(default=None)  # "polymarket", "kalshi", etc.
    
    # Platform-specific fields (stored as extra, accessible via model_extra)
    # Kalshi: event_ticker, market_ticker, series_ticker, etc.
    # Polymarket: clobTokenIds, etc.

    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketData':
        """Create MarketData from dictionary, handling legacy field names."""
        # Handle legacy Kalshi field names
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
        if 'outcome' not in data and 'result' in data:
            data['outcome'] = data['result']
        
        # Convert timestamp strings to floats
        for ts_field in ['start_ts', 'end_ts', 'open_time', 'close_time', 'expiration_time']:
            if ts_field in data and isinstance(data[ts_field], str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(data[ts_field].replace('Z', '+00:00'))
                    data[ts_field] = dt.timestamp()
                except (ValueError, AttributeError):
                    pass
        
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'  # Allow platform-specific fields


class DailyNewsData(BaseModel):
    """News article data."""
    uuid: Optional[str] = Field(default=None)
    title: str
    description: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)
    published_at: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DailyNewsData':
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'


# Backward compatibility aliases
PolyMarketData = MarketData
KalshiData = MarketData
