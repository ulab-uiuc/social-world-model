from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class PolyMarketData(BaseModel):
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
    time_series: Optional[Dict[str, List[Dict[str, Union[int, float]]]]] = Field(
        default=None
    )
    breakpoint_ts_pairs: Optional[Dict[str, List[Tuple[float, float, float]]]] = Field(
        default=None
    )
    window_series: Optional[List[Dict[str, Union[int, float]]]] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolyMarketData':
        if 'breakpoint_ts_pairs' in data and data['breakpoint_ts_pairs']:
            data['breakpoint_ts_pairs'] = {
                key: [tuple(pair) for pair in pairs]
                for key, pairs in data['breakpoint_ts_pairs'].items()
            }

        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class DailyNewsData(BaseModel):
    uuid: str
    title: str
    description: str
