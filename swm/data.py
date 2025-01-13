from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class Consensus(BaseModel):
    event_id: str
    market_id: str
    question: str
    resolution_source: str
    volumn: Optional[float] = Field(default=None)
    outcome: Optional[str] = Field(default=None)
    discrption: Optional[str] = Field(default=None)
    start_ts: Optional[float] = Field(default=None)
    end_ts: Optional[float] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    tag_ids: Optional[List[str]] = Field(default=None)
    time_series: Optional[Dict[str, List[Dict[str, Union[int, float]]]]] = Field(
        default=None
    )
    breakpoint_ts_pairs: Optional[Dict[str, List[Tuple[int, int]]]] = Field(
        default=None
    )

    @classmethod
    def from_dict(cls, data: Dict) -> 'Consensus':
        """Create a Consensus instance from a dictionary (e.g., from model_dump())"""
        # Handle the special case of breakpoint_ts_pairs to convert lists to tuples
        if 'breakpoint_ts_pairs' in data and data['breakpoint_ts_pairs']:
            data['breakpoint_ts_pairs'] = {
                key: [tuple(pair) for pair in pairs]
                for key, pairs in data['breakpoint_ts_pairs'].items()
            }

        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class Action(BaseModel):
    event_id: str
    market_id: str
    question: str
    resolution_source: str
    volumn: Optional[float] = Field(default=None)
    end_ts: Optional[float] = Field(default=None)
    outcome: Optional[str] = Field(default=None)
    discrption: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    tag_ids: Optional[List[str]] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        """Create an Action instance from a dictionary (e.g., from model_dump())"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
