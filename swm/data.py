"""v6 record schema for prediction-market training data.

Each line of the v6 jsonl files is one record = one training example. No
nesting; no daily_breakpoints. See data/v6/README.md for the full schema.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Record(BaseModel):
    """One v6 training record. Mirrors data/v6/*.jsonl exactly."""

    market_id: str
    event_id: str
    question: str
    description: Optional[str] = None
    categories: Optional[List[str]] = None
    z_score: Optional[float] = None
    news: List[Dict[str, Any]] = Field(default_factory=list)
    attributions: List[Dict[str, Any]] = Field(default_factory=list)
    history: List[Dict[str, float]] = Field(default_factory=list)
    target: Dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Record':
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'
