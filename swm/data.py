from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class Consensus(BaseModel):
    question: str
    answer: str
    is_outcome: Optional[bool] = Field(default=None)
    discrption: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    time_series: Optional[Dict[str, Union[int, float]]] = Field(default=None)


class Action(BaseModel):
    question: str
    answer: str
    description: Optional[str] = Field(default=None)
