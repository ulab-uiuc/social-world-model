from typing import List, Dict, Union, Optional
from pydantic import BaseModel, ConfigDict, Field

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