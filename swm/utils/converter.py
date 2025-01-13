from ..data import Consensus
from typing import List

def convert_polymarket_event_into_consensus(event: dict) -> List[Consensus]:
    markets = event['markets']
    for market in markets:
        import pdb; pdb.set_trace()
        