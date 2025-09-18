# swm/utils/wrappers.py

from typing import List, Dict, Tuple, Optional
from swm.data import PolyMarketData, DailyNewsData
from swm.reasoner import BasicPriorReasoner


class PriorAsReasoner:
    """
    Wraps a BasicPriorReasoner to provide a .reason() interface,
    compatible with BasicPosteriorReasoner's expected output.
    """
    def __init__(self, prior_reasoner: BasicPriorReasoner):
        self.prior_reasoner = prior_reasoner
        self.cache: Dict[Tuple[str, int], List[Dict]] = {}
        self.model_name = prior_reasoner.model_name

    def reason(self, t: int, market: PolyMarketData) -> List[Dict]:
        key = (market.market_id, t)
        if key not in self.cache:
            # Get prediction result
            preds = self.prior_reasoner.predict([market], posterior_reasoner=None)
            # Expect only one item
            q_dist = preds[0]['q_dist']
            news_items = preds[0]['news']
            self.cache[key] = [
                {'news': news_items[i], 'score': q_dist[i]}
                for i in range(len(q_dist))
            ]
        return self.cache[key]
