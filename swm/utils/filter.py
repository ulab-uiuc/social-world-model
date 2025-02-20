from typing import Dict, List, Optional

from ..data import DailyNewsData, PolyMarketData


class TimeBasedDailyNewsFilter:
    def __init__(self, corpus_news: List[DailyNewsData]):
        self.corpus_news_by_date = self._index_news(corpus_news)

    @staticmethod
    def _extract_date(corpus_news: DailyNewsData) -> Optional[str]:
        return corpus_news.date

    def _index_news(
        self, corpus_news: List[DailyNewsData]
    ) -> Dict[str, List[DailyNewsData]]:
        non_duplicate_news = []
        uuids = set()
        for item in corpus_news:
            if item.uuid not in uuids:
                non_duplicate_news.append(item)
                uuids.add(item.uuid)

        news_dict = {}
        for item in non_duplicate_news:
            date = self._extract_date(item)
            if date:
                news_dict.setdefault(date, []).append(item)
        return news_dict

    def filter(
        self,
        target_date: str,
    ) -> List[DailyNewsData]:
        relevant = self.corpus_news_by_date.get(target_date, [])
        return relevant


class TimeBasedPolyMarketFilter:
    def __init__(self, corpus_markets: List[PolyMarketData]):
        self.corpus_markets = corpus_markets

    def filter(
        self,
        target_ts: int,
    ) -> List[PolyMarketData]:
        relevant_markets = []

        for market in self.corpus_markets:
            if not market.start_ts or not market.end_ts:
                continue
            if not (market.start_ts <= target_ts <= market.end_ts):
                continue
            if not market.daily_time_series:
                continue
            relevant_markets.append(market)

        return relevant_markets
