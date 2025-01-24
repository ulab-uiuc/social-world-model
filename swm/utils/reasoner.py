from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..data import DailyNewsData, PolyMarketData
from .utils import unix_to_date


class MarketNewsReasoner:
    def __init__(
        self,
        tokenizer,
        model,
        markets: List[PolyMarketData],
        news: List[DailyNewsData],
        top_k: int = 5,
        news_window_days: int = 1,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.markets = {m.market_id: m for m in markets}
        self.news = news
        self.top_k = top_k
        self.news_window_days = news_window_days

        # Index news by date for faster lookup
        self.news_by_date = self._index_news()

    def _index_news(self) -> Dict[str, List[DailyNewsData]]:
        news_dict = {}
        for news_item in self.news:
            date = self._extract_date(news_item)
            if date:
                if date not in news_dict:
                    news_dict[date] = []
                news_dict[date].append(news_item)
        return news_dict

    def _extract_date(self, news: DailyNewsData) -> Optional[str]:
        # Implement date extraction based on your news data structure
        # Example: news.date.strftime('%Y-%m-%d')
        pass

    def _get_relevant_news(self, date: str) -> List[DailyNewsData]:
        target_date = datetime.strptime(date, '%Y-%m-%d')
        relevant_news = []

        for i in range(self.news_window_days + 1):
            check_date = (target_date - timedelta(days=i)).strftime('%Y-%m-%d')
            if check_date in self.news_by_date:
                relevant_news.extend(self.news_by_date[check_date])

        return relevant_news

    def analyze_date_changes(self, date: str) -> List[float]:
        market_changes = []
        for market in self.markets.values():
            if not market.daily_time_series:
                continue

            for outcome, series in market.daily_time_series.items():
                for i, point in enumerate(series):
                    if unix_to_date(point['t']) == date and i > 0:
                        change = abs(point['p'] - series[i - 1]['p'])
                        market_changes.append(
                            {
                                'market': market,
                                'outcome': outcome,
                                'prev_point': series[i - 1],
                                'target_point': point,
                                'abs_change': change,
                            }
                        )

        top_changes = sorted(
            market_changes, key=lambda x: x['abs_change'], reverse=True
        )[: self.top_k]
        if not top_changes:
            return []

        relevant_news = self._get_relevant_news(date)
        if not relevant_news:
            return []

        prompt = self._create_analysis_prompt(top_changes, date, relevant_news)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
        outputs = self.model.generate(**inputs, max_length=1024)
        return self._parse_scores(
            self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        )

    def _create_analysis_prompt(
        self, changes: List[Dict], date: str, news: List[DailyNewsData]
    ) -> str:
        prompt = (
            f'Analyze which news caused these significant market changes on {date}:\n\n'
        )

        for change in changes:
            market = change['market']
            direction = (
                'increased'
                if change['target_point']['p'] > change['prev_point']['p']
                else 'decreased'
            )
            prompt += f"- {market.question}: {direction} from {change['prev_point']['p']:.3f} to {change['target_point']['p']:.3f}\n"

        prompt += '\nNews:\n'
        for n in news:
            prompt += f'- {n.title}: {n.description}\n'

        prompt += "\nRate each news item's likelihood (0-100) of causing these market changes."
        prompt += '\nReturn: JSON array of scores matching news order'
        return prompt

    def _parse_scores(self, model_output: str) -> List[float]:
        import json
        import re

        try:
            match = re.search(r'\[[\d\s,\.]+\]', model_output)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(e)
            pass
        return []
