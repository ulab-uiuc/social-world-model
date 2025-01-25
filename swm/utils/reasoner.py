# swm/utils.py

import json
from datetime import datetime
from typing import Dict, List, Optional

import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import DailyNewsData, PolyMarketData
from .filter import TimeBasedDailyNewsFilter, TimeBasedPolyMarketFilter
from .prompter import model_prompting


class PolyMarketDailyNewsReasoner:
    def __init__(
        self,
        corpus_markets: List[PolyMarketData],
        corpus_news: List[DailyNewsData],
        top_k: int = 5,
        news_window_days: int = 1,
        openai_api_key: Optional[str] = None,
        model_name: str = 'gpt-4o',
    ):
        self.corpus_markets = {market.market_id: market for market in corpus_markets}
        self.top_k = top_k
        self.news_window_days = news_window_days
        self.polymarket_filter = TimeBasedPolyMarketFilter(corpus_markets)
        self.news_filter = TimeBasedDailyNewsFilter(corpus_news)

        openai.api_key = openai_api_key

    def analyze(self, date: str) -> List[Dict]:
        top_changes = self._get_top_market_changes(date)
        if not top_changes:
            return []

        target_date_news = self.news_filter.filter(date)
        if not target_date_news:
            return []

        prompt = self._create_prompt(top_changes, date, target_date_news)
        model_output = self._get_model_response(prompt)
        parsed_results = self._parse_scores(model_output, target_date_news)
        return parsed_results, top_changes

    def _get_top_market_changes(self, date: str) -> List[Dict]:
        changes = []
        for market in self.corpus_markets.values():
            if not market.daily_time_series:
                continue
            series = market.daily_time_series.get('Yes', [])
            for i, point in enumerate(series):
                current_date = datetime.fromtimestamp(series[i]['t']).strftime('%Y-%m-%d')
                if current_date == date and i < len(series) - 1:
                    change = abs(point['p'] - series[i+1]['p'])
                    changes.append(
                        {
                            'market': market,
                            'prev_point': series[i+1],
                            'current_point': point,
                            'change': change,
                        }
                    )
        sorted_changes = sorted(changes, key=lambda x: x['change'], reverse=True)
        return sorted_changes[: self.top_k]

    def _create_prompt(
        self, changes: List[Dict], date: str, news: List[DailyNewsData]
    ) -> str:
        prompt = (
            f'Analyze which news caused these significant market changes on {date}:\n\n'
        )
        for change in changes:
            market = change['market']
            direction = (
                'increased'
                if change['current_point']['p'] > change['prev_point']['p']
                else 'decreased'
            )
            prompt += f"- {market.question}: {direction} from {change['prev_point']['p']:.3f} to {change['current_point']['p']:.3f}\n"

        prompt += '\nNews:\n'
        for idx, item in enumerate(news):
            prompt += f'- [news_id{idx}] {item.title}: {item.description}\n'

        prompt += "\nRate each news item's likelihood (0-100) of causing these market changes."
        prompt += '\nReturn: JSON array of objects with "news_id" and "score" fields matching news order. Do not return anything else.'
        return prompt

    def _get_model_response(self, prompt: str) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        response = model_prompting(
            llm_model='gpt-4o',
            messages=messages,
            temperature=0.0,
            max_token_num=2048,
        )[0]
        return response


    def _parse_scores(self, model_output: str, news: List[DailyNewsData]) -> List[Dict]:
        parsed_results = []
        try:
            start = model_output.index('[')
            end = model_output.rindex(']') + 1
            json_str = model_output[start:end]
            results = json.loads(json_str)
            for result in results:
                parsed_result = {}
                parsed_result['news'] = news[result['news_id']]
                parsed_result['score'] = result['score'] / 100
                parsed_results.append(parsed_result)
            return parsed_results
        except (ValueError, json.JSONDecodeError):
            return []
