# swm/utils.py

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import openai
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..data import DailyNewsData, PolyMarketData
from .prompter import model_prompting
from .retriever import DailyNewsRetriever


class PolyMarketDailyNewsReasoner:
    def __init__(
        self,
        markets: List[PolyMarketData],
        news: List[DailyNewsData],
        top_k: int = 5,
        news_window_days: int = 1,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt2",
    ):
        self.markets = {market.market_id: market for market in markets}
        self.top_k = top_k
        self.news_window_days = news_window_days
        self.news_retriever = DailyNewsRetriever(news)

        self.use_openai = bool(openai_api_key)
        if self.use_openai:
            openai.api_key = openai_api_key
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def analyze(self, date: str) -> List[Dict]:
        top_changes = self._get_top_market_changes(date)
        if not top_changes:
            return []

        relevant_news = self.news_retriever.get_relevant_news(date, self.news_window_days)
        if not relevant_news:
            return []

        prompt = self._create_prompt(top_changes, date, relevant_news)
        model_output = self._get_model_response(prompt)
        return self._parse_scores(model_output)

    def _get_top_market_changes(self, date: str) -> List[Dict]:
        changes = []
        for market in self.markets.values():
            if not market.daily_time_series:
                continue
            for outcome, series in market.daily_time_series.items():
                for i, point in enumerate(series):
                    current_date = datetime.fromtimestamp(point['t']).strftime('%Y-%m-%d')
                    if current_date == date and i > 0:
                        change = abs(point['p'] - series[i - 1]['p'])
                        changes.append({
                            'market': market,
                            'outcome': outcome,
                            'prev_point': series[i - 1],
                            'current_point': point,
                            'change': change,
                        })
        sorted_changes = sorted(changes, key=lambda x: x['change'], reverse=True)
        return sorted_changes[:self.top_k]

    def _create_prompt(
        self,
        changes: List[Dict],
        date: str,
        news: List[DailyNewsData]
    ) -> str:
        prompt = f"Analyze which news caused these significant market changes on {date}:\n\n"
        for change in changes:
            market = change['market']
            direction = 'increased' if change['current_point']['p'] > change['prev_point']['p'] else 'decreased'
            prompt += f"- {market.question}: {direction} from {change['prev_point']['p']:.3f} to {change['current_point']['p']:.3f}\n"

        prompt += "\nNews:\n"
        for item in news:
            prompt += f"- {item.title}: {item.description}\n"

        prompt += "\nRate each news item's likelihood (0-100) of causing these market changes."
        prompt += '\nReturn: JSON array of objects with "news" and "score" fields matching news order.'
        return prompt

    def _get_model_response(self, prompt: str) -> str:
        if self.use_openai:
            messages = [{"role": "user", "content": prompt}]
            response = model_prompting(llm_model='gpt-4o', messages=messages)[0]
            return response
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
            outputs = self.model.generate(**inputs, max_length=1024)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _parse_scores(self, model_output: str) -> List[Dict]:
        try:
            start = model_output.index('[')
            end = model_output.rindex(']') + 1
            json_str = model_output[start:end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            return []
