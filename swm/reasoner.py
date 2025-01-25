# swm/utils.py

import json
from datetime import datetime
from typing import Dict, List, Optional

import openai

from ..data import DailyNewsData, PolyMarketData
from .filter import TimeBasedDailyNewsFilter, TimeBasedPolyMarketFilter
from .prompter import model_prompting


class PolyMarketDailyNewsPosteriorReasoner:
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
                current_date = datetime.fromtimestamp(series[i]['t']).strftime(
                    '%Y-%m-%d'
                )
                if current_date == date and i < len(series) - 1:
                    change = abs(point['p'] - series[i + 1]['p'])
                    changes.append(
                        {
                            'market': market,
                            'prev_point': series[i + 1],
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
                parsed_result['score'] = result['score'] / sum(
                    r['score'] for r in results
                )
                parsed_results.append(parsed_result)
            return parsed_results
        except (ValueError, json.JSONDecodeError):
            return []


class PolyMarketDailyNewsPriorReasoner:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
    ):
        self.model = WeightedBasicSocialWM(
            model_name=model_name, cache_dir=cache_dir, max_seq_length=max_seq_length
        )

    def analyze(
        self, news_data: List[DailyNewsData], market_changes: List[Dict], date: str
    ) -> List[Dict]:
        news_texts = [f'{n.title}: {n.description}' for n in news_data]

        market_change_texts = []
        for change in market_changes:
            market = change['market']
            direction = (
                'increased'
                if change['current_point']['p'] > change['prev_point']['p']
                else 'decreased'
            )
            change_text = (
                f"{market.question}: {direction} from "
                f"{change['prev_point']['p']:.3f} to {change['current_point']['p']:.3f}"
            )
            market_change_texts.append(change_text)

        prompt = self._create_prompt(market_change_texts, date, news_texts)
        scores = self.model.predict([prompt])[0]

        results = []
        for news, score in zip(news_data, scores):
            results.append(
                {
                    'news': news,
                    'score': score,
                }
            )

        return results

    def train(
        self,
        train_dates: List[str],
        valid_dates: List[str],
        posterior_reasoner: PolyMarketDailyNewsPosteriorReasoner,
        training_args: TrainingArguments,
    ) -> str:
        train_data = []
        valid_data = []

        for date in train_dates:
            parsed_results, top_changes = posterior_reasoner.analyze(date)
            if not parsed_results:
                continue

            posterior_scores = [r['score'] for r in parsed_results]
            news_data = [r['news'] for r in parsed_results]
            news_texts = [f'{n.title}: {n.description}' for n in news_data]

            market_changes = []
            for change in top_changes:
                market = change['market']
                direction = (
                    'increased'
                    if change['current_point']['p'] > change['prev_point']['p']
                    else 'decreased'
                )
                change_text = (
                    f"{market.question}: {direction} from "
                    f"{change['prev_point']['p']:.3f} to {change['current_point']['p']:.3f}"
                )
                market_changes.append(change_text)

            prompt = self._create_prompt(market_changes, date, news_texts)
            train_data.append({'prompt': prompt, 'scores': posterior_scores})

        # Do the same for validation data
        for date in valid_dates:
            parsed_results, top_changes = posterior_reasoner.analyze(date)
            if parsed_results:
                prompt = self._process_results(parsed_results, top_changes, date)
                valid_data.append(
                    {'prompt': prompt, 'scores': [r['score'] for r in parsed_results]}
                )

        return self.model.train(
            train_data=train_data, valid_data=valid_data, training_args=training_args
        )

    def _create_prompt(
        self, market_changes: List[str], date: str, news: List[str]
    ) -> str:
        prompt = (
            f'Analyze which news caused these significant market changes on {date}:\n\n'
        )
        prompt += '\n'.join(f'- {change}' for change in market_changes)
        prompt += '\n\nNews:\n'
        prompt += '\n'.join(f'- {news_item}' for news_item in news)
        prompt += "\nRate each news item's likelihood (0-100) of causing these market changes."
        return prompt

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model.load(path)
