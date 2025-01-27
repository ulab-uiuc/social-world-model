import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from transformers import TrainingArguments

from .data import DailyNewsData, PolyMarketData
from .utils.error_handler import (
    api_calling_error_exponential_backoff,
    parsing_error_exponential_backoff,
)
from .utils.filter import TimeBasedDailyNewsFilter
from .utils.prompter import model_prompting
from .utils.utils import convert_to_date
from pathlib import Path
import hashlib
import jsonlines

PROMPT_TEMPLATE = """Analyze market price change causation for {date}:

Market: {question}
Price Change: {direction} from {current_price:.3f} ({current_date}) to {next_price:.3f} ({next_date}) ({change_pct:.1f}%)

News:
{news_items}

Task: Rate each news item's likelihood (0-100) of causing this price change.
Format: Return JSON array of objects with "news_id" and "score" fields. Example:
[{{"news_id": 0, "score": 85}}, {{"news_id": 1, "score": 15}}]"""



class BasicPosteriorReasoner:
    def __init__(
        self,
        corpus_news: List[DailyNewsData],
        model_name: str = 'gpt-4',
        max_news_items: int = 10,
        change_threshold: float = 0.25,
        cache_dir: str = './reasoning_cache'
    ):
        self.news_filter = TimeBasedDailyNewsFilter(corpus_news)
        self.model_name = model_name
        self.max_news_items = max_news_items
        self.change_threshold = change_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, time: Union[str, int], market_id: str) -> str:
        key = f"{market_id}_{time}"
        return hashlib.md5(key.encode()).hexdigest()

    def reason(
        self, time: Union[str, int], market: PolyMarketData
    ) -> List[Dict[str, Any]]:
        cache_key = self._get_cache_key(time, market.market_id)
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if cache_path.exists():
            try:
                with jsonlines.open(cache_path, mode='r') as reader:
                    serialized_results = list(reader)
                results = [{'news': DailyNewsData.from_dict(r['news']), 'score': r['score']} for r in serialized_results]
            except (json.JSONDecodeError, IOError):
                pass

        results = self._compute_reasoning(time, market)
        
        try:
            serialized_results = [{'news': r['news'].model_dump(), 'score': r['score']} for r in results]
            with jsonlines.open(cache_path, mode='w') as writer:
                writer.write_all(serialized_results)
        except IOError:
            pass
            
        return results

    def _compute_reasoning(
        self, time: Union[str, int], market: PolyMarketData
    ) -> List[Dict[str, Any]]:
        """Actual reasoning computation."""
        date = convert_to_date(time)
        change = self._get_next_day_change(date, market)
        if not change:
            return []

        if abs(change['change']) < self.change_threshold:
            news = self._get_filtered_news(date)
            return [{'news': news_item, 'score': 0.01} for news_item in news][
                : self.max_news_items
            ]

        news = self._get_filtered_news(date)
        if not news:
            return []

        prompt = self._create_prompt(change, date, news)
        response = self._get_model_response(prompt)
        return self._parse_scores(response, news)


    def _get_next_day_change(self, date: str, market: PolyMarketData) -> Optional[Dict]:
        if not market.daily_time_series or 'Yes' not in market.daily_time_series:
            return None

        current_date = datetime.strptime(date, '%Y-%m-%d')
        next_date = current_date + timedelta(days=1)
        series = market.daily_time_series['Yes']

        current_point = next(
            (
                p
                for p in series
                if datetime.fromtimestamp(p['t']).date() == current_date.date()
            ),
            None,
        )
        next_point = next(
            (
                p
                for p in series
                if datetime.fromtimestamp(p['t']).date() == next_date.date()
            ),
            None,
        )

        if not (current_point and next_point):
            return None

        change = abs(current_point['p'] - next_point['p'])

        return {
            'market': market,
            'current_point': current_point,
            'next_point': next_point,
            'change': change,
            'direction': 'increased'
            if next_point['p'] > current_point['p']
            else 'decreased',
        }

    def _get_filtered_news(self, date: str) -> List[DailyNewsData]:
        news = self.news_filter.filter(date)
        return news

    def _create_prompt(self, change: Dict, date: str, news: List[DailyNewsData]) -> str:
        return PROMPT_TEMPLATE.format(
            date=date,
            question=change['market'].question,
            direction=change['direction'],
            current_price=change['current_point']['p'],
            next_price=change['next_point']['p'],
            current_date=change['current_point']['t'],
            next_date=change['next_point']['t'],
            change_pct=abs(change['change']) * 100,
            news_items=self._format_news_items(news),
        )

    def _format_news_items(self, news: List[DailyNewsData]) -> str:
        return '\n'.join(
            f'[news_id{i}] {item.title}: {item.description}'
            for i, item in enumerate(news)
        )

    @api_calling_error_exponential_backoff()
    def _get_model_response(self, prompt: str) -> str:
        messages = [
            {
                'role': 'system',
                'content': 'You analyze news impact on prediction markets.',
            },
            {'role': 'user', 'content': prompt},
        ]
        return model_prompting(
            llm_model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_token_num=2048,
        )[0]

    @parsing_error_exponential_backoff()
    def _parse_scores(self, model_output: str, news: List[DailyNewsData]) -> List[Dict]:
        json_match = re.search(r'\[.*\]', model_output, re.DOTALL)
        if not json_match:
            return []

        results = json.loads(json_match.group())
        if not results:
            return []

        # Normalize scores and select top-k news items
        scored_news = [
            {
                'news': news[r['news_id']],
                'score': r['score'] / 100 if r['score'] > 0 else 0.01,
            }
            for r in results
        ]
        scored_news.sort(key=lambda x: x['score'], reverse=True)
        return scored_news[: self.max_news_items]


class BasicPriorReasoner:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
    ):
        return

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
        posterior_reasoner: BasicPosteriorReasoner,
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
