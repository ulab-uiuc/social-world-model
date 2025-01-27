import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import jsonlines

from ..data import DailyNewsData, PolyMarketData
from .error_handler import api_calling_error_exponential_backoff, parsing_error_exponential_backoff
from .filter import TimeBasedDailyNewsFilter
from .prompter import model_prompting
from .utils import convert_to_date
import openai

PROMPT_TEMPLATE = """Analyze market price change causation for {date}:

Market: {question}
Current Price Change: {direction} from {current_price:.3f} ({current_date}) to {next_price:.3f} ({next_date}) ({change_pct:.1f}%)

Historical Price Data (Previous 5 days):
{historical_data}

News:
{news_items}

Task: Rate each news item's likelihood (0-100) of causing this price change.
Format: Return JSON array of objects with "news_id" and "score" fields. Example:
[{{"news_id": 0, "score": 85}}, {{"news_id": 1, "score": 15}}]"""


class BasicPosteriorReasoner:
    def __init__(
        self,
        corpus_news: List[DailyNewsData],
        model_name: str = 'gpt-4o-mini',
        max_news_items: int = 10,
        change_threshold: float = 0.25,
        cache_dir: str = './reasoning_cache',
        history_days: int = 5
    ):
        self.news_filter = TimeBasedDailyNewsFilter(corpus_news)
        self.model_name = model_name
        self.max_news_items = max_news_items
        self.change_threshold = change_threshold
        self.cache_dir = Path(cache_dir)
        self.history_days = history_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, time: str, market_id: str) -> str:
        """Generate a unique cache key for the given time and market."""
        key = f'{market_id}_{time}_{self.model_name}'
        return key

    def _get_historical_data(
        self, 
        date: datetime, 
        market: PolyMarketData
    ) -> List[Dict[str, Any]]:
        """Retrieve historical price data for the specified number of days."""
        if not market.daily_time_series or 'Yes' not in market.daily_time_series:
            return []

        historical_data = []
        series = market.daily_time_series['Yes']
        
        for i in range(self.history_days, 0, -1):
            historical_date = date - timedelta(days=i)
            next_date = historical_date + timedelta(days=1)

            current_point = next(
                (p for p in series if datetime.fromtimestamp(p['t']).date() == historical_date.date()),
                None
            )
            next_point = next(
                (p for p in series if datetime.fromtimestamp(p['t']).date() == next_date.date()),
                None
            )

            if current_point and next_point:
                change_pct = ((next_point['p'] - current_point['p']) / current_point['p']) * 100
                historical_data.append({
                    'date': historical_date.strftime('%Y-%m-%d'),
                    'start_price': current_point['p'],
                    'end_price': next_point['p'],
                    'change_pct': change_pct
                })

        return historical_data

    def _format_historical_data(self, historical_data: List[Dict[str, Any]]) -> str:
        """Format historical data for the prompt."""
        if not historical_data:
            return "No historical data available."
            
        formatted_data = []
        for data in historical_data:
            formatted_data.append(
                f"{data['date']}: {data['start_price']:.3f} → {data['end_price']:.3f} "
                f"({data['change_pct']:+.1f}%)"
            )
        return "\n".join(formatted_data)

    def _get_price_change(
        self, 
        date: datetime, 
        market: PolyMarketData
    ) -> Optional[Dict[str, Any]]:
        """Get price change data for the specified date."""
        if not market.daily_time_series or 'Yes' not in market.daily_time_series:
            return None

        next_date = date + timedelta(days=1)
        series = market.daily_time_series['Yes']

        current_point = next(
            (p for p in series if datetime.fromtimestamp(p['t']).date() == date.date()),
            None
        )
        next_point = next(
            (p for p in series if datetime.fromtimestamp(p['t']).date() == next_date.date()),
            None
        )

        if not (current_point and next_point):
            return None

        price_change = next_point['p'] - current_point['p']
        change_pct = (price_change / current_point['p']) * 100

        return {
            'current_point': current_point,
            'next_point': next_point,
            'change': abs(price_change),
            'change_pct': change_pct,
            'direction': 'increased' if price_change > 0 else 'decreased'
        }

    def reason(
        self, 
        time: Union[str, int], 
        market: PolyMarketData
    ) -> List[Dict[str, Any]]:
        """Main reasoning method with caching."""
        cache_key = self._get_cache_key(str(time), market.market_id)
        cache_path = self.cache_dir / f'{cache_key}.json'

        # Try to load from cache
        if cache_path.exists():
            try:
                with jsonlines.open(cache_path, mode='r') as reader:
                    serialized_results = list(reader)
                return [
                    {'news': DailyNewsData.from_dict(r['news']), 'score': r['score']}
                    for r in serialized_results
                ]
            except (json.JSONDecodeError, IOError):
                pass

        # Compute new results
        results = self._compute_reasoning(time, market)
        if not results:
            return []

        # Cache the results
        try:
            serialized_results = [
                {
                    'news': r['news'].model_dump(),
                    'score': r['score'],
                    'time': time,
                    'market': market.market_id,
                }
                for r in results
            ]
            with jsonlines.open(cache_path, mode='w') as writer:
                writer.write_all(serialized_results)
        except IOError:
            pass

        return results[:self.max_news_items]

    def _compute_reasoning(
        self, 
        time: Union[str, int], 
        market: PolyMarketData
    ) -> List[Dict[str, Any]]:
        """Compute reasoning results for the given time and market."""
        date = datetime.strptime(convert_to_date(time), '%Y-%m-%d')
        
        # Get price change and historical data
        price_change = self._get_price_change(date, market)
        if not price_change or abs(price_change['change']) < self.change_threshold:
            return []

        historical_data = self._get_historical_data(date, market)
        news = self._get_filtered_news(date.strftime('%Y-%m-%d'))
        if not news:
            return []

        # Create and process prompt
        prompt = self._create_prompt(price_change, date, news, market, historical_data)
        response = self._get_model_response(prompt)
        return self._parse_scores(response, news)

    def _get_filtered_news(self, date: str) -> List[DailyNewsData]:
        """Get filtered news for the specified date."""
        return self.news_filter.filter(date)

    def _create_prompt(
        self, 
        change: Dict[str, Any],
        date: datetime,
        news: List[DailyNewsData],
        market: PolyMarketData,
        historical_data: List[Dict[str, Any]]
    ) -> str:
        """Create a prompt with historical context."""
        return PROMPT_TEMPLATE.format(
            date=date.strftime('%Y-%m-%d'),
            question=market.question,
            direction=change['direction'],
            current_price=change['current_point']['p'],
            next_price=change['next_point']['p'],
            current_date=datetime.fromtimestamp(change['current_point']['t']).strftime('%Y-%m-%d'),
            next_date=datetime.fromtimestamp(change['next_point']['t']).strftime('%Y-%m-%d'),
            change_pct=change['change_pct'],
            historical_data=self._format_historical_data(historical_data),
            news_items=self._format_news_items(news)
        )

    def _format_news_items(self, news: List[DailyNewsData]) -> str:
        """Format news items for the prompt."""
        return '\n'.join(
            f'[news_id{i}] {item.title}: {item.description}'
            for i, item in enumerate(news)
        )

    @api_calling_error_exponential_backoff()
    def _get_model_response(self, prompt: str) -> List[Dict[str, Any]]:
        """Get model response using OpenAI's chat completion API with function calling."""
        functions = [{
            "name": "rate_news_impact",
            "description": "Rate the likelihood of each news item causing the observed market price change",
            "parameters": {
                "type": "object",
                "properties": {
                    "news_ratings": {
                        "type": "array",
                        "description": "Array of ratings for each news item",
                        "items": {
                            "type": "object",
                            "properties": {
                                "news_id": {
                                    "type": "integer",
                                    "description": "ID of the news item"
                                },
                                "score": {
                                    "type": "number",
                                    "description": "Impact score from 0-100",
                                    "minimum": 0,
                                    "maximum": 100
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation for the score"
                                }
                            },
                            "required": ["news_id", "score", "reasoning"]
                        }
                    }
                },
                "required": ["news_ratings"]
            }
        }]

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze news impact on prediction markets."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                functions=functions,
                function_call={"name": "rate_news_impact"},
                temperature=0.0,
                max_tokens=3064
            )

            # Extract function call results
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return function_args.get('news_ratings', [])

        except Exception as e:
            print(f"Error in model response: {e}")
            return []


    @parsing_error_exponential_backoff()
    def _parse_scores(
        self, 
        results: Dict[str, Any],
        news: List[DailyNewsData]
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        # Normalize scores and sort by score
        scored_news = [
            {
                'news': news[r['news_id']],
                'score': r['score'] / 100
            }
            for r in results
        ]
        return sorted(scored_news, key=lambda x: x['score'], reverse=True)