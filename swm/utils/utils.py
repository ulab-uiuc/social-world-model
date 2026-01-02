import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Union

import jsonlines
import numpy as np
import torch

from ..data import DailyNewsData, PolyMarketData


def extract_search_keywords(question: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Use GPT to extract search keywords from a market question.
    
    Args:
        question: The market question (e.g., "Will Jumper airdrop in 2024?")
        model: OpenAI model to use
        
    Returns:
        Extracted search keywords (e.g., "Jumper airdrop crypto")
    """
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search keyword extractor. Given a prediction market question, "
                        "extract 2-4 key search terms that would find relevant news articles. "
                        "Focus on the main entity/topic names. Remove filler words like 'Will', 'by', 'in 2024'. "
                        "Output ONLY the keywords separated by ' OR ', nothing else. "
                        "Use full names like 'Federal Reserve System' instead of 'FED'. "
                        "Example output: Bitcoin OR halving OR price"
                    )
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            temperature=0,
            max_tokens=50,
        )
        
        keywords = response.choices[0].message.content.strip()
        return keywords
    except Exception as e:
        print(f"  Warning: Failed to extract keywords via GPT: {e}")
        return None


def unix_to_date(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M:%S')


def date_to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())


def filter_midnight_points(series: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Filter to get one point per day, closest to midnight (00:00).
    
    For each date, finds the point with timestamp closest to midnight.
    If there's an exact midnight point, it will be selected.
    Otherwise, the closest point (before or after midnight) is chosen.
    """
    if not series:
        return []
    
    # Group points by date
    date_points: Dict[str, List[Dict[str, float]]] = {}
    for point in series:
        dt = datetime.fromtimestamp(point['t'])
        date_str = dt.strftime('%Y-%m-%d')
        if date_str not in date_points:
            date_points[date_str] = []
        date_points[date_str].append(point)
    
    # For each date, find the point closest to midnight (00:00)
    midnight_points = []
    for date_str, points in sorted(date_points.items()):
        # Calculate distance to midnight for each point
        # Midnight is at hour=0, minute=0, second=0
        def distance_to_midnight(point: Dict[str, float]) -> int:
            dt = datetime.fromtimestamp(point['t'])
            # Seconds since midnight
            seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
            # Also consider distance to next midnight (24 hours - seconds)
            seconds_to_next_midnight = 86400 - seconds_since_midnight
            return min(seconds_since_midnight, seconds_to_next_midnight)
        
        # Find the point with minimum distance to midnight
        closest_point = min(points, key=distance_to_midnight)
        midnight_points.append(closest_point)
    
    return midnight_points


def normalize_timestamp(ts: float) -> int:
    return ts - (ts % 60)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_polymarket_data(data_path):
    with jsonlines.open(data_path, 'r') as reader:
        dataset = list(reader)
        return [PolyMarketData.from_dict(d) for d in dataset]


def load_dailynews_data(data_path):
    with jsonlines.open(data_path, 'r') as reader:
        metadata = list(reader)
        return [DailyNewsData.from_dict(d) for d in metadata]


def convert_to_date(time: Union[float, int, str]) -> str:
    if isinstance(time, int) or isinstance(time, float):
        return datetime.fromtimestamp(time).strftime('%Y-%m-%d')
    elif isinstance(time, str):
        return datetime.strptime(time, '%Y-%m-%d')
    else:
        raise ValueError(f'Invalid time format: {time}')