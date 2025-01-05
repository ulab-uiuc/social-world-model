import datetime
import json
import os
import time
from typing import Dict, List, Union

import httpx
import jsonlines
import matplotlib.pyplot as plt
from py_clob_client.client import ClobClient
from tqdm import tqdm


def visualize_price_history(history: List[Dict[str, Union[int, float]]],
                          title: str = 'Price History',
                          save_path: str = None):
    timestamps = [item['t'] for item in history]
    prices = [item['p'] for item in history]
    datetimes = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, prices, marker='o', linestyle='-', color='b')
    plt.xlabel('Date/Time')
    plt.ylabel('Price')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def get_history_from_token_id(token_id: str, fidelity: int = 60, max_retries: int = 5) -> List[Dict[str, Union[int, float]]]:
    host = "https://clob.polymarket.com"
    key = os.getenv("PK")
    chain_id = 137

    if not key:
        raise ValueError("Private key not found. Please set PK in the environment variables.")

    client = ClobClient(host, key=key, chain_id=chain_id)

    for attempt in range(max_retries):
        try:
            price_data = client.get_price_history_for_interval(
                token_id=token_id,
                interval="max",
                fidelity=fidelity
            )
            return price_data['history']

        except Exception as e:
            wait_time = (2 ** attempt)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            print(f"Waiting {wait_time} seconds before retry...")

            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed for token {token_id}")
                return []

def get_market(market_id: str | int) -> dict:
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    response = httpx.get(url)

    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch market data: HTTP {response.status_code}")

def get_events(active: bool = False, closed: bool = True, archived: bool = False, limit: int = 4) -> List[Dict]:
    params = {
        "active": active,
        "closed": closed,
        "archived": archived,
        "limit": limit,
        "end_date_max": "2025-01-05T00:00:00Z",
        "start_date_min": "2024-12-05T00:00:00Z",
    }

    response = httpx.get("https://gamma-api.polymarket.com/events", params=params)
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch events: HTTP {response.status_code}")


def get_event_from_id(event_id: str | int) -> dict:
    params = {
        "active": False,
        "closed": True,
        "end_date_max": "2025-01-05T00:00:00Z",
        "start_date_min": "2024-12-05T00:00:00Z",
        "id": event_id,
    }

    response = httpx.get("https://gamma-api.polymarket.com/events", params=params)
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch events: HTTP {response.status_code}")


def load_existing_data(filename: str) -> Dict[str, Dict]:
    """Load existing data and create a map of token_ids that have already been processed"""
    existing_tokens = {}
    if os.path.exists(filename):
        with jsonlines.open(filename, "r") as reader:
            for event in reader:
                for market in event['markets']:
                    if 'history' in market:
                        for token_id in market['history'].keys():
                            existing_tokens[token_id] = market['history'][token_id]
    return existing_tokens


if __name__ == '__main__':
    output_file = "data.jsonl"
    existing_tokens = load_existing_data(output_file)
    print(f"Found {len(existing_tokens)} existing token histories")

    current_events = get_event_from_id(event_id="15802")
    import pdb; pdb.set_trace()

    for idx, event in tqdm(enumerate(current_events), total=len(current_events)):
        for idy, market in enumerate(event['markets']):
            token_ids = json.loads(market['clobTokenIds'])
            current_events[idx]['markets'][idy]['history'] = {}
            import pdb; pdb.set_trace()
            for token_id in token_ids:
                if token_id in existing_tokens:
                    print(f"Skipping existing token {token_id}")
                    current_events[idx]['markets'][idy]['history'][token_id] = existing_tokens[token_id]
                    continue
                history = get_history_from_token_id(token_id)
                if history == []:
                    continue
                current_events[idx]['markets'][idy]['history'][token_id] = history
                existing_tokens[token_id] = history

        # Write after each event is processed
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(current_events)
