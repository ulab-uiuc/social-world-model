import datetime
import json
import os
import time
from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor
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

def get_events(active: bool = False, closed: bool = True, archived: bool = False, limit: int = 100) -> List[Dict]:
    params = {
        "active": active,
        "closed": closed,
        "archived": archived,
        "limit": limit,
        "end_date_max": "2025-01-05T00:00:00Z",
        "end_date_min": "2024-12-05T00:00:00Z",
    }

    response = httpx.get("https://gamma-api.polymarket.com/events", params=params)
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch events: HTTP {response.status_code}")


def get_event_from_offset(offset: str | int) -> dict:
    response = httpx.get(f"https://gamma-api.polymarket.com/events?offset={offset}&limit=100")
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch events: HTTP {response.status_code}")


class DataManager:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.existing_tokens = {}
        self.cache_size = 100  # Number of events to buffer before writing
        self.event_buffer = []
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Load existing data using a more efficient streaming approach"""
        if not os.path.exists(self.output_file):
            return

        with jsonlines.open(self.output_file, "r") as reader:
            for event in reader:
                for market in event.get('markets', []):
                    if history := market.get('history'):
                        self.existing_tokens.update(history)

    def _write_buffer(self) -> None:
        """Write buffered events to file"""
        if not self.event_buffer:
            return

        with jsonlines.open(self.output_file, mode="a") as writer:
            writer.write_all(self.event_buffer)
        self.event_buffer = []

    def process_token(self, token_id: str) -> Dict:
        """Process a single token, either from cache or by fetching"""
        if token_id in self.existing_tokens:
            return self.existing_tokens[token_id]
        
        history = get_history_from_token_id(token_id)
        if history:
            self.existing_tokens[token_id] = history
        return history

    def process_events(self, events: List[Dict]) -> None:
        """Process events with buffered writing and parallel token processing"""
        if not events:
            return

        for event in tqdm(events, desc="Processing events"):
            modified_event = event.copy()
            
            for market in modified_event.get('markets', []):
                token_ids = json.loads(market.get('clobTokenIds', '[]'))
                market['history'] = {}
                
                # Process tokens in parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    token_futures = {
                        token_id: executor.submit(self.process_token, token_id)
                        for token_id in token_ids
                    }
                    
                    for token_id, future in token_futures.items():
                        if history := future.result():
                            market['history'][token_id] = history

            self.event_buffer.append(modified_event)
            
            # Write to file when buffer reaches threshold
            if len(self.event_buffer) >= self.cache_size:
                self._write_buffer()
        
        # Write any remaining events
        self._write_buffer()

def main():
    data_manager = DataManager("data_with_offset_0112.jsonl")
    print(f"Found {len(data_manager.existing_tokens)} existing token histories")

    for offset in range(1, 11000, 100):
        print(f"Processing offset: {offset}")
        try:
            current_events = get_event_from_offset(offset=str(offset))
            data_manager.process_events(current_events)
        except Exception as e:
            print(f"Error fetching event with offset {offset}: {e}")
            continue

if __name__ == '__main__':
    main()
