import datetime
import json
import os
import time
from typing import Dict, List, Union, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import httpx
import jsonlines
import matplotlib.pyplot as plt
from py_clob_client.client import ClobClient
from tqdm import tqdm

def visualize_price_history(
        history: List[Dict[str, Union[int, float]]],
        title: str = 'Price History',
        save_path: str = None
    ):
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

def get_history_from_token_id(token_id: str, fidelity: int = 60, max_retries: int = 5, start_ts: Optional[int] = None) -> List[Dict[str, Union[int, float]]]:
    host = "https://clob.polymarket.com"
    key = os.getenv("PK")
    chain_id = 137

    if not key:
        raise ValueError("Private key not found. Please set PK in the environment variables.")

    client = ClobClient(host, key=key, chain_id=chain_id)

    for attempt in range(max_retries):
        try:
            if start_ts is None:
                price_data = client.get_price_history_for_interval(
                    token_id=token_id,
                    fidelity=fidelity,
                    interval="max",
                )
            else:
                price_data = client.get_price_history_with_start_ts_only(
                    token_id=token_id,
                    fidelity=str(fidelity),
                    start_ts=str(start_ts),
                )
            return price_data['history']

        except Exception as e:
            wait_time = (2 ** attempt)
            print(f"Attempt {attempt + 1}/{max_retries} failed for token {token_id}: {e}")
            print(f"Waiting {wait_time} seconds before retry...")

            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed for token {token_id}")
                return []


def get_event_from_offset(offset: str | int) -> dict:
    """Fetch events from the API with given offset"""
    response = httpx.get(f"https://gamma-api.polymarket.com/events?offset={offset}&limit=100")
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch events: HTTP {response.status_code}")

class EventCollector:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.cache_size = 100
        self.event_buffer = []

    def _write_buffer(self) -> None:
        """Write buffered events to file"""
        if not self.event_buffer:
            return

        mode = "a" if os.path.exists(self.output_file) else "w"
        with jsonlines.open(self.output_file, mode=mode) as writer:
            writer.write_all(self.event_buffer)
        self.event_buffer = []

    def collect_events(self, start_offset: int, end_offset: int, step: int = 100):
        """Collect events from API within the specified offset range"""
        for offset in tqdm(range(start_offset, end_offset, step), desc="Collecting events"):
            try:
                events = get_event_from_offset(offset)
                self.event_buffer.extend(events)

                if len(self.event_buffer) >= self.cache_size:
                    self._write_buffer()
                    print(f"Collected and saved events up to offset {offset}")

            except Exception as e:
                print(f"Error collecting events at offset {offset}: {e}")
                continue

        # Write any remaining events
        self._write_buffer()
        print(f"Completed collecting events from offset {start_offset} to {end_offset}")


def collect_events(start_offset: int, end_offset: int, output_file: str):
    """Convenience function to collect events"""
    collector = EventCollector(output_file)
    collector.collect_events(start_offset, end_offset)

class HistoryCollector:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.cache_size = 5
        self.event_buffer = []
        self.processed_events = self._load_processed_events()

    def _load_processed_events(self) -> Set[str]:
        """Load IDs of already processed events from output file"""
        processed_ids = set()
        if os.path.exists(self.output_file):
            with jsonlines.open(self.output_file, 'r') as reader:
                for event in reader:
                    if 'id' in event:
                        processed_ids.add(str(event['id']))
        return processed_ids

    def _write_buffer(self) -> None:
        """Write buffered events to file"""
        if not self.event_buffer:
            return

        mode = "a" if os.path.exists(self.output_file) else "w"
        with jsonlines.open(self.output_file, mode=mode) as writer:
            writer.write_all(self.event_buffer)
        self.event_buffer = []

    def process_event(self, event: Dict) -> Dict:
        """Process a single event by collecting history for all its markets"""
        modified_event = event.copy()
        
        for market in modified_event.get('markets', []):
            token_ids = json.loads(market.get('clobTokenIds', '[]'))
            start_ts = 1
            market['history'] = {}
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                token_futures = {
                    token_id: executor.submit(get_history_from_token_id, token_id, 60, 5, start_ts)
                    for token_id in token_ids
                }
                
                for token_id, future in token_futures.items():
                    if history := future.result():
                        market['history'][token_id] = history

        return modified_event

    def collect_histories(self):
        """Read events and collect histories for all markets, skipping already processed events"""
        processed_count = 0
        skipped_count = 0
        
        with jsonlines.open(self.input_file, 'r') as reader:
            for event in tqdm(reader, desc="Collecting market histories"):
                try:
                    # Skip if event was already processed
                    event_id = str(event.get('id'))
                    if event_id in self.processed_events:
                        skipped_count += 1
                        if skipped_count % 100 == 0:  # Log progress every 100 skipped events
                            print(f"Skipped {skipped_count} already processed events")
                        continue

                    processed_event = self.process_event(event)
                    self.event_buffer.append(processed_event)
                    self.processed_events.add(event_id)
                    processed_count += 1
                    
                    if len(self.event_buffer) >= self.cache_size:
                        self._write_buffer()
                        print(f"Processed {processed_count} new events")
                
                except Exception as e:
                    print(f"Error processing event {event.get('id', 'unknown')}: {e}")
                    continue
        
        self._write_buffer()
        print(f"Processing complete:")
        print(f"- Total new events processed: {processed_count}")
        print(f"- Total events skipped: {skipped_count}")

def collect_histories(input_file: str, output_file: str):
    """Convenience function to collect histories"""
    collector = HistoryCollector(input_file, output_file)
    collector.collect_histories()

def main():
    # Example usage
    events_file = "data_with_offset_0112.jsonl"
    final_file = "data_with_offset_0112_merged_with_history.jsonl"
    
    # Step 1: Collect events
    #print("Step 1: Collecting events...")
    #collect_events(0, 15000, events_file)
    
    # Step 2: Collect histories
    #print("\nStep 2: Collecting market histories...")
    collect_histories(events_file, final_file)

if __name__ == '__main__':
    main()