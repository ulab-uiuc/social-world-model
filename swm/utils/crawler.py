import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Union

import httpx
import jsonlines
from py_clob_client.client import ClobClient
from tqdm import tqdm


class PolyMarketCollector:
    def __init__(self, output_file: str, cache_size: int = 100):
        self.output_file = output_file
        self.cache_size = cache_size
        self.event_buffer = []

    def _write_buffer(self) -> None:
        """Write buffered events to file"""
        if not self.event_buffer:
            return

        mode = 'a' if os.path.exists(self.output_file) else 'w'
        with jsonlines.open(self.output_file, mode=mode) as writer:
            writer.write_all(self.event_buffer)
        self.event_buffer = []

    def _load_processed_events(self, input_file: Optional[str] = None) -> Set[str]:
        """Load processed events to avoid duplication"""
        processed_ids = set()
        if input_file and os.path.exists(input_file):
            with jsonlines.open(input_file, 'r') as reader:
                for event in reader:
                    if 'id' in event:
                        processed_ids.add(str(event['id']))
        return processed_ids

    def collect(self):
        """Override this in subclasses for specific event collection logic."""
        raise NotImplementedError

    def process_event(self, event: Dict) -> Dict:
        """Override this in subclasses to process events"""
        return event


class PolyMarketEventCollector(PolyMarketCollector):
    def __init__(self, output_file: str):
        super().__init__(output_file, cache_size=100)

    def collect(self, start_offset: int, end_offset: int, step: int = 100):
        """Collect events from PolyMarket API and store them in the output file."""
        for offset in tqdm(
            range(start_offset, end_offset, step), desc='Collecting events'
        ):
            try:
                events = self.get_event_from_offset(offset)
                self.event_buffer.extend(events)

                if len(self.event_buffer) >= self.cache_size:
                    self._write_buffer()
                    print(f'Collected and saved events up to offset {offset}')

            except Exception as e:
                print(f'Error collecting events at offset {offset}: {e}')
                continue

        # Write any remaining events
        self._write_buffer()
        print(f'Completed collecting events from offset {start_offset} to {end_offset}')

    def get_event_from_offset(self, offset: Union[str, int]) -> List[Dict]:
        response = httpx.get(
            f'https://gamma-api.polymarket.com/events?offset={offset}&limit=100'
        )
        if response.status_code == 200:
            return response.json()
        raise Exception(f'Failed to fetch events: HTTP {response.status_code}')


class PolyMarketHistoryCollector(PolyMarketCollector):
    def __init__(self, input_file: str, output_file: str):
        super().__init__(output_file, cache_size=5)
        self.input_file = input_file
        self.processed_events = self._load_processed_events(input_file)

    def collect(self):
        """Collect market histories based on the events in the input file."""
        processed_count = 0
        skipped_count = 0

        with jsonlines.open(self.input_file, 'r') as reader:
            for event in tqdm(reader, desc='Collecting market histories'):
                try:
                    # Skip if event was already processed
                    event_id = str(event.get('id'))
                    if event_id in self.processed_events:
                        skipped_count += 1
                        if skipped_count % 100 == 0:
                            print(f'Skipped {skipped_count} already processed events')
                        continue

                    processed_event = self.process_event(event)
                    self.event_buffer.append(processed_event)
                    self.processed_events.add(event_id)
                    processed_count += 1

                    if len(self.event_buffer) >= self.cache_size:
                        self._write_buffer()
                        print(f'Processed {processed_count} new events')

                except Exception as e:
                    print(f"Error processing event {event.get('id', 'unknown')}: {e}")
                    continue

        self._write_buffer()
        print('Processing complete:')
        print(f'- Total new events processed: {processed_count}')
        print(f'- Total events skipped: {skipped_count}')

    def process_event(self, event: Dict) -> Dict:
        """Override this method to modify how event data is processed."""
        modified_event = event.copy()

        for market in modified_event.get('markets', []):
            token_ids = json.loads(market.get('clobTokenIds', '[]'))
            start_ts = 1
            market['history'] = {}

            with ThreadPoolExecutor(max_workers=3) as executor:
                token_futures = {
                    token_id: executor.submit(
                        self.get_history_from_token_id, token_id, 60, 5, start_ts
                    )
                    for token_id in token_ids
                }

                for token_id, future in token_futures.items():
                    if history := future.result():
                        market['history'][token_id] = history

        return modified_event

    def get_history_from_token_id(
        self,
        token_id: str,
        fidelity: int = 60,
        max_retries: int = 5,
        start_ts: Optional[int] = None,
    ) -> List[Dict[str, Union[int, float]]]:
        host = 'https://clob.polymarket.com'
        key = os.getenv('PK')
        chain_id = 137

        if not key:
            raise ValueError(
                'Private key not found. Please set PK in the environment variables.'
            )

        client = ClobClient(host, key=key, chain_id=chain_id)

        for attempt in range(max_retries):
            try:
                if start_ts is None:
                    price_data = client.get_price_history_for_interval(
                        token_id=token_id,
                        fidelity=fidelity,
                        interval='max',
                    )
                else:
                    price_data = client.get_price_history_with_start_ts_only(
                        token_id=token_id,
                        fidelity=str(fidelity),
                        start_ts=str(start_ts),
                    )
                return price_data['history']

            except Exception as e:
                wait_time = 2**attempt
                print(
                    f'Attempt {attempt + 1}/{max_retries} failed for token {token_id}: {e}'
                )
                print(f'Waiting {wait_time} seconds before retry...')

                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f'All {max_retries} attempts failed for token {token_id}')
                    return []
