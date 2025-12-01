import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union

import httpx
import jsonlines
import requests
import serpapi
from py_clob_client.client import ClobClient
from scholarly import scholarly
from tqdm import tqdm
from urllib.parse import urlencode

try:
    import kalshi_python
    from kalshi_python.rest import ApiException
    HAS_KALSHI_SDK = True
except ImportError:
    HAS_KALSHI_SDK = False


class PolyMarketCrawler:
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


class PolyMarketEventCrawler(PolyMarketCrawler):
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


class PolyMarketHistoryCrawler(PolyMarketCrawler):
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
                    history = future.result()
                    if history:
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


class GoogleScholarCrawler:
    def __init__(self, input_file: str, output_file: str, api_key: str):
        self.input_file = input_file
        self.output_file = output_file
        self.api_key = api_key

    def retry_on_error(self, func, *args, retries=3, delay=2, **kwargs):
        """Helper method to retry a function call on error"""
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f'Attempt {attempt + 1} failed: {e}')
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def get_scholar_id_from_citedby(self, data):
        """Extract Scholar ID from the 'citedby_url' field"""
        citedby_url = data.get('citedby_url', '')
        if 'cites=' in citedby_url:
            parts = citedby_url.split('cites=')
            if len(parts) > 1:
                id_and_rest = parts[1]
                return id_and_rest.split('&')[0]
        return None

    def get_citation_id(self, title):
        """Search for a publication and extract its citation ID"""

        def search_title():
            pubs = scholarly.search_pubs(title)
            for pub in pubs:
                if title.lower() == pub['bib']['title'].lower():
                    scholar_id = self.get_scholar_id_from_citedby(pub)
                    pub_year = int(pub['bib']['pub_year'])
                    return pub, scholar_id, pub_year
            return None, None, None

        return self.retry_on_error(search_title)

    def fetch_citation_count(self, citation_id, year_from=None, year_to=None):
        """Fetch citation count from Google Scholar"""

        def fetch_count():
            params = {
                'engine': 'google_scholar',
                'cites': citation_id,
                'api_key': self.api_key,
            }
            if year_from:
                params['as_ylo'] = year_from
            if year_to:
                params['as_yhi'] = year_to

            search = serpapi.search(params)
            results = search.get_dict()
            return results.get('search_information', {}).get('total_results')

        return self.retry_on_error(fetch_count)

    def get_paper_data(self, title):
        """Fetch detailed paper data including citation count"""
        try:
            pub, citation_id, pub_year = self.get_citation_id(title)
            if citation_id:
                total_citations = self.fetch_citation_count(citation_id)
                citations = {
                    year: self.fetch_citation_count(
                        citation_id, year_from=year, year_to=year
                    )
                    for year in range(pub_year, 2025)
                }
                return {
                    'title': title,
                    'citation_id': citation_id,
                    'total_citations': total_citations,
                    'citations': citations,
                    'raw_data': pub,
                }
            else:
                print('Failed to retrieve Citation ID.')
                return None
        except Exception as e:
            print(f'An error occurred: {e}')
            return None

    def crawl(self):
        """Main method to crawl and collect paper data"""
        with open(self.input_file, 'r') as f:
            papers = json.load(f)

        titles = [data['paper_data']['title'] for data in papers.values()]

        all_paper_data = {}
        for title in titles:
            paper_data = self.get_paper_data(title)
            if paper_data:
                print(paper_data)
                all_paper_data[paper_data['citation_id']] = paper_data
                with open(self.output_file, 'w') as f:
                    json.dump(all_paper_data, f)
            else:
                print('Failed to retrieve paper data.')


class KalshiCrawler:
    """
    Crawler for Kalshi prediction market data.
    Supports fetching events, markets, and historical data from Kalshi API.
    """

    def __init__(
        self,
        output_file: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        cache_size: int = 50,
    ):
        """
        Initialize Kalshi crawler.
        
        Args:
            output_file: Path to output JSONL file
            email: Kalshi account email (for email+password auth)
            password: Kalshi account password (for email+password auth)
            api_key: Kalshi API key (alternative to email+password)
            api_key_id: Kalshi API key ID (for SDK auth with private key)
            private_key_path: Path to private key file (most secure option)
            cache_size: Number of events to buffer before writing to disk
            
        Note:
            Authentication priority (highest to lowest):
            1. SDK (api_key_id + private_key_path) - best for history
            2. private_key_path - most secure
            3. api_key - secure and convenient
            4. email+password - basic auth
            5. no auth - public endpoints only
        """
        self.output_file = output_file
        self.cache_size = cache_size
        self.event_buffer = []
        self.base_url = 'https://api.elections.kalshi.com/trade-api/v2'
        self.demo_base_url = 'https://demo-api.kalshi.co/trade-api/v2'
        self.token = None
        self.api_key = api_key
        self.api_key_id = api_key_id or os.getenv('KALSHI_API_KEY_ID')
        self.email = email
        self.password = password
        self.private_key = None
        self.private_key_path = private_key_path
        self.client = None
        
        # Try to initialize SDK client if possible
        if HAS_KALSHI_SDK and self.api_key_id and self.private_key_path:
            self._init_sdk_client()
        
        if not self.client:
            print('⚠️ Kalshi SDK client not initialized. Please ensure kalshi-python is installed and credentials are provided.')
    
    def _init_sdk_client(self):
        """Initialize Kalshi SDK client."""
        try:
            print("Initializing Kalshi SDK client...")
            configuration = kalshi_python.Configuration()
            configuration.host = self.base_url
            
            # Read private key
            with open(self.private_key_path, 'r') as f:
                private_key_content = f.read()
            
            configuration.api_key_id = self.api_key_id
            configuration.private_key_pem = private_key_content
            
            # Initialize the Kalshi client with robust fallback
            if hasattr(kalshi_python, 'KalshiClient'):
                self.client = kalshi_python.KalshiClient(configuration)
            elif hasattr(kalshi_python, 'ApiInstance'):
                api_client = kalshi_python.ApiClient(configuration)
                self.client = kalshi_python.ApiInstance(api_client)
            else:
                try:
                    from kalshi_python.api_instance import ApiInstance
                    api_client = kalshi_python.ApiClient(configuration)
                    self.client = ApiInstance(api_client)
                except ImportError:
                    print("⚠️  ApiInstance not found, using MarketApi")
                    api_client = kalshi_python.ApiClient(configuration)
                    self.client = kalshi_python.MarketApi(api_client)
            
            print("✅ Kalshi SDK client initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize SDK client: {e}")
            self.client = None

    def _write_buffer(self) -> None:
        """Write buffered events to file in JSONL format."""
        if not self.event_buffer:
            return

        mode = 'a' if os.path.exists(self.output_file) else 'w'
        with jsonlines.open(self.output_file, mode=mode) as writer:
            writer.write_all(self.event_buffer)
        self.event_buffer = []

    def _load_processed_markets(self) -> Set[str]:
        """Load processed market tickers to avoid duplication."""
        processed_tickers = set()
        if os.path.exists(self.output_file):
            try:
                with jsonlines.open(self.output_file, 'r') as reader:
                    for market in reader:
                        if 'ticker' in market:
                            processed_tickers.add(market['ticker'])
                        elif 'market_ticker' in market:
                            processed_tickers.add(market['market_ticker'])
            except Exception as e:
                print(f'Error loading processed markets: {e}')
        return processed_tickers

    def get_events(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch events from Kalshi API using SDK.
        """
        if not self.client:
            return []

        try:
            kwargs = {'limit': limit}
            if status: kwargs['status'] = status
            if series_ticker: kwargs['series_ticker'] = series_ticker
            
            response = self.client.get_events(**kwargs)
            return response.to_dict().get('events', [])
        except Exception as e:
            print(f"SDK get_events error: {e}")
            return []

    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ) -> Dict:
        """
        Fetch markets from Kalshi API with pagination support using SDK.
        """
        if not self.client:
            return {'markets': [], 'cursor': None}

        try:
            kwargs = {'limit': limit}
            if cursor: kwargs['cursor'] = cursor
            if event_ticker: kwargs['event_ticker'] = event_ticker
            if series_ticker: kwargs['series_ticker'] = series_ticker
            if status: kwargs['status'] = status
            if min_close_ts: kwargs['min_close_ts'] = min_close_ts
            if max_close_ts: kwargs['max_close_ts'] = max_close_ts
            
            response = self.client.get_markets(**kwargs)
            return response.to_dict()
        except Exception as e:
            print(f"SDK get_markets error: {e}")
            return {'markets': [], 'cursor': None}

    def get_market_history(
        self,
        ticker: str,
        limit: int = 500,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch historical data for a specific market using SDK.
        """
        if not self.client:
            return []

        try:
            kwargs = {'ticker': ticker, 'limit': limit}
            if min_ts: kwargs['min_ts'] = min_ts
            if max_ts: kwargs['max_ts'] = max_ts
            
            response = self.client.get_market_history(**kwargs)
            return response.to_dict().get('history', [])
        except Exception as e:
            print(f"SDK get_market_history error for {ticker}: {e}")
            return []

    def _create_basic_time_series_from_market(
        self, market: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Create basic time series from current market data when history API is unavailable.
        Uses current prices and previous prices to create minimal time series.
        
        Args:
            market: Market dictionary with current price data
            
        Returns:
            Dictionary with basic time series data in PolyMarket-compatible format
        """
        # Get timestamps
        current_time = int(time.time())
        
        # Try to get open time
        open_time_str = market.get('open_time')
        if open_time_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
                open_time = int(dt.timestamp())
            except:
                open_time = current_time - 86400  # 1 day ago as fallback
        else:
            open_time = current_time - 86400
        
        # Create yes price series
        yes_price_data = []
        no_price_data = []
        
        # Add previous price point if available
        prev_yes_bid = market.get('previous_yes_bid')
        prev_yes_ask = market.get('previous_yes_ask')
        if prev_yes_bid is not None or prev_yes_ask is not None:
            prev_price = market.get('previous_price', 
                                   (prev_yes_bid + prev_yes_ask) / 2 if prev_yes_bid and prev_yes_ask else None)
            if prev_price is not None:
                yes_price_data.append({
                    't': open_time,  # PolyMarket format: 't' for timestamp
                    'p': prev_price / 100 if prev_price > 1 else prev_price,  # 'p' for price
                })
        
        # Add current yes price point
        yes_bid = market.get('yes_bid')
        yes_ask = market.get('yes_ask')
        last_price = market.get('last_price')
        if yes_bid is not None or yes_ask is not None or last_price is not None:
            current_price = last_price if last_price is not None else (
                (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else None
            )
            if current_price is not None:
                yes_price_data.append({
                    't': current_time,
                    'p': current_price / 100 if current_price > 1 else current_price,
                })
        
        # Create no price series
        prev_no_bid = market.get('previous_no_bid')
        prev_no_ask = market.get('previous_no_ask')
        if prev_no_bid is not None or prev_no_ask is not None:
            prev_no_price = (prev_no_bid + prev_no_ask) / 2 if prev_no_bid and prev_no_ask else None
            if prev_no_price is not None:
                no_price_data.append({
                    't': open_time,
                    'p': prev_no_price / 100 if prev_no_price > 1 else prev_no_price,
                })
        
        no_bid = market.get('no_bid')
        no_ask = market.get('no_ask')
        if no_bid is not None or no_ask is not None:
            current_no_price = (no_bid + no_ask) / 2 if no_bid and no_ask else None
            if current_no_price is not None:
                no_price_data.append({
                    't': current_time,
                    'p': current_no_price / 100 if current_no_price > 1 else current_no_price,
                })
        
        # Return in PolyMarket format: {"Yes": [...], "No": [...]}
        return {
            'Yes': yes_price_data,
            'No': no_price_data,
        }
    def _format_market_for_output(self, market: Dict) -> Dict:
        """
        Format Kalshi market data to be compatible with PolyMarket format.
        Focus on core fields: question and time_series format.
        
        Args:
            market: Raw Kalshi market data
            
        Returns:
            Formatted market data with PolyMarket-compatible structure
        """
        # Extract timestamps
        start_ts = None
        end_ts = None
        
        if 'open_time' in market:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(market['open_time'].replace('Z', '+00:00'))
                start_ts = dt.timestamp()
            except:
                pass
        
        if 'close_time' in market:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
                end_ts = dt.timestamp()
            except:
                pass
        
        # Determine outcome (if market is settled)
        outcome = None
        if market.get('status') == 'settled':
            result = market.get('result')
            if result == 'yes':
                outcome = 'Yes'
            elif result == 'no':
                outcome = 'No'
        
        # Extract tags from series ticker or category
        tags = []
        categories = []
        if 'series_ticker' in market:
            tags.append(market['series_ticker'])
            categories.append(market['series_ticker'])
        if 'category' in market:
            if market['category'] not in tags:
                tags.append(market['category'])
            if market['category'] not in categories:
                categories.append(market['category'])
        
        # Core PolyMarket-compatible output (only required fields)
        formatted = {
            # Core fields (always present)
            'market_id': market.get('ticker', ''),
            'question': market.get('title', ''),  # Most important field
            'daily_time_series': market.get('daily_time_series', {'Yes': [], 'No': []}),  # Most important field
            'hourly_time_series': market.get('hourly_time_series', {'Yes': [], 'No': []}),  # Most important field
            
            # Optional fields (only if available)
            'event_id': market.get('event_ticker', market.get('ticker', '')),
            'outcome': outcome,
            'start_ts': start_ts,
            'end_ts': end_ts,
        }
        
        # Add optional fields only if they have meaningful values
        if market.get('subtitle'):
            formatted['description'] = market['subtitle']
        
        if market.get('volume') is not None:
            formatted['volumn'] = market['volume']  # Note: PolyMarket uses 'volumn' (typo preserved)
        
        if tags:
            formatted['tags'] = tags
            formatted['categories'] = categories
        
        if market.get('ranged_group_name'):
            formatted['resolution_source'] = market['ranged_group_name']
        
        # Keep raw Kalshi data for reference
        formatted['kalshi_raw'] = {
            'ticker': market.get('ticker'),
            'status': market.get('status'),
            'yes_bid': market.get('yes_bid'),
            'yes_ask': market.get('yes_ask'),
            'no_bid': market.get('no_bid'),
            'no_ask': market.get('no_ask'),
            'last_price': market.get('last_price'),
            'open_interest': market.get('open_interest'),
        }
        
        return formatted

    def _process_history_to_time_series(self, history: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Convert Kalshi history format to PolyMarket-compatible time series data.
        
        Args:
            history: Raw history data from Kalshi API
            
        Returns:
            Dictionary with 'Yes' and 'No' price series in PolyMarket format
        """
        if not history:
            return {'Yes': [], 'No': []}
        
        yes_price_data = []
        no_price_data = []
        
        for entry in history:
            timestamp = entry.get('ts', entry.get('timestamp', 0))
            
            # Yes price series - PolyMarket format: {"t": timestamp, "p": price}
            if 'yes_price' in entry or 'yes_ask' in entry:
                yes_price = entry.get('yes_price', entry.get('yes_ask'))
                if yes_price is not None:
                    yes_price_data.append({
                        't': timestamp,
                        'p': yes_price / 100 if yes_price > 1 else yes_price,  # Convert cents to dollars
                    })
            
            # No price series
            if 'no_price' in entry or 'no_ask' in entry:
                no_price = entry.get('no_price', entry.get('no_ask'))
                if no_price is not None:
                    no_price_data.append({
                        't': timestamp,
                        'p': no_price / 100 if no_price > 1 else no_price,
                    })
        
        return {
            'Yes': yes_price_data,
            'No': no_price_data,
        }


    def get_market_orderbook(self, ticker: str) -> Dict:
        """
        Fetch current orderbook for a market using SDK.
        """
        if not self.client:
            return {}

        try:
            response = self.client.get_market_orderbook(ticker)
            return response.to_dict().get('orderbook', {})
        except Exception as e:
            print(f"SDK get_market_orderbook error for {ticker}: {e}")
            return {}

    def get_market_forecast_history(self, series_ticker: str, ticker: str) -> List[Dict]:
        """
        Fetch forecast history for an event using the V2 endpoint via SDK.
        URL: /series/{series_ticker}/events/{ticker}/forecast_percentile_history
        """
        try:
            path = f"/series/{series_ticker}/events/{ticker}/forecast_percentile_history"
            
            now = datetime.now()
            start_ts = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())  - 86400 * 10
            end_ts = start_ts + 86400 * 10
            
            
            query_params = {
                "percentiles": "5000",  # 50%, 75%, 90%
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": 1440,
            }
            if self.client:
                # doseq=True 让 list 变成多个同名参数
                query_string = urlencode(query_params, doseq=True)
                full_path = f"{path}?{query_string}"
                full_url = f"{self.base_url}{full_path}"
                #full_url = "https://api.elections.kalshi.com/trade-api/v2/series/TIPPINGPOINT/events/TIPPINGPOINT-24/forecast_percentile_history?percentiles=5000&start_ts=1728306000&end_ts=1736190240&&period_interval=60"
                response = self.client.api_client.call_api("GET", full_url)
                # Read response data
                if hasattr(response, 'read'):
                    data = json.loads(response.read().decode('utf-8'))
                    print(data)
                    return data.get('forecast_history', None)
                return []
            return []
                
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            return []

    def _process_forecast_history_to_time_series(self, history: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Convert V1 forecast history to PolyMarket-compatible time series data.
        """
        if not history:
            return {'Yes': [], 'No': []}
            
        yes_price_data = []
        
        for entry in history:
            # V1 format: end_period_ts, numerical_forecast (0-100)
            timestamp = entry.get('end_period_ts')
            forecast = float(entry.get('percentile_points')[0]['formatted_forecast'])
            
            if timestamp is not None and forecast is not None:
                yes_price_data.append({
                    't': timestamp,
                    'p': forecast / 100.0  # Convert 0-100 to 0-1
                })
        
        return {
            'Yes': yes_price_data,
            'No': []
        }

    def collect_markets(
        self,
        max_markets: Optional[int] = None,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        include_history: bool = True,
        include_orderbook: bool = False,
        history_limit: int = 500,
    ) -> None:
        """
        Collect markets from Kalshi using Series-based traversal.
        Follows the pattern: Get Series -> Get Markets -> Get Market Forecast History.
        """
        processed_tickers = self._load_processed_markets()
        total_collected = 0
        total_skipped = 0
        
        if not self.client:
            print("⚠️ SDK client not initialized. Cannot collect markets.")
            return

        print(f'Starting market collection via Series (already processed: {len(processed_tickers)})')
        
        # 1. Get Series List
        try:
            if series_ticker:
                series_tickers = [series_ticker]
            else:
                print("Fetching all series...")
                series_resp = self.client.get_series()
                series_tickers = [s.ticker for s in series_resp.series]
                print(f"Found {len(series_tickers)} series")
        except Exception as e:
            print(f"Error fetching series list: {e}")
            return

        # 2. Iterate Series
        for s_ticker in tqdm(series_tickers, desc="Processing Series"):
            if max_markets and total_collected >= max_markets:
                print(f"Reached max_markets limit ({max_markets}). Stopping.")
                break

            try:
                # Get markets for this Series using reliable get_markets endpoint
                result = self.get_markets(
                    limit=100,
                    series_ticker=s_ticker,
                    status=status,
                )
                
                markets = result.get('markets', [])
                
                if not markets:
                    continue
                
                for market in markets:
                    ticker = market.get('ticker')
                    event_ticker = market.get('event_ticker')
                    
                    # Skip if already processed
                    if ticker in processed_tickers:
                        total_skipped += 1
                        continue
                    
                    # Enrich market data if requested
                    if include_history:
                        history = []
                        # Prefer event_ticker for event-level forecast history
                        
                        if event_ticker:
                            history = self.get_market_forecast_history(s_ticker, event_ticker)
                        
                        if history:
                            time_series = self._process_forecast_history_to_time_series(history)
                        else:
                            # Fallback: create basic time series from current market data
                            time_series = self._create_basic_time_series_from_market(market)
                        
                        market['daily_time_series'] = time_series
                        market['hourly_time_series'] = time_series
                    
                    if include_orderbook:
                        orderbook = self.get_market_orderbook(ticker)
                        if orderbook:
                            market['orderbook'] = orderbook
                            time.sleep(0.1)
                    
                    # Convert and save
                    formatted_market = self._format_market_for_output(market)
                    self.event_buffer.append(formatted_market)
                    processed_tickers.add(ticker)
                    total_collected += 1
                    
                    if len(self.event_buffer) >= self.cache_size:
                        self._write_buffer()
                        print(f'Collected {total_collected} markets (skipped {total_skipped})')
                    
                    if max_markets and total_collected >= max_markets:
                        print(f"Hit max_markets ({max_markets}) limit.")
                        break
                
                if max_markets and total_collected >= max_markets:
                    break
                
                time.sleep(0.2)  # Rate limiting between series
                    
            except Exception as e:
                print(f"Error processing series {s_ticker}: {e}")
                continue
        
        # Write remaining buffer
        self._write_buffer()
        
        print('\n=== Collection Summary ===')
        print(f'Total markets collected: {total_collected}')
        print(f'Total markets skipped: {total_skipped}')
        print(f'Output file: {self.output_file}')

    def collect_events_with_markets(
        self,
        status: Optional[str] = None,
        include_history: bool = True,  # Changed default to True
        history_limit: int = 500,
    ) -> None:
        """
        Collect events and their associated markets with time series data.
        
        Args:
            status: Filter by status ('open', 'closed', 'settled')
            include_history: Whether to fetch historical data for markets (default: True)
            history_limit: Maximum number of historical data points per market
        """
        events = self.get_events(limit=1000, status=status)
        print(f'Found {len(events)} events')
        if include_history:
            print('📈 Time series data will be included for all markets')
        
        for event in tqdm(events, desc='Processing events'):
            event_ticker = event.get('event_ticker')
            
            # Fetch markets for this event
            markets_data = self.get_markets(event_ticker=event_ticker, limit=100)
            event['markets'] = markets_data.get('markets', [])
            
            # Optionally fetch history for each market
            if include_history:
                for market in event['markets']:
                    ticker = market.get('ticker')
                    if ticker:
                        history = self.get_market_history(ticker, limit=history_limit)
                        if history:
                            # Process history to PolyMarket format
                            time_series = self._process_history_to_time_series(history)
                        else:
                            # Fallback: create basic time series from current data
                            time_series = self._create_basic_time_series_from_market(market)
                        
                        # Store in PolyMarket format
                        market['daily_time_series'] = time_series
                        market['hourly_time_series'] = time_series
                        
                        time.sleep(0.1)  # Rate limiting
            
            # Format each market in the event
            formatted_markets = []
            for market in event.get('markets', []):
                formatted_market = self._format_market_for_output(market)
                formatted_markets.append(formatted_market)
            
            # Store formatted markets
            for formatted_market in formatted_markets:
                self.event_buffer.append(formatted_market)
            
            if len(self.event_buffer) >= self.cache_size:
                self._write_buffer()
        
        self._write_buffer()
        print(f'\n✅ Collected {len(events)} events with markets')
        if include_history:
            print(f'✅ Time series data included for all markets')


class DailyNewsCrawler:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        output_file: str,
        api_token: str,
    ):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.output_file = output_file
        self.api_token = api_token
        self.base_url = 'https://api.thenewsapi.com/v1/news/headlines'

    def retry_on_error(self, func, *args, retries=3, delay=2, **kwargs):
        """Helper method to retry a function call on error"""
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f'Attempt {attempt + 1} failed: {e}')
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def fetch_news_for_date(self, date: datetime) -> Optional[Dict]:
        def fetch_articles():
            params = {
                'api_token': self.api_token,
                'language': 'en',
                'published_on': date.strftime('%Y-%m-%d'),
                'headlines_per_category': 10,
                'locale': 'us',
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()

        return self.retry_on_error(fetch_articles)

    def process_articles(self, data: Dict) -> List[Dict]:
        """Process and clean article data"""
        processed_articles = []

        # Handle nested category structure
        categories_data = data.get('data', {})
        if not isinstance(categories_data, dict):
            print(f'Unexpected data format: {type(categories_data)}')
            return []

        # Iterate through each category
        for category, articles in categories_data.items():
            if not isinstance(articles, list):
                print(
                    f'Unexpected articles format for category {category}: {type(articles)}'
                )
                continue

            for article in articles:
                if not isinstance(article, dict):
                    continue

                try:
                    processed_article = {
                        'uuid': article.get('uuid'),
                        'title': article.get('title'),
                        'url': article.get('url'),
                        'source': article.get('source'),
                        'published_at': article.get('published_at'),
                        'description': article.get('description'),
                        'snippet': article.get('snippet'),
                        'image_url': article.get('image_url'),
                        'language': article.get('language'),
                        'categories': article.get('categories', []),
                        'similar_articles': [],
                    }

                    # Process similar articles
                    similar = article.get('similar', [])
                    if isinstance(similar, list):
                        for similar_item in similar:
                            if isinstance(similar_item, dict):
                                similar_article = {
                                    'uuid': similar_item.get('uuid'),
                                    'title': similar_item.get('title'),
                                    'url': similar_item.get('url'),
                                    'source': similar_item.get('source'),
                                    'published_at': similar_item.get('published_at'),
                                    'categories': similar_item.get('categories', []),
                                }
                                processed_article['similar_articles'].append(
                                    similar_article
                                )

                    if all(
                        [
                            processed_article['title'],
                            processed_article['url'],
                            processed_article['published_at'],
                        ]
                    ):
                        processed_articles.append(processed_article)

                except Exception as e:
                    print(f'Error processing article: {e}')
                    continue

        return processed_articles

    def save_articles(self, articles: List[Dict], mode='a'):
        """Save articles in jsonlines format"""
        try:
            with open(self.output_file, mode, encoding='utf-8') as f:
                for article in articles:
                    json.dump(article, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f'Error saving articles: {e}')
            raise

    def crawl(self):
        """Main method to crawl and collect news data for date range"""
        try:
            current_date = self.start_date
            while current_date <= self.end_date:
                try:
                    print(f"Fetching news for {current_date.strftime('%Y-%m-%d')}...")
                    raw_data = self.fetch_news_for_date(current_date)

                    if not raw_data or 'data' not in raw_data:
                        print('No articles found or invalid API response')
                        current_date += timedelta(days=1)
                        continue

                    articles = self.process_articles(raw_data)
                    if articles:
                        self.save_articles(articles)
                        print(
                            f'Successfully processed {len(articles)} articles for {current_date.date()}'
                        )
                    else:
                        print(f'No valid articles found for {current_date.date()}')

                    current_date += timedelta(days=1)
                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    print(f'Error processing date {current_date.date()}: {e}')
                    current_date += timedelta(days=1)
                    continue

        except Exception as e:
            print(f'An error occurred during crawling: {e}')
            import traceback

            print(traceback.format_exc())
