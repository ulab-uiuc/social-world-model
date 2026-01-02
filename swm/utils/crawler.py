import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import parse_qs, quote_plus, urlencode, urlparse

import httpx
import jsonlines
import requests
import serpapi
from bs4 import BeautifulSoup
from scholarly import scholarly
from tqdm import tqdm

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
            f'https://gamma-api.polymarket.com/events?offset={offset}&limit=100&active=true&closed=false'
        )
        if response.status_code == 200:
            return response.json()
        raise Exception(f'Failed to fetch events: HTTP {response.status_code}')


class PolyMarketHistoryCrawler(PolyMarketCrawler):
    def __init__(self, input_file: str, output_file: str):
        super().__init__(output_file, cache_size=5)
        self.input_file = input_file
        self.processed_events = self._load_processed_events(output_file)

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
            start_ts = None
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
        """Fetch price history for a token using the CLOB prices-history API."""
        base_url = 'https://clob.polymarket.com/prices-history'

        for attempt in range(max_retries):
            try:
                # Always use interval=max, filter by start_ts locally
                url = f'{base_url}?market={token_id}&interval=max&fidelity={fidelity}'
                response = httpx.get(url, timeout=30)
                
                if response.status_code == 200:
                    price_data = response.json()
                    history = price_data.get('history', [])
                    
                    # Debug: print first attempt's response
                    if attempt == 0 and not history:
                        print(f'DEBUG: Empty history for token {token_id[:20]}...')
                        print(f'DEBUG: Response: {price_data}')
                    
                    # Filter by start_ts if provided
                    if start_ts is not None and history:
                        history = [p for p in history if p.get('t', 0) >= start_ts]
                    
                    return history
                else:
                    raise Exception(f'HTTP {response.status_code}: {response.text}')

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
    """Crawler for Kalshi prediction market data."""

    def __init__(
        self,
        output_file: str,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        cache_size: int = 50,
    ):
        self.output_file = output_file
        self.cache_size = cache_size
        self.event_buffer = []
        self.base_url = 'https://api.elections.kalshi.com/trade-api/v2'
        self.api_key_id = api_key_id or os.getenv('KALSHI_API_KEY_ID')
        self.private_key_path = private_key_path
        self.client = None
        
        if HAS_KALSHI_SDK and self.api_key_id and self.private_key_path:
            self._init_sdk_client()
        
        if not self.client:
            print('⚠️ Kalshi SDK client not initialized.')
    
    def _init_sdk_client(self):
        try:
            configuration = kalshi_python.Configuration()
            configuration.host = self.base_url
            
            with open(self.private_key_path, 'r') as f:
                private_key_content = f.read()
            
            configuration.api_key_id = self.api_key_id
            configuration.private_key_pem = private_key_content
            
            if hasattr(kalshi_python, 'KalshiClient'):
                self.client = kalshi_python.KalshiClient(configuration)
            elif hasattr(kalshi_python, 'ApiInstance'):
                api_client = kalshi_python.ApiClient(configuration)
                self.client = kalshi_python.ApiInstance(api_client)
            else:
                from kalshi_python.api_instance import ApiInstance
                api_client = kalshi_python.ApiClient(configuration)
                self.client = ApiInstance(api_client)
            
            print("✅ Kalshi SDK client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize SDK client: {e}")
            self.client = None

    def _write_buffer(self) -> None:
        if not self.event_buffer:
            return
        mode = 'a' if os.path.exists(self.output_file) else 'w'
        with jsonlines.open(self.output_file, mode=mode) as writer:
            writer.write_all(self.event_buffer)
        self.event_buffer = []

    def _load_processed_markets(self) -> Set[str]:
        processed_tickers = set()
        if os.path.exists(self.output_file):
            try:
                with jsonlines.open(self.output_file, 'r') as reader:
                    for market in reader:
                        ticker = market.get('ticker') or market.get('market_id')
                        if ticker:
                            processed_tickers.add(ticker)
            except Exception as e:
                print(f'Error loading processed markets: {e}')
        return processed_tickers

    def get_markets(self, limit: int = 100, series_ticker: Optional[str] = None) -> Dict:
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?limit={limit}&series_ticker={series_ticker}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"HTTP get_markets error: {e}")
            return {'markets': [], 'cursor': None}

    def _format_market_for_output(self, market: Dict) -> Dict:
        start_ts = None
        end_ts = None
        
        if 'open_time' in market:
            try:
                dt = datetime.fromisoformat(market['open_time'].replace('Z', '+00:00'))
                start_ts = int(dt.timestamp())
            except:
                pass
        
        if 'close_time' in market:
            try:
                dt = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
                end_ts = int(dt.timestamp())
            except:
                pass
        
        outcome = None
        if market.get('status') in ('settled', 'finalized'):
            outcome = market.get('result')
        
        return {
            'event_id': market.get('event_ticker', ''),
            'market_id': market.get('ticker', ''),
            'question': market.get('title', ''),
            'yes_sub_title': market.get('yes_sub_title', ''),
            'no_sub_title': market.get('no_sub_title', ''),
            'time_series': market.get('time_series', []),
            'outcome': outcome,
            'start_ts': start_ts,
            'end_ts': end_ts,
        }

    def get_market_price(self, series_ticker: str, market_ticker: str) -> List[Dict]:
        try:
            path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
            now = datetime.now()
            start_ts = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()) - 86400 * 600
            end_ts = start_ts + 86400 * 600
            
            query_params = {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": 1440,
            }
            if self.client:
                query_string = urlencode(query_params, doseq=True)
                full_url = f"{self.base_url}{path}?{query_string}"
                response = self.client.api_client.call_api("GET", full_url)
                if hasattr(response, 'read'):
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get('candlesticks', None)
            return []
        except Exception as e:
            print(f"Error fetching price for {market_ticker}: {e}")
            return []

    def _process_price_to_ts(self, history: List[Dict]) -> List[Dict]:
        if not history or len(history) < 2:
            return []
        
        price_data = []
        for entry in history:
            timestamp = entry.get('end_period_ts')
            raw_price = entry.get('price', {}).get('mean_dollars', None)
            if raw_price is None:
                continue
            price_data.append({'t': timestamp, 'p': float(raw_price)})
        
        return price_data if len(price_data) >= 2 else []

    def collect_markets(
        self,
        max_markets: Optional[int] = None,
        series_ticker: Optional[str] = None,
    ) -> None:
        processed_tickers = self._load_processed_markets()
        total_collected = 0
        total_skipped = 0
        
        if not self.client:
            print("⚠️ SDK client not initialized.")
            return

        print(f'Starting collection (already processed: {len(processed_tickers)})')
        
        try:
            if series_ticker:
                series_tickers = [series_ticker]
            else:
                series_resp = self.client.get_series()
                series_tickers = [s.ticker for s in series_resp.series]
                print(f"Found {len(series_tickers)} series")
        except Exception as e:
            print(f"Error fetching series: {e}")
            return

        for s_ticker in tqdm(series_tickers, desc="Processing"):
            if max_markets and total_collected >= max_markets:
                break

            try:
                result = self.get_markets(limit=1000, series_ticker=s_ticker)
                markets = result.get('markets', [])
                
                for market in markets:
                    ticker = market.get('ticker')
                    
                    if ticker in processed_tickers:
                        total_skipped += 1
                        continue
                    
                    history = self.get_market_price(s_ticker, ticker)
                    time_series = self._process_price_to_ts(history) if history else []
                    
                    if len(time_series) == 0:
                        continue
                    
                    
                    market['time_series'] = time_series
                    formatted_market = self._format_market_for_output(market)
                    self.event_buffer.append(formatted_market)
                    processed_tickers.add(ticker)
                    total_collected += 1
                    
                    if len(self.event_buffer) >= self.cache_size:
                        self._write_buffer()
                        print(f'Collected {total_collected} (skipped {total_skipped})')
                    
                    if max_markets and total_collected >= max_markets:
                        break

                time.sleep(0.2)
            except Exception as e:
                print(f"Error processing {s_ticker}: {e}")
                continue
        
        self._write_buffer()
        print(f'\n✅ Collected {total_collected} markets, skipped {total_skipped}')


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


class GoogleNewsCrawler:
    """Crawler for Google News search results."""

    def __init__(self, min_delay: float = 2.0, max_delay: float = 6.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    def _random_delay(self):
        import random
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def _normalize_date(self, s: str) -> tuple[str, datetime]:
        s = s.strip()
        now = datetime.now()
        if "-" in s:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y"), dt
        if "/" in s:
            try:
                dt = datetime.strptime(s, "%m/%d/%Y")
                return s, dt
            except ValueError:
                pass
        return now.strftime("%m/%d/%Y"), now

    def _parse_relative_date(self, text: str, ref: datetime) -> float:
        t = text.strip().lower()
        
        # English: "5 days ago", "1 hour ago"
        m = re.match(r"^\s*(\d+)\s+(second|minute|hour|day)s?\s+ago\s*$", t)
        if m:
            num, unit = int(m.group(1)), m.group(2)
            delta = {
                "second": timedelta(seconds=num),
                "minute": timedelta(minutes=num),
                "hour": timedelta(hours=num),
                "day": timedelta(days=num),
            }[unit]
            return (ref - delta).timestamp()
        
        # Absolute date formats
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text.strip(), fmt).timestamp()
            except ValueError:
                continue
        
        return ref.timestamp()

    def _clean_google_href(self, href: str) -> str:
        if href.startswith("/url?"):
            qs = parse_qs(urlparse(href).query)
            if "q" in qs and qs["q"]:
                return qs["q"][0]
        return href

    def _find_snippet(self, card, title_text: str, source_text: str, date_text: str) -> str:
        for div in card.find_all(["div", "span"]):
            text = div.get_text(strip=True)
            if not text or len(text) < 20:
                continue
            if text in [title_text, source_text, date_text]:
                continue
            if title_text in text and len(text) < len(title_text) + 50:
                continue
            return text
        return ""

    def _extract_content_from_url(self, url: str, max_chars: int = 10000) -> str:
        """Extract article content from URL."""
        try:
            time.sleep(0.5)
            resp = self.session.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                return ""
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            # Try common article content selectors
            selectors = [
                "article p",
                "[itemprop='articleBody'] p",
                ".article-content p",
                ".article-body p",
                ".post-content p",
                ".entry-content p",
                ".story-body p",
                "main p",
            ]
            
            for selector in selectors:
                paragraphs = soup.select(selector)
                if paragraphs:
                    texts = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if len(text) > 30:
                            texts.append(text)
                    if texts:
                        content = " ".join(texts)
                        return content[:max_chars] if len(content) > max_chars else content
            
            # Fallback: get all paragraphs
            paragraphs = soup.find_all("p")
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
            if texts:
                content = " ".join(texts[:5])
                return content[:max_chars] if len(content) > max_chars else content
            
            return ""
        except Exception:
            return ""

    def fetch(
        self,
        query: str,
        start_date: str,
        end_date: str,
        max_pages: int = 10,
        fetch_full_content: bool = False,
    ) -> List[Dict[str, Any]]:
        start_fmt, _ = self._normalize_date(start_date)
        end_fmt, ref_date = self._normalize_date(end_date)
        
        results: List[Dict[str, Any]] = []
        
        for page in range(max_pages):
            encoded_query = quote_plus(query)
            url = (
                f"https://www.google.com/search?q={encoded_query}"
                f"&tbs=cdr:1,cd_min:{start_fmt},cd_max:{end_fmt}"
                f"&tbm=nws&start={page * 10}"
            )
            
            try:
                self._random_delay()
                resp = self.session.get(url, headers=self.headers, timeout=15)
                soup = BeautifulSoup(resp.text, "html.parser")
            except Exception as e:
                print(f"Request failed: {e}")
                break
            
            cards = soup.select("div.SoaBEf")
            if not cards:
                break
            
            for el in cards:
                try:
                    a = el.find("a")
                    if not a or "href" not in a.attrs:
                        continue
                    
                    link = self._clean_google_href(a["href"])
                    title_el = el.select_one("div.MBeuO")
                    date_el = el.select_one(".LfVVr")
                    source_el = el.select_one(".NUnG9d span")
                    
                    if not (title_el and date_el and source_el):
                        continue
                    
                    title = title_el.get_text(strip=True)
                    source = source_el.get_text(strip=True)
                    date_text = date_el.get_text(strip=True)
                    
                    snippet = self._find_snippet(el, title, source, date_text)
                    ts = self._parse_relative_date(date_text, ref_date)
                    
                    result = {
                        "link": link,
                        "title": title,
                        "snippet": snippet,
                        "date": ts,
                        "source": source,
                    }
                    
                    # Fetch full content if requested
                    if fetch_full_content:
                        content = self._extract_content_from_url(link)
                        if content:
                            result["content"] = content
                    
                    results.append(result)
                except Exception:
                    continue
            
            if not soup.find("a", id="pnnext"):
                break
        
        return results

    def crawl(
        self,
        query: str,
        start_date: str,
        end_date: str,
        output_file: str,
        max_pages: int = 10,
        fetch_full_content: bool = False,
    ) -> None:
        print(f"Fetching news for '{query}' from {start_date} to {end_date}")
        if fetch_full_content:
            print("⚠️ Fetching full content (slower)")
        
        results = self.fetch(query, start_date, end_date, max_pages, fetch_full_content)
        
        if not results:
            print("No results found")
            return
        
        results = sorted(results, key=lambda x: x["date"], reverse=True)
        
        mode = 'a' if os.path.exists(output_file) else 'w'
        with jsonlines.open(output_file, mode=mode) as writer:
            writer.write_all(results)
        
        print(f"✅ Saved {len(results)} articles to {output_file}")
