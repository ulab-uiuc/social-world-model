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
