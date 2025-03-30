import json
import logging
import time
from datetime import datetime
from typing import Dict, List

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('polymarket_updater.log'), logging.StreamHandler()],
)
logger = logging.getLogger('polymarket_updater')

client = MongoClient('mongodb://localhost:27017/')
db = client['electionDB']
cards_collection = db['cards']
history_collection = db['vote_history']


class PolymarketUpdater:
    def __init__(self, start_offset: int = 5000, step: int = 100):
        self.start_offset = start_offset
        self.step = step

    def get_events_from_offset(self, offset: int) -> List[Dict]:
        try:
            response = httpx.get(
                f'https://gamma-api.polymarket.com/events?offset={offset}&limit=100',
                timeout=30,
            )
            if response.status_code != 200:
                logger.error(f'Failed to fetch events: HTTP {response.status_code}')
                return []

            events = response.json()
            logger.info(f'Number of raw events: {len(events)}')
            if events:
                logger.info(
                    f'Example of raw event data: {json.dumps(events[0] if events else {}, indent=2)[:500]}...'
                )

            return events
        except Exception as e:
            logger.error(f'Error fetching events from offset {offset}: {e}')
            return []

    def is_market_active(self, market: Dict) -> bool:
        return (
            market.get('closed') is False
            and market.get('acceptingOrders') is True
            and market.get('umaResolutionStatus') != 'resolved'
        )

    def is_event_active(self, event: Dict) -> bool:
        return event.get('closed') is False and any(
            self.is_market_active(m) for m in event.get('markets', [])
        )

    def convert_event_to_card(self, event: Dict) -> Dict:
        try:
            active_markets = [
                m for m in event.get('markets', []) if self.is_market_active(m)
            ]

            if not active_markets:
                logger.warning(
                    f"No active markets for event {event.get('id', 'unknown')}"
                )
                return {}

            market = active_markets[0]

            if not market.get('outcomes') or not market.get('outcomePrices'):
                logger.warning(
                    f"Missing outcomes or outcomePrices for market {market.get('id', 'unknown')}"
                )
                return {}

            try:
                outcome_options = json.loads(market['outcomes'])
                outcome_prices = json.loads(market['outcomePrices'])
            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON parsing error: {e}, outcomes={market.get('outcomes')}, prices={market.get('outcomePrices')}"
                )
                return {}

            total_price = sum(float(price) for price in outcome_prices)

            options = []
            for i, option in enumerate(outcome_options):
                if i < len(outcome_prices):
                    percentage = float(outcome_prices[i])
                    if total_price > 0:
                        percentage = (percentage / total_price) * 100
                    options.append({'option': option, 'bets': round(percentage, 2)})

            tags = [tag['label'] for tag in event.get('tags', [])]
            if not tags:
                tags = ['other']

            card = {
                'card_id': str(event['id']),
                'question': market['question'],
                'tags': tags,
                'options': options,
                'polymarket_id': event['id'],
                'market_id': market['id'],
                'last_updated': datetime.utcnow().isoformat(),
            }

            logger.info(f'Converted card data: {json.dumps(card, indent=2)}')
            return card

        except Exception as e:
            logger.error(
                f"Error converting event to card: {e}, event_id={event.get('id', 'unknown')}"
            )
            return {}

    def get_price_history(self, market_id: str, card_id: str) -> List[Dict]:
        try:
            url = f'https://gamma-api.polymarket.com/markets/{market_id}'
            response = httpx.get(url, timeout=30)
            if response.status_code != 200:
                logger.error(f'Failed to fetch market: HTTP {response.status_code}')
                return []

            market_data = response.json()
            logger.info(
                f"Retrieved market data: {market_data.get('id')}, question: {market_data.get('question', '')[:50]}..."
            )

            if not market_data.get('clobTokenIds') or not market_data.get('outcomes'):
                logger.warning(f'Market data missing required fields: {market_id}')
                return []

            try:
                token_ids = json.loads(market_data.get('clobTokenIds', '[]'))
                outcomes = json.loads(market_data.get('outcomes', '[]'))
            except json.JSONDecodeError as e:
                logger.error(f'JSON parsing error: {e}')
                return []

            result = []
            all_timestamps = set()
            option_prices = {}

            for i, token_id in enumerate(token_ids):
                if i >= len(outcomes):
                    continue

                option = outcomes[i]
                price_history = self.get_history_from_token_id(token_id)
                logger.info(
                    f"Retrieved {len(price_history)} history entries for option '{option}'"
                )

                for point in price_history:
                    timestamp = point['t']
                    all_timestamps.add(timestamp)

                    if timestamp not in option_prices:
                        option_prices[timestamp] = {}

                    option_prices[timestamp][option] = point['p']

            logger.info(f'Retrieved data for {len(all_timestamps)} total timestamps')

            for timestamp in sorted(all_timestamps):
                prices = option_prices.get(timestamp, {})
                if not prices:
                    continue

                total_price = sum(prices.values())

                votes = {}
                for option, price in prices.items():
                    percentage = (price / total_price * 100) if total_price > 0 else 0
                    votes[option] = round(percentage, 2)

                history_entry = {
                    'card_id': card_id,
                    'timestamp': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z',
                    'votes': votes,
                }
                result.append(history_entry)

            return result

        except Exception as e:
            logger.error(f'Error getting price history for market {market_id}: {e}')
            return []

    def get_history_from_token_id(
        self, token_id: str, fidelity: int = 60
    ) -> List[Dict]:
        try:
            base_url = 'https://clob.polymarket.com/prices-history'
            params = {'market': token_id, 'fidelity': fidelity, 'startTs': 1}

            response = httpx.get(base_url, params=params, timeout=30)
            if response.status_code != 200:
                logger.error(f'Failed to fetch history: HTTP {response.status_code}')
                return []

            data = response.json()
            return data.get('history', [])

        except Exception as e:
            logger.error(f'Error fetching history for token {token_id}: {e}')
            return []

    def update_active_events(self):
        logger.info('Starting active events update...')
        offset = self.start_offset
        events_processed = 0
        events_added = 0
        events_removed = 0

        try:
            while True:
                events = self.get_events_from_offset(offset)
                logger.info(f'Fetched {len(events)} events from API at offset {offset}')

                if not events:
                    break

                for event in events:
                    events_processed += 1

                    if self.is_event_active(event):
                        card = self.convert_event_to_card(event)
                        if not card:
                            continue

                        result = cards_collection.update_one(
                            {'card_id': card['card_id']}, {'$set': card}, upsert=True
                        )

                        if result.upserted_id:
                            events_added += 1
                            logger.info(f"Added new card: {card['question']}")
                    else:
                        result = cards_collection.delete_one(
                            {'card_id': str(event['id'])}
                        )
                        if result.deleted_count > 0:
                            events_removed += 1
                            logger.info(
                                f"Removed inactive card: {event.get('question', event['id'])}"
                            )

                offset += self.step

                time.sleep(1)

            logger.info(
                f'Events update completed: processed {events_processed}, added {events_added}, removed {events_removed}'
            )

        except Exception as e:
            logger.error(f'Error in update_active_events: {e}')

    def update_price_history(self):
        logger.info('Starting price history update...')
        updates_count = 0

        try:
            active_cards = list(cards_collection.find({}))
            logger.info(f'Found {len(active_cards)} active cards')

            for card in active_cards:
                market_id = card.get('market_id')
                card_id = card.get('card_id')

                if not market_id or not card_id:
                    logger.warning(
                        f"Missing market_id or card_id for card: {card.get('question', 'unknown')}"
                    )
                    continue

                history_entries = self.get_price_history(market_id, card_id)

                if not history_entries:
                    logger.warning(
                        f"No history entries for card: {card.get('question', market_id)}"
                    )
                    continue

                new_entries_count = 0
                for entry in history_entries:
                    existing = history_collection.find_one(
                        {'card_id': card_id, 'timestamp': entry['timestamp']}
                    )

                    if not existing:
                        history_collection.insert_one(entry)
                        new_entries_count += 1
                        updates_count += 1

                logger.info(
                    f"Added {new_entries_count} new history entries for card '{card.get('question', '')[:30]}...'"
                )

                cards_collection.update_one(
                    {'card_id': card_id},
                    {'$set': {'last_updated': datetime.utcnow().isoformat()}},
                )

                time.sleep(0.5)

            logger.info(
                f'Price history update completed: {updates_count} new entries added'
            )

        except Exception as e:
            logger.error(f'Error in update_price_history: {e}')


def main():
    updater = PolymarketUpdater()

    scheduler = BackgroundScheduler()

    scheduler.add_job(updater.update_active_events, 'interval', hours=1)

    scheduler.add_job(updater.update_price_history, 'interval', minutes=30)

    logger.info('Running initial data update...')
    updater.update_active_events()
    updater.update_price_history()

    logger.info('Starting scheduler...')
    scheduler.start()

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info('Shutting down scheduler...')
        scheduler.shutdown()


if __name__ == '__main__':
    main()
