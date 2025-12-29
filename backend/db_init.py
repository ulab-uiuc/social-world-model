import json
import os

from pymongo import ASCENDING, DESCENDING, MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['electionDB']


def setup_collections():
    if 'cards' not in db.list_collection_names():
        db.create_collection('cards')
        print("Created 'cards' collection")

    if 'vote_history' not in db.list_collection_names():
        db.create_collection('vote_history')
        print("Created 'vote_history' collection")

    if 'card_news_relations' not in db.list_collection_names():
        db.create_collection('card_news_relations')
        print("Created 'card_news_relations' collection")

    if 'estimated_vote_history' not in db.list_collection_names():
        db.create_collection('estimated_vote_history')
        print("Created 'estimated_vote_history' collection")

    if 'news' not in db.list_collection_names():
        db.create_collection('news')
        print("Created 'news' collection")

    if 'option_reasons' not in db.list_collection_names():
        db.create_collection('option_reasons')
        print("Created 'option_reasons' collection")

    if 'counters' not in db.list_collection_names():
        db.create_collection('counters')
        db.counters.insert_one({'_id': 'news_id', 'seq': 0})
        print("Created 'counters' collection and initialized news_id counter")


def setup_indexes():
    print("Cleaning invalid 'cards' documents before creating indexes...")
    db.cards.delete_many({'card_id': None})
    db.cards.delete_many({'card_id': {'$exists': False}})

    db.cards.create_index([('card_id', ASCENDING)], unique=True)
    db.cards.create_index([('tags', ASCENDING)])
    print("Created indexes for 'cards' collection")

    db.vote_history.create_index([('card_id', ASCENDING), ('timestamp', ASCENDING)])
    print("Created indexes for 'vote_history' collection")

    db.card_news_relations.create_index([('card_id', ASCENDING)], unique=True)
    db.card_news_relations.create_index([('news_ids', ASCENDING)])
    print("Created indexes for 'card_news_relations' collection")

    db.news.create_index([('news_id', ASCENDING)], unique=True)
    db.news.create_index([('tags', ASCENDING)])
    db.news.create_index([('timestamp', DESCENDING)])
    print("Created indexes for 'news' collection")

    db.option_reasons.create_index(
        [('card_id', ASCENDING), ('option', ASCENDING)], unique=True
    )
    print("Created indexes for 'option_reasons' collection")


def load_initial_data():
    cards_file = os.path.join(os.path.dirname(__file__), 'cards.json')
    if os.path.exists(cards_file):
        with open(cards_file, 'r') as f:
            cards_data = json.load(f)
            if cards_data and db.cards.count_documents({}) == 0:
                db.cards.insert_many(cards_data)
                print(f'Loaded {len(cards_data)} cards from cards.json')

    history_file = os.path.join(os.path.dirname(__file__), 'vote_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history_data = json.load(f)
            if history_data and db.vote_history.count_documents({}) == 0:
                db.vote_history.insert_many(history_data)
                print(
                    f'Loaded {len(history_data)} history entries from vote_history.json'
                )


def main():
    print('Initializing ElectionDB database...')

    setup_collections()

    setup_indexes()

    load_initial_data()

    print('Database initialization complete.')


if __name__ == '__main__':
    main()
