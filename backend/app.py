from datetime import datetime
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)


client: MongoClient[Any] = MongoClient('mongodb://localhost:27017/')
db = client['electionDB']
cards_collection = db['cards']
history_collection = db['vote_history']
relations_collection = db['card_news_relations']
estimated_history_collection = db['estimated_vote_history']

option_reasons_collection = db['option_reasons']


def record_hourly_data():
    try:
        cards = list(cards_collection.find({}, {'_id': 0}))
        for card in cards:
            options = card.get('options', [])
            votes = {opt['option']: opt['bets'] for opt in options}

            history = {
                'card_id': card['card_id'],
                'timestamp': datetime.utcnow().isoformat(),
                'votes': votes,
            }
            history_collection.insert_one(history)
    except Exception as e:
        print(f'Error in record_hourly_data: {str(e)}')


@app.route('/api/cards', methods=['GET'])
def get_cards() -> Response:
    try:
        tag_filter = request.args.get('tag')
        query = {'tags': {'$in': [tag_filter]}} if tag_filter else {}

        cards = list(cards_collection.find(query, {'_id': 0}))

        for card in cards:
            options = card.get('options', [])
            if any('percentage' not in opt for opt in options):
                total_bets = sum(option.get('bets', 0) for option in options)
                for option in options:
                    if 'percentage' not in option:
                        option['percentage'] = (
                            round((option.get('bets', 0) / total_bets) * 100, 2)
                            if total_bets > 0
                            else 0
                        )
            else:
                for option in options:
                    option['percentage'] = option.get('bets', 0)

        return jsonify(cards), 200
    except Exception as e:
        print(f'Error in get_cards: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/tags', methods=['GET'])
def get_tags():
    try:
        tags_cursor = cards_collection.aggregate(
            [{'$unwind': '$tags'}, {'$group': {'_id': '$tags'}}]
        )
        tags = [tag['_id'] for tag in tags_cursor]
        return jsonify(tags), 200
    except Exception as e:
        print(f'Error in get_tags: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/option_reasons/<card_id>/<option>', methods=['GET'])
def get_option_reasons(card_id, option):
    try:
        model = request.args.get('model', 'gpt-4o-mini')
        reason_doc = option_reasons_collection.find_one(
            {'card_id': card_id, 'option': option, 'model': model}, {'_id': 0}
        )

        if not reason_doc:
            return jsonify(
                {
                    'card_id': card_id,
                    'option': option,
                    'model': model,
                    'reasons': []
                }
            ), 200

        return jsonify(reason_doc), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_reasons', methods=['POST'])
def generate_reasons():
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from llm_reasoning_generator import generate_and_store_reasons_for_option
        
        data = request.json
        card_id = data.get('card_id')
        option = data.get('option')
        model = data.get('model', 'gpt-4o-mini')
        
        if not card_id or not option:
            return jsonify({'error': 'Missing card_id or option'}), 400
        
        # Check if the reasoning already exists
        existing = option_reasons_collection.find_one({
            'card_id': card_id, 
            'option': option, 
            'model': model
        })
        
        if existing:
            return jsonify(existing), 200
            
        # Get card information
        card = cards_collection.find_one({'card_id': card_id}, {'_id': 0})
        if not card:
            return jsonify({'error': 'Card not found'}), 404
            
        # Generate reasoning
        reasons = generate_and_store_reasons_for_option(
            card_id=card_id,
            question=card['question'], 
            option=option,
            model_name=model
        )
        
        return jsonify({
            'card_id': card_id,
            'option': option, 
            'model': model,
            'reasons': reasons
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/vote', methods=['POST'])
def vote():
    try:
        data = request.json
        card_id = data.get('card_id')
        option = data.get('option')
        reason_id = data.get('reason_id')

        if not card_id or not option:
            return jsonify({'error': 'Missing card_id or option'}), 400

        if reason_id:
            model = data.get('model', 'gpt-4o-mini')
            option_reasons_collection.update_one(
                {'card_id': card_id, 'option': option, 'model': model, 'reasons.reason_id': reason_id},
                {'$inc': {'reasons.$.votes': 1}},
                upsert=False,
            )

        return jsonify({'message': 'Vote recorded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/vote_history/<card_id>', methods=['GET'])
def get_vote_history(card_id):
    try:
        time_filter = request.args.get('days')
        query = {'card_id': card_id}

        if time_filter:
            days = int(time_filter)
            cutoff_date = datetime.utcnow() - datetime.timedelta(days=days)
            query['timestamp'] = {'$gte': cutoff_date.isoformat()}

        history = list(history_collection.find(query, {'_id': 0}).sort('timestamp', 1))
        return jsonify(history), 200
    except Exception as e:
        print(f'Error in get_vote_history: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/estimated_vote_history/<card_id>', methods=['GET'])
def get_estimated_vote_history(card_id):
    try:
        history = list(
            estimated_history_collection.find({'card_id': card_id}, {'_id': 0})
        )

        if not history:
            return jsonify([]), 200

        return jsonify(history), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/card_news/<card_id>', methods=['GET'])
def get_related_news(card_id):
    try:
        relation = relations_collection.find_one({'card_id': card_id}, {'_id': 0})

        if not relation:
            return jsonify([]), 200

        return jsonify(relation), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/news_cards/<news_id>', methods=['GET'])
def get_related_cards(news_id):
    try:
        relations = list(relations_collection.find({'news_ids': news_id}, {'_id': 0}))

        card_ids = [relation['card_id'] for relation in relations]

        cards = list(cards_collection.find({'card_id': {'$in': card_ids}}, {'_id': 0}))

        for card in cards:
            options = card.get('options', [])
            if any('percentage' not in opt for opt in options):
                total_bets = sum(option.get('bets', 0) for option in options)
                for option in options:
                    if 'percentage' not in option:
                        option['percentage'] = (
                            round((option.get('bets', 0) / total_bets) * 100, 2)
                            if total_bets > 0
                            else 0
                        )
            else:
                for option in options:
                    option['percentage'] = option.get('bets', 0)

        return jsonify(cards), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/card_news_relation', methods=['POST'])
def create_card_news_relation():
    try:
        data = request.json
        card_id = data.get('card_id')
        news_ids = data.get('news_ids', [])

        if not card_id:
            return jsonify({'error': 'Missing card_id'}), 400

        relations_collection.update_one(
            {'card_id': card_id}, {'$set': {'news_ids': news_ids}}, upsert=True
        )

        return jsonify({'message': 'Relation updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/polymarket_info/<card_id>', methods=['GET'])
def get_polymarket_info(card_id):
    try:
        card = cards_collection.find_one({'card_id': card_id}, {'_id': 0})
        if not card:
            return jsonify({'error': 'Card not found'}), 404

        polymarket_info = {
            'polymarket_id': card.get('polymarket_id'),
            'market_id': card.get('market_id'),
            'last_updated': card.get('last_updated'),
        }

        return jsonify(polymarket_info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5002, debug=True)
    finally:
        pass
