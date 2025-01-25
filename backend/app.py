from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)


client: MongoClient[Any] = MongoClient('mongodb://localhost:27017/')
db = client['electionDB']
collection = db['cards']


@app.route('/api/cards', methods=['GET'])
def get_cards() -> Response:
    try:
        tag_filter = request.args.get('tag')
        query = {}

        if tag_filter:
            query = {'tags': tag_filter}

        cards = list(collection.find(query, {'_id': 0}))

        for card in cards:
            options = card.get('options', [])
            total_bets = sum(option.get('bets', 0) for option in options)

            for option in options:
                if total_bets > 0:
                    option['percentage'] = round(
                        (option.get('bets', 0) / total_bets) * 100, 2
                    )
                else:
                    option['percentage'] = 0

        response = jsonify(cards)
        response.status_code = 200
        return response
    except Exception as e:
        error_response = jsonify({'error': str(e)})
        error_response.status_code = 500
        return error_response


@app.route('/api/tags', methods=['GET'])
def get_tags():
    try:
        tags_cursor = collection.aggregate(
            [{'$unwind': '$tags'}, {'$group': {'_id': '$tags'}}]
        )

        tags = [tag['_id'] for tag in tags_cursor]

        return jsonify(tags), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/vote', methods=['POST'])
def vote():
    try:
        data = request.json
        card_id = data['card_id']
        option = data['option']

        result = collection.update_one(
            {'card_id': card_id, 'options.option': option},
            {'$inc': {'options.$.bets': 1}},
        )

        if result.matched_count == 0:
            return jsonify({'error': 'Card or option not found'}), 404

        return jsonify({'message': 'Vote recorded successfully'}), 200
    except Exception as e:
        error_response = jsonify({'error': str(e)})
        error_response.status_code = 500
        return error_response


if __name__ == '__main__':
    app.run(debug=True)
