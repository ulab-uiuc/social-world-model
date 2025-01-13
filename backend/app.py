from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


CARDS = [
    {
        'id': 1,
        'title': 'Who will win the president election in 2024?',
        'options': [
            {'label': 'A', 'percentage': '45%'},
            {'label': 'B', 'percentage': '40%'},
            {'label': 'C', 'percentage': '10%'},
            {'label': 'D', 'percentage': '5%'},
        ],
    },
    {
        'id': 2,
        'title': 'Who will win the president election in 2020?',
        'options': [
            {'label': 'A', 'percentage': '45%'},
            {'label': 'B', 'percentage': '40%'},
            {'label': 'C', 'percentage': '10%'},
            {'label': 'D', 'percentage': '5%'},
        ],
    },
    {
        'id': 3,
        'title': 'Who will win the president election in 2016?',
        'options': [
            {'label': 'A', 'percentage': '45%'},
            {'label': 'B', 'percentage': '40%'},
            {'label': 'C', 'percentage': '10%'},
            {'label': 'D', 'percentage': '5%'},
        ],
    },
    {
        'id': 1,
        'title': 'Who will win the president election in 2012?',
        'options': [
            {'label': 'A', 'percentage': '45%'},
            {'label': 'B', 'percentage': '40%'},
            {'label': 'C', 'percentage': '10%'},
            {'label': 'D', 'percentage': '5%'},
        ],
    },
]


@app.route('/api/cards', methods=['GET'])
def get_cards() -> Response:
    return jsonify(CARDS)


if __name__ == '__main__':
    app.run(debug=True)
