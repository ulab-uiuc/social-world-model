from typing import Any, Dict, List

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)



client: MongoClient[Any] = MongoClient("mongodb://localhost:27017/")
db = client['electionDB'] 
collection = db['cards'] 

@app.route('/api/cards', methods=['GET'])
def get_cards() -> Response:
    try:
        cards: List[Dict[str, Any]] = list(collection.find({}, {"_id": 0}))
        response = jsonify(cards)
        response.status_code = 200
        return response
    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.status_code = 500
        return error_response

if __name__ == '__main__':
    app.run(debug=True)


