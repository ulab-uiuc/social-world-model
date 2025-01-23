import json
from typing import List

import pandas as pd

from swm.data import PolyMarketData
from swm.swm import RAGSocialWM


def load_market_data(data_path: str) -> List[PolyMarketData]:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return [PolyMarketData.from_dict(d) for d in data]


def train_and_evaluate():
    # Load data
    train_data = load_market_data('data/train.json')
    valid_data = load_market_data('data/valid.json')
    test_data = load_market_data('data/test.json')
    corpus_data = load_market_data('data/corpus.json')

    # Initialize model
    model = RAGSocialWM(model_name='mistralai/Mistral-7B-v0.1')

    # Train
    model.train(
        train_data=train_data, valid_data=valid_data, corpus_data=corpus_data, epochs=3
    )

    # Predict
    predictions = model.predict(test_data, corpus_data)

    # Save predictions
    results = []
    for market_id, outcomes in predictions.items():
        for outcome, value in outcomes.items():
            results.append(
                {'market_id': market_id, 'outcome': outcome, 'predicted_value': value}
            )

    pd.DataFrame(results).to_csv('predictions.csv', index=False)

    # Save model
    model.save('saved_model')


if __name__ == '__main__':
    train_and_evaluate()
