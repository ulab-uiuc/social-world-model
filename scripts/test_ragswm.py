# predict_ragswm.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from swm.swm import RAGSocialWM
from swm.utils.utils import load_polymarket_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict using the RAG Social Wisdom Model'
    )

    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--corpus-data-path', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--retriever-name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--predictions-path', type=str, default='predictions.csv')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--retriever-batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def predict(args):
    set_seed(args.seed)
    test_data = load_polymarket_data(args.test_data_path)
    corpus_data = load_polymarket_data(args.corpus_data_path)

    swm = RAGSocialWM(
        model_name=args.model_name,
        retriever_name=args.retriever_name,
        cache_dir=args.cache_dir,
        lora_config=None,
        corpus_markets=corpus_data,
        max_seq_length=args.max_seq_length,
        top_k=args.top_k,
        retriever_batch_size=args.retriever_batch_size,
    )

    swm.load(args.model_checkpoint)

    predictions = swm.predict(markets=test_data, batch_size=args.test_batch_size)

    results = []
    for market_id, outcomes in predictions.items():
        market = next((m for m in test_data if m.market_id == market_id), None)
        if not market:
            continue  # Skip if market not found

        for outcome, values in outcomes.items():
            results.append(
                {
                    'event_id': market.event_id,
                    'market_id': market.market_id,
                    'question': market.question,
                    'outcome': outcome,
                    'pred': values['pred'],
                    'label': values['label'],
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.predictions_path, index=False)

    valid_results = results_df.dropna(subset=['label'])

    if not valid_results.empty:
        y_pred = valid_results['pred'].astype(float).values
        y_true = valid_results['label'].astype(float).values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'MSE: {mse:.4f}')
    else:
        print('No valid labels available to calculate RMSE and MAE.')

    metrics = {
        'RMSE': rmse if not valid_results.empty else None,
        'MAE': mae if not valid_results.empty else None,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Path(args.output_dir) / 'evaluation_metrics.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    predict(args)
