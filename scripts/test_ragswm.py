# predict_ragswm.py

import argparse
from pathlib import Path

import pandas as pd
import torch

from swm.data import PolyMarketData
from swm.swm import RAGSocialWM


def set_seed(seed: int = 42):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_polymarket_data(data_path):
    import jsonlines

    with jsonlines.open(data_path, 'r') as reader:
        return [PolyMarketData.from_dict(d) for d in reader]


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
    for market in test_data:
        preds = predictions.get(market.market_id, {})
        for outcome, value in preds.items():
            results.append(
                {
                    'event_id': market.event_id,
                    'market_id': market.market_id,
                    'question': market.question,
                    'outcome': outcome,
                    'prediction': value,
                    'label': market.label.get(outcome, None)
                    if hasattr(market, 'label')
                    else None,
                }
            )

    results_df = pd.DataFrame(results)
    predictions_file_path = Path(args.output_dir) / args.predictions_path
    results_df.to_csv(predictions_file_path, index=False)


if __name__ == '__main__':
    args = parse_args()
    predict(args)
