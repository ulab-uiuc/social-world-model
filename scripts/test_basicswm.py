import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from swm.swm import BasicSocialWM
from swm.utils.utils import load_polymarket_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict using the Basic Social Wisdom Model'
    )

    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity-check', action='store_true')
    return parser.parse_args()


def predict(args):
    set_seed(args.seed)
    if args.sanity_check:
        test_data = load_polymarket_data(args.test_data_path)[:1]
    else:
        test_data = load_polymarket_data(args.test_data_path)

    swm = BasicSocialWM(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
    )

    swm.load(args.model_checkpoint)

    preds, gths = swm.predict(markets=test_data, batch_size=args.test_batch_size)
    rmse = np.sqrt(mean_squared_error(gths, preds))
    mae = mean_absolute_error(gths, preds)
    mse = mean_squared_error(gths, preds)

    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Path(args.output_dir) / 'evaluation_metrics.csv', index=False)

if __name__ == '__main__':
    args = parse_args()
    predict(args)
