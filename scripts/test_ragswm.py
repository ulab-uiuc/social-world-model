import argparse
from pathlib import Path
import jsonlines
import pandas as pd
import torch
from swm.data import PolyMarketData
from swm.swm import RAGSocialWM
from swm.utils.metric import calculate_mae, calculate_rmse
from tqdm import tqdm
import logging
import sys
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_polymarket_data(data_path):
    with jsonlines.open(data_path, 'r') as reader:
        return [PolyMarketData.from_dict(d) for d in reader]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the RAG Social Wisdom Model'
    )

    parser.add_argument('--test-data-path', type=str, 
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl',
        help='Path to the test dataset in JSON Lines format.'
    )
    parser.add_argument('--corpus-data-path', type=str,
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl',
        help='Path to the corpus dataset in JSON Lines format.'
    )
    parser.add_argument('--model-checkpoint', type=str, required=True,
        help='Path to the saved model checkpoint directory.'
    )
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
        help='Name or path of the base model.'
    )
    parser.add_argument('--retriever-name', type=str, default='all-MiniLM-L6-v2',
        help='Name or path of the SentenceTransformer model for retrieval.'
    )
    parser.add_argument('--batch-size', type=int, default=8,
        help='Batch size for prediction.'
    )
    parser.add_argument('--cache-dir', type=str, default='./cache',
        help='Directory to cache models and tokenizers.'
    )
    parser.add_argument('--output-dir', type=str, default='../output',
        help='Directory to save output files.'
    )
    parser.add_argument('--predictions-path', type=str, default='predictions.csv',
        help='Filename for saving prediction results.'
    )
    parser.add_argument('--max-seq-length', type=int, default=1024,
        help='Maximum sequence length for model inputs.'
    )
    parser.add_argument('--top-k', type=int, default=50,
        help='Number of top similar markets to retrieve.'
    )
    parser.add_argument('--retriever-batch-size', type=int, default=8,
        help='Batch size for the retriever.'
    )
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )
    return parser.parse_args()


def evaluate():
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory set to: {output_dir}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")

    logger.info("Loading test data...")
    test_data = load_polymarket_data(args.test_data_path)
    logger.info(f"Loaded {len(test_data)} test samples.")

    logger.info("Loading corpus data...")
    corpus_data = load_polymarket_data(args.corpus_data_path)
    logger.info(f"Loaded {len(corpus_data)} corpus samples.")

    model = RAGSocialWM(
        model_name=args.model_name, 
        retriever_name=args.retriever_name,
        cache_dir=args.cache_dir,
        output_dir=output_dir,
        corpus_markets=corpus_data,
        max_seq_length=args.max_seq_length,
        top_k=args.top_k,
        retriever_batch_size=args.retriever_batch_size
    )
    logger.info("RAGSocialWM initialized.")

    try:
        logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
        model.load(args.model_checkpoint, device=device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    try:
        logger.info("Starting batch predictions...")
        predictions_dict = model.predict_batch(test_data, batch_size=args.batch_size)
        logger.info("Batch predictions completed.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        sys.exit(1)

    logger.info("Aggregating results...")
    results = []
    for market in test_data:
        if market.market_id in predictions_dict:
            pred = predictions_dict[market.market_id]
            if pred:
                outcome = list(pred.keys())[0]
                prediction_value = pred[outcome]
                results.append({
                    'event_id': market.event_id,
                    'market_id': market.market_id,
                    'question': market.question,
                    'prediction': prediction_value,
                    'label': market.label,
                })
            else:
                results.append({
                    'event_id': market.event_id,
                    'market_id': market.market_id,
                    'question': market.question,
                    'prediction': None,
                    'label': market.label,
                })
        else:
            results.append({
                'event_id': market.event_id,
                'market_id': market.market_id,
                'question': market.question,
                'prediction': None,
                'label': market.label,
            })

    results_df = pd.DataFrame(results)

    valid_df = results_df.dropna(subset=['prediction', 'label'])
    logger.info(f"Valid predictions: {len(valid_df)} out of {len(results_df)}")

    try:
        rmse = calculate_rmse(valid_df['prediction'], valid_df['label'])
        mae = calculate_mae(valid_df['prediction'], valid_df['label'])
        logger.info(f'Test RMSE: {rmse:.4f}')
        logger.info(f'Test MAE: {mae:.4f}')
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        sys.exit(1)

    try:
        predictions_file_path = output_dir / args.predictions_path
        results_df.to_csv(predictions_file_path, index=False)
        logger.info(f"Predictions saved to {predictions_file_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {e}")
        sys.exit(1)


if __name__ == '__main__':
    evaluate()
