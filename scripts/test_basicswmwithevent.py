import argparse
from pathlib import Path

import pandas as pd
from peft import LoraConfig

from swm.predictor import BasicPredictor
from swm.reasoner import BasicPriorReasoner
from swm.swm import BasicSocialWMWithEvent
from swm.utils.metric import calculate_reg_metric
from swm.utils.utils import load_polymarket_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test BasicSocialWMWithEvent')

    # Data paths
    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, required=True)

    # Model configs
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')

    # Test configs
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')

    # LoRA configs
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--r', type=int, default=16)

    # Prior Reasoner configs
    parser.add_argument(
        '--prior-model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct'
    )
    parser.add_argument('--prior-model-checkpoint', type=str, required=True)
    parser.add_argument('--prior-max-seq-length', type=int, default=1024)

    # Debug
    parser.add_argument('--sanity-check', action='store_true')

    return parser.parse_args()


def test_social_wm(args):
    # Set random seed
    set_seed(args.seed)

    # Load test data
    if args.sanity_check:
        test_data = load_polymarket_data(args.test_data_path)[:2]
    else:
        test_data = load_polymarket_data(args.test_data_path)

    # Setup LoRA configs
    predictor_lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    prior_lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # Initialize predictor
    predictor = BasicPredictor(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        lora_config=predictor_lora_config,
    )

    # Initialize prior reasoner
    prior_reasoner = BasicPriorReasoner(
        model_name=args.prior_model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.prior_max_seq_length,
        lora_config=prior_lora_config,
    )

    # Initialize the combined model
    social_wm = BasicSocialWMWithEvent(
        predictor=predictor,
        prior_reasoner=prior_reasoner,
    )

    # Load checkpoints
    social_wm.load_models(args.model_checkpoint)

    # Run predictions
    print('Running predictions...')
    results = social_wm.predict(
        markets=test_data,
        batch_size=args.test_batch_size,
    )

    # Process results and calculate metrics
    predictions = []
    ground_truth = []
    market_ids = []
    timestamps = []

    for result in results:
        predictions.append(result['prediction'])
        ground_truth.append(result['ground_truth'])
        market_ids.append(result['market_id'])
        timestamps.append(result['t'])

    # Calculate metrics
    metrics = calculate_reg_metric(predictions, ground_truth)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics
    pd.DataFrame([metrics]).to_csv(output_dir / 'metrics.csv', index=False)

    # Save detailed predictions
    predictions_df = pd.DataFrame(
        {
            'market_id': market_ids,
            'timestamp': timestamps,
            'prediction': predictions,
            'ground_truth': ground_truth,
        }
    )
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Print metrics
    print('\nTest Metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')


if __name__ == '__main__':
    args = parse_args()
    test_social_wm(args)
