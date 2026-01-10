"""
Inference with MultiEventForecaster.

Can use either:
1. Precomputed attributions from file
2. PriorAttributer to generate attributions on-the-fly

Usage with precomputed attributions:
    python inference_multievent_forecaster.py \
        --test-data-path ../data/attributed/test.jsonl \
        --model-path ../saves/multievent_forecaster/checkpoint-best \
        --output-path ../results/predictions.jsonl

Usage with PriorAttributer:
    python inference_multievent_forecaster.py \
        --test-data-path ../data/processed/test.jsonl \
        --model-path ../saves/multievent_forecaster/checkpoint-best \
        --attributer-path ../saves/prior_attributer/checkpoint-best \
        --output-path ../results/predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import jsonlines

from swm.forecaster import MultiEventForecaster
from swm.attributer import BasicPriorAttributer
from swm.utils.utils import load_market_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference with MultiEventForecaster'
    )
    # Data paths
    parser.add_argument('--test-data-path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained forecaster checkpoint')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save predictions')
    
    # Optional: use attributer for on-the-fly attribution
    parser.add_argument('--attributer-path', type=str, default=None,
                        help='Path to trained PriorAttributer (optional)')
    
    # Model config
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of markets to process')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load test data
    print(f"Loading test data from {args.test_data_path}...")
    test_data = load_market_data(args.test_data_path)
    if args.limit:
        test_data = test_data[:args.limit]
    print(f"Loaded {len(test_data)} markets")
    
    # Check if data has attributions (attributions are inside daily_breakpoints)
    def has_attributions(market):
        if not market.daily_breakpoints:
            return False
        for bp in market.daily_breakpoints:
            if bp.get('attributions') and len(bp.get('attributions', [])) > 0:
                return True
        return False
    
    data_with_attr = sum(1 for m in test_data if has_attributions(m))
    print(f"Markets with precomputed attributions: {data_with_attr}/{len(test_data)}")
    
    # Load forecaster
    print(f"Loading forecaster from {args.model_path}...")
    forecaster = MultiEventForecaster(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
    )
    forecaster.load(args.model_path)
    
    # Optionally load attributer for on-the-fly attribution
    attributer = None
    if args.attributer_path:
        print(f"Loading attributer from {args.attributer_path}...")
        attributer = BasicPriorAttributer(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            max_seq_length=args.max_seq_length,
        )
        attributer.load(args.attributer_path)
    elif data_with_attr == 0:
        raise ValueError(
            "No precomputed attributions found and no attributer provided. "
            "Either use data with attributions or provide --attributer-path"
        )
    
    # Run inference
    print("Running inference...")
    results = forecaster.predict(
        markets=test_data,
        attributer=attributer,
        batch_size=args.batch_size,
    )
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(results)} predictions to {args.output_path}...")
    with jsonlines.open(output_path, mode='w') as writer:
        for result in results:
            writer.write(result)
    
    # Compute and print metrics
    if results:
        import numpy as np
        
        predictions = np.array([r['prediction'] for r in results])
        ground_truths = np.array([r['ground_truth'] for r in results])
        
        mse = np.mean((predictions - ground_truths) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truths))
        
        # Correlation
        if len(results) > 1:
            corr = np.corrcoef(predictions, ground_truths)[0, 1]
        else:
            corr = float('nan')
        
        # Direction accuracy (for price movement)
        # Assuming > 0.5 means "Yes", < 0.5 means "No"
        pred_direction = predictions > 0.5
        true_direction = ground_truths > 0.5
        direction_acc = np.mean(pred_direction == true_direction)
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results")
        print(f"{'='*50}")
        print(f"  Model: {args.model_path}")
        print(f"  Test samples: {len(results)}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  Direction Accuracy: {direction_acc:.2%}")
        print(f"{'='*50}")
        
        # Save metrics to JSON
        metrics_path = output_path.with_suffix('.metrics.json')
        metrics = {
            'model_path': args.model_path,
            'test_samples': len(results),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(corr) if not np.isnan(corr) else None,
            'direction_accuracy': float(direction_acc),
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()

