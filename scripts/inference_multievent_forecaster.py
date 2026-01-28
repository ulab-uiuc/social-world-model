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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jsonlines

from swm.forecaster import MultiEventForecaster
from swm.attributer import BasicPriorAttributer
from swm.utils.utils import load_flat_samples_as_markets, set_seed


def unix_to_date(ts: int) -> str:
    """Convert Unix timestamp to date string."""
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')


def build_prompt_for_display(
    question: str,
    description: str,
    window_history: List[Dict],
    target: Dict,
    news_items: List[Dict],
) -> str:
    """Build a human-readable prompt for the output JSON."""
    lines = [f'Event: {question}']
    if description:
        lines.append(f'Description: {description}')
    
    # Exclude target point from history
    target_ts = target.get('t')
    history_before_target = [
        day for day in window_history 
        if day.get('t') != target_ts
    ]
    
    lines.append('\nRecent price history:')
    for day in history_before_target:
        date = unix_to_date(day['t'])
        lines.append(f"  {date}: {day['p']:.3f}")
    
    target_date = unix_to_date(target['t'])
    
    # Add news items
    if news_items:
        lines.append('\nRelevant news:')
        for i, news in enumerate(news_items[:5], 1):  # Limit to top 5 for display
            news_title = news.get('title', '')
            lines.append(f"  [{i}] {news_title}")
    
    lines.append(f'\nPredict the probability on {target_date}:')
    
    return '\n'.join(lines)


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
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-news-per-bp', type=int, default=30,
                        help='Max news articles per breakpoint')
    parser.add_argument('--max-history-len', type=int, default=None,
                        help='Max history points to use (None = use all)')
    
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
    test_data = load_flat_samples_as_markets(args.test_data_path)
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
        max_news_per_bp=args.max_news_per_bp,
        max_history_len=args.max_history_len,
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
    
    # Run inference (this may generate attributions if attributer is provided)
    print("Running inference...")
    results = forecaster.predict(
        markets=test_data,
        attributer=attributer,
        batch_size=args.batch_size,
    )
    
    # Build lookup dict AFTER predict, so it includes any newly generated attributions
    market_lookup = {}
    for market in test_data:
        if not market.daily_breakpoints:
            continue
        for bp in market.daily_breakpoints:
            after = bp.get('after', {})
            t = after.get('t')
            if t is not None:
                key = (market.market_id, t)
                market_lookup[key] = {
                    'question': market.question,
                    'description': market.description or '',
                    'window_history': bp.get('window_history', []),
                    'before': bp.get('before', {}),
                    'after': after,
                    'news': bp.get('news', []),
                    'attributions': bp.get('attributions', []),
                }
    
    # Enrich results with additional information
    print("Enriching results with detailed information...")
    enriched_results = []
    for result in results:
        key = (result['market_id'], result['t'])
        market_info = market_lookup.get(key, {})
        
        # Get related news items (those with positive attribution scores)
        news_list = market_info.get('news', [])
        attributions = market_info.get('attributions', [])
        related_news = []
        for attr in attributions:
            news_idx = attr.get('news_idx', 0)
            score = attr.get('score') or 0
            if score > 0 and news_idx < len(news_list):
                news_item = news_list[news_idx].copy()
                news_item['attribution_score'] = score
                related_news.append(news_item)
        # Sort by attribution score descending
        related_news.sort(key=lambda x: x.get('attribution_score', 0), reverse=True)
        
        # Build input prompt for display
        input_prompt = build_prompt_for_display(
            question=market_info.get('question', ''),
            description=market_info.get('description', ''),
            window_history=market_info.get('window_history', []),
            target=market_info.get('after', {}),
            news_items=related_news,
        )
        
        # Calculate errors
        delta_error = result['pred_delta'] - result['true_delta']
        price_error = result['pred_price'] - result['true_price']
        abs_delta_error = abs(delta_error)
        abs_price_error = abs(price_error)
        
        enriched_result = {
            # Original fields
            'event_id': result['event_id'],
            'market_id': result['market_id'],
            't': result['t'],
            'date': unix_to_date(result['t']) if result['t'] else None,
            
            # Question and context
            'question': market_info.get('question', ''),
            
            # Window history
            'window_history': market_info.get('window_history', []),
            
            # Input prompt
            'input_prompt': input_prompt,
            
            # All news and attributions (for reproducibility)
            'news': news_list,
            'attributions': attributions,
            
            # Related news used for prediction (filtered by positive score)
            'related_news': related_news[:10],  # Limit to top 10
            
            # Predictions
            'pred_delta': result['pred_delta'],
            'pred_price': result['pred_price'],
            
            # Ground truth
            'true_delta': result['true_delta'],
            'true_price': result['true_price'],
            'before_price': result['before_price'],
            
            # Errors
            'delta_error': delta_error,
            'price_error': price_error,
            'abs_delta_error': abs_delta_error,
            'abs_price_error': abs_price_error,
        }
        enriched_results.append(enriched_result)
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(enriched_results)} predictions to {args.output_path}...")
    with jsonlines.open(output_path, mode='w') as writer:
        for result in enriched_results:
            writer.write(result)
    
    # Compute and print metrics
    if enriched_results:
        import numpy as np
        
        # Extract delta and price predictions
        pred_delta = np.array([r['pred_delta'] for r in enriched_results])
        true_delta = np.array([r['true_delta'] for r in enriched_results])
        pred_price = np.array([r['pred_price'] for r in enriched_results])
        true_price = np.array([r['true_price'] for r in enriched_results])
        
        # Delta metrics (what the model directly predicts)
        delta_mse = np.mean((pred_delta - true_delta) ** 2)
        delta_rmse = np.sqrt(delta_mse)
        delta_mae = np.mean(np.abs(pred_delta - true_delta))
        
        # Price metrics (after recovering absolute price)
        price_mse = np.mean((pred_price - true_price) ** 2)
        price_rmse = np.sqrt(price_mse)
        price_mae = np.mean(np.abs(pred_price - true_price))
        
        # Correlation
        if len(enriched_results) > 1:
            delta_corr = np.corrcoef(pred_delta, true_delta)[0, 1]
            price_corr = np.corrcoef(pred_price, true_price)[0, 1]
        else:
            delta_corr = float('nan')
            price_corr = float('nan')
        
        # Direction accuracy (did we predict the direction of price change correctly?)
        pred_direction = pred_delta > 0  # Predicted price increase
        true_direction = true_delta > 0  # Actual price increase
        direction_acc = np.mean(pred_direction == true_direction)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"  Model: {args.model_path}")
        print(f"  Test samples: {len(enriched_results)}")
        print(f"\n  Delta Metrics (y_t+1 - y_t):")
        print(f"    MSE:  {delta_mse:.6f}")
        print(f"    RMSE: {delta_rmse:.6f}")
        print(f"    MAE:  {delta_mae:.6f}")
        print(f"    Correlation: {delta_corr:.4f}")
        print(f"\n  Price Metrics (absolute price):")
        print(f"    MSE:  {price_mse:.6f}")
        print(f"    RMSE: {price_rmse:.6f}")
        print(f"    MAE:  {price_mae:.6f}")
        print(f"    Correlation: {price_corr:.4f}")
        print(f"\n  Direction Accuracy: {direction_acc:.2%}")
        print(f"{'='*60}")
        
        # Save metrics to JSON
        metrics_path = output_path.with_suffix('.metrics.json')
        metrics = {
            'model_path': args.model_path,
            'test_samples': len(enriched_results),
            'delta_mse': float(delta_mse),
            'delta_rmse': float(delta_rmse),
            'delta_mae': float(delta_mae),
            'delta_correlation': float(delta_corr) if not np.isnan(delta_corr) else None,
            'price_mse': float(price_mse),
            'price_rmse': float(price_rmse),
            'price_mae': float(price_mae),
            'price_correlation': float(price_corr) if not np.isnan(price_corr) else None,
            'direction_accuracy': float(direction_acc),
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()

