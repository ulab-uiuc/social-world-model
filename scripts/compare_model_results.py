"""
Compare two model results on the same test set across different data buckets.

Usage:
    python compare_model_results.py \
        --model1 ../results/forecaster_kalshi_0.6b_precomputed.jsonl \
        --model2 ../results/test_results_kalshi_data_processed_with_news_test_2025-11-01.jsonl \
        --model1-name "MultiEventForecaster" \
        --model2-name "Baseline"
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

import jsonlines
import numpy as np
import pandas as pd


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load results from jsonl file, handling different formats."""
    results = []
    with jsonlines.open(path) as reader:
        for item in reader:
            # Skip metrics-only lines
            if item.get('type') == 'metrics':
                continue
            results.append(item)
    return results


def get_sample_key(result: Dict) -> str:
    """Get unique key for a sample using (market_id, full_timestamp_sequence).
    
    This ensures we match samples with the same window history, not just the same target time.
    """
    market_id = result.get('market_id', '')
    
    # Get full timestamp sequence
    wh = result.get('window_history', [])
    if wh:
        # MultiEventForecaster format: window_history is list of {t, p}
        timestamps = tuple(int(p['t']) for p in wh)
    else:
        # Baseline format: timestamps_used is list of floats
        ts_used = result.get('timestamps_used', [])
        if ts_used:
            timestamps = tuple(int(t) for t in ts_used)
        else:
            timestamps = ()
    
    # Use hash for shorter key
    return f"{market_id}_{hash(timestamps)}"


def normalize_result(result: Dict, source: str) -> Dict:
    """Normalize result format from different sources."""
    normalized = {
        'market_id': result.get('market_id', ''),
        'event_id': result.get('event_id', ''),
        'question': result.get('question', ''),
        'source': source,
        'sample_key': get_sample_key(result),  # Unique key for matching
    }
    
    # Handle different result formats
    if 'pred_delta' in result:
        # MultiEventForecaster format
        normalized['t'] = result.get('t')
        normalized['pred_price'] = result['pred_price']
        normalized['true_price'] = result['true_price']
        normalized['before_price'] = result['before_price']
        normalized['pred_delta'] = result['pred_delta']
        normalized['true_delta'] = result['true_delta']
        normalized['abs_error'] = abs(result['pred_price'] - result['true_price'])
    elif 'y_pred' in result:
        # Baseline format (y_pred, y_true are lists)
        y_pred = result['y_pred'][0] if isinstance(result['y_pred'], list) else result['y_pred']
        y_true = result['y_true'][0] if isinstance(result['y_true'], list) else result['y_true']
        
        # Get timestamp
        ts = result.get('timestamps_used', [])
        normalized['t'] = ts[-1] if ts else None
        
        normalized['pred_price'] = y_pred
        normalized['true_price'] = y_true
        
        # Try to get before_price from prices_used (second to last value)
        prices_used = result.get('prices_used', [])
        if len(prices_used) >= 2:
            before_price = prices_used[-2]  # Price before the target
            normalized['before_price'] = before_price
            normalized['true_delta'] = y_true - before_price
            normalized['pred_delta'] = y_pred - before_price
        else:
            normalized['before_price'] = None
            normalized['pred_delta'] = None
            normalized['true_delta'] = None
        
        normalized['abs_error'] = abs(y_pred - y_true)
    
    # Extra metadata if available
    normalized['sample_type'] = result.get('sample_type', 'unknown')
    normalized['categories'] = result.get('categories', [])
    
    return normalized


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute metrics for a list of results."""
    if not results:
        return {'mae': float('nan'), 'mse': float('nan'), 'rmse': float('nan'), 'count': 0}
    
    errors = [r['abs_error'] for r in results]
    sq_errors = [e ** 2 for e in errors]
    
    return {
        'mae': np.mean(errors),
        'mse': np.mean(sq_errors),
        'rmse': np.sqrt(np.mean(sq_errors)),
        'count': len(results),
    }


def bucket_by_field(results: List[Dict], field: str) -> Dict[str, List[Dict]]:
    """Bucket results by a specific field."""
    buckets = defaultdict(list)
    for r in results:
        value = r.get(field, 'unknown')
        if isinstance(value, list):
            # For categories, add to each category bucket
            if value:
                for v in value:
                    buckets[v].append(r)
            else:
                buckets['no_category'].append(r)
        else:
            buckets[str(value)].append(r)
    return dict(buckets)


def bucket_by_price_range(results: List[Dict], bins: List[float] = None) -> Dict[str, List[Dict]]:
    """Bucket results by true_price range."""
    if bins is None:
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    buckets = defaultdict(list)
    for r in results:
        price = r.get('true_price')
        if price is None:
            buckets['unknown'].append(r)
            continue
        
        for i in range(len(bins) - 1):
            if bins[i] <= price < bins[i + 1]:
                bucket_name = f'{bins[i]:.1f}-{bins[i+1]:.1f}'
                buckets[bucket_name].append(r)
                break
        else:
            buckets['>=1.0'].append(r)
    
    return dict(buckets)


def bucket_by_abs_delta(results: List[Dict], bins: List[float] = None) -> Dict[str, List[Dict]]:
    """Bucket results by absolute true delta (volatility)."""
    if bins is None:
        bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
    
    buckets = defaultdict(list)
    for r in results:
        delta = r.get('true_delta')
        if delta is None:
            buckets['unknown'].append(r)
            continue
        
        abs_delta = abs(delta)
        for i in range(len(bins) - 1):
            if bins[i] <= abs_delta < bins[i + 1]:
                bucket_name = f'|Δ|={bins[i]:.2f}-{bins[i+1]:.2f}'
                buckets[bucket_name].append(r)
                break
        else:
            if abs_delta >= bins[-1]:
                bucket_name = f'|Δ|>={bins[-1]:.2f}'
                buckets[bucket_name].append(r)
    
    return dict(buckets)


def bucket_by_delta_topk(results: List[Dict], top_k: int = 50) -> Dict[str, List[Dict]]:
    """Bucket results by top-k largest absolute delta."""
    # Filter results with valid delta
    valid_results = [(r, abs(r.get('true_delta', 0))) for r in results if r.get('true_delta') is not None]
    
    if not valid_results:
        return {}
    
    # Sort by absolute delta descending
    valid_results.sort(key=lambda x: x[1], reverse=True)
    
    buckets = {}
    buckets[f'Top {top_k} largest |Δ|'] = [r for r, _ in valid_results[:top_k]]
    buckets[f'Top {top_k*2} largest |Δ|'] = [r for r, _ in valid_results[:top_k*2]]
    buckets[f'Bottom {top_k} smallest |Δ|'] = [r for r, _ in valid_results[-top_k:]]
    buckets['All'] = [r for r, _ in valid_results]
    
    return buckets


def bucket_by_delta_percentile(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Bucket results by delta percentile."""
    # Filter results with valid delta
    valid_results = [(r, abs(r.get('true_delta', 0))) for r in results if r.get('true_delta') is not None]
    
    if not valid_results:
        return {}
    
    # Sort by absolute delta descending
    valid_results.sort(key=lambda x: x[1], reverse=True)
    n = len(valid_results)
    
    buckets = {
        'Top 10% (largest Δ)': [r for r, _ in valid_results[:int(n*0.1)]],
        'Top 10-25%': [r for r, _ in valid_results[int(n*0.1):int(n*0.25)]],
        'Middle 25-75%': [r for r, _ in valid_results[int(n*0.25):int(n*0.75)]],
        'Bottom 25-10%': [r for r, _ in valid_results[int(n*0.75):int(n*0.9)]],
        'Bottom 10% (smallest Δ)': [r for r, _ in valid_results[int(n*0.9):]],
    }
    
    return buckets


def compare_models(results1: List[Dict], results2: List[Dict], 
                   name1: str, name2: str) -> None:
    """Compare two models across different buckets."""
    
    # Create lookup by sample_key (market_id, t) for matching
    lookup1 = {r['sample_key']: r for r in results1}
    lookup2 = {r['sample_key']: r for r in results2}
    
    common_keys = set(lookup1.keys()) & set(lookup2.keys())
    print(f"\n{'='*80}")
    print(f"Model Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    print(f"Model 1 ({name1}): {len(results1)} samples")
    print(f"Model 2 ({name2}): {len(results2)} samples")
    print(f"Common samples (matched by market_id + timestamp): {len(common_keys)}")
    
    # Use common samples for fair comparison
    common1 = [lookup1[key] for key in common_keys]
    common2 = [lookup2[key] for key in common_keys]
    
    # Overall metrics
    print(f"\n{'='*80}")
    print("Overall Metrics (on common samples)")
    print(f"{'='*80}")
    metrics1 = compute_metrics(common1)
    metrics2 = compute_metrics(common2)
    
    print(f"\n{'Metric':<15} {name1:<20} {name2:<20} {'Δ (M1-M2)':<15}")
    print("-" * 70)
    for metric in ['mae', 'rmse', 'mse']:
        v1, v2 = metrics1[metric], metrics2[metric]
        diff = v1 - v2
        better = "✓" if diff < 0 else ""
        print(f"{metric.upper():<15} {v1:<20.6f} {v2:<20.6f} {diff:+.6f} {better}")
    
    # Bucket analysis functions
    bucket_analyses = [
        ("Delta Percentile (by |true_delta|)", bucket_by_delta_percentile),
        ("Delta Top-K (by |true_delta|)", lambda r: bucket_by_delta_topk(r, top_k=50)),
        ("Absolute Delta Range", bucket_by_abs_delta),
        ("Sample Type", lambda r: bucket_by_field(r, 'sample_type')),
        ("Category", lambda r: bucket_by_field(r, 'categories')),
        ("True Price Range", bucket_by_price_range),
    ]
    
    for bucket_name, bucket_fn in bucket_analyses:
        print(f"\n{'='*80}")
        print(f"Bucket Analysis: {bucket_name}")
        print(f"{'='*80}")
        
        buckets1 = bucket_fn(common1)
        buckets2 = bucket_fn(common2)
        
        all_bucket_names = sorted(set(buckets1.keys()) | set(buckets2.keys()))
        
        print(f"\n{'Bucket':<25} {'Count':<6} {name1[:8]+' MAE':<12} {name2[:8]+' MAE':<12} {'Δ MAE':<10} {name1[:8]+' RMSE':<12} {name2[:8]+' RMSE':<12} {'Δ RMSE':<10} {'Winner(RMSE)':<12}")
        print("-" * 120)
        
        for b in all_bucket_names:
            b1 = buckets1.get(b, [])
            b2 = buckets2.get(b, [])
            
            # Get common samples in this bucket using sample_key
            keys1 = {r['sample_key'] for r in b1}
            keys2 = {r['sample_key'] for r in b2}
            common_bucket_keys = keys1 & keys2
            
            if not common_bucket_keys:
                continue
            
            bucket1_common = [r for r in b1 if r['sample_key'] in common_bucket_keys]
            bucket2_common = [r for r in b2 if r['sample_key'] in common_bucket_keys]
            
            m1 = compute_metrics(bucket1_common)
            m2 = compute_metrics(bucket2_common)
            
            diff_mae = m1['mae'] - m2['mae']
            diff_rmse = m1['rmse'] - m2['rmse']
            winner = name1 if diff_rmse < 0 else name2
            
            print(f"{b:<25} {m1['count']:<6} {m1['mae']:<12.4f} {m2['mae']:<12.4f} {diff_mae:+.4f}   {m1['rmse']:<12.4f} {m2['rmse']:<12.4f} {diff_rmse:+.4f}   {winner}")
    
    # Detailed per-sample comparison for worst cases
    print(f"\n{'='*80}")
    print("Samples where models differ most")
    print(f"{'='*80}")
    
    diffs = []
    for key in common_keys:
        r1, r2 = lookup1[key], lookup2[key]
        err1, err2 = r1['abs_error'], r2['abs_error']
        diffs.append({
            'market_id': r1.get('market_id', ''),
            'question': r1.get('question', '')[:60],
            'true_price': r1['true_price'],
            'true_delta': r1.get('true_delta'),
            f'{name1}_pred': r1['pred_price'],
            f'{name2}_pred': r2['pred_price'],
            f'{name1}_err': err1,
            f'{name2}_err': err2,
            'diff': err1 - err2,  # Negative = model1 better
        })
    
    diffs.sort(key=lambda x: x['diff'])
    
    print(f"\nTop 10 samples where {name1} is BETTER:")
    print("-" * 100)
    for d in diffs[:10]:
        print(f"  {d['market_id']}: {d['question']}")
        print(f"    True: {d['true_price']:.4f}, {name1}: {d[f'{name1}_pred']:.4f} (err={d[f'{name1}_err']:.4f}), "
              f"{name2}: {d[f'{name2}_pred']:.4f} (err={d[f'{name2}_err']:.4f})")
    
    print(f"\nTop 10 samples where {name2} is BETTER:")
    print("-" * 100)
    for d in diffs[-10:][::-1]:
        print(f"  {d['market_id']}: {d['question']}")
        print(f"    True: {d['true_price']:.4f}, {name1}: {d[f'{name1}_pred']:.4f} (err={d[f'{name1}_err']:.4f}), "
              f"{name2}: {d[f'{name2}_pred']:.4f} (err={d[f'{name2}_err']:.4f})")


def load_metadata(path: str) -> Dict[str, Dict]:
    """Load metadata from original test data."""
    metadata = {}
    with jsonlines.open(path) as reader:
        for item in reader:
            market_id = item.get('market_id', '')
            metadata[market_id] = {
                'sample_type': item.get('sample_type', 'unknown'),
                'categories': item.get('categories', []),
                'question': item.get('question', ''),
            }
    return metadata


def enrich_with_metadata(results: List[Dict], metadata: Dict[str, Dict]) -> List[Dict]:
    """Enrich results with metadata from original test data."""
    for r in results:
        mid = r.get('market_id', '')
        meta = metadata.get(mid, {})
        if not r.get('sample_type') or r.get('sample_type') == 'unknown':
            r['sample_type'] = meta.get('sample_type', 'unknown')
        if not r.get('categories'):
            r['categories'] = meta.get('categories', [])
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare two model results')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model results')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model results')
    parser.add_argument('--model1-name', type=str, default='Model1', help='Name for first model')
    parser.add_argument('--model2-name', type=str, default='Model2', help='Name for second model')
    parser.add_argument('--test-data', type=str, default=None, help='Path to original test data for metadata')
    args = parser.parse_args()
    
    # Load metadata if provided
    metadata = {}
    if args.test_data:
        print(f"Loading metadata from {args.test_data}...")
        metadata = load_metadata(args.test_data)
        print(f"  Found metadata for {len(metadata)} markets")
    
    print(f"Loading {args.model1}...")
    results1_raw = load_results(args.model1)
    results1 = [normalize_result(r, args.model1_name) for r in results1_raw]
    results1 = enrich_with_metadata(results1, metadata)
    
    print(f"Loading {args.model2}...")
    results2_raw = load_results(args.model2)
    results2 = [normalize_result(r, args.model2_name) for r in results2_raw]
    results2 = enrich_with_metadata(results2, metadata)
    
    compare_models(results1, results2, args.model1_name, args.model2_name)


if __name__ == '__main__':
    main()
