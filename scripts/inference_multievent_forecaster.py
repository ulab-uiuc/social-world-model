"""Inference with MultiEventForecaster on v6 records.

Can use either:
1. Precomputed attributions in the input file
2. A PriorAttributer to generate attributions on the fly

Usage with precomputed attributions:
    python inference_multievent_forecaster.py \
        --test-data-path ../data/v6/test_polymarket.jsonl \
        --model-path ../saves/multievent_forecaster/checkpoint-best \
        --output-path ../results/predictions.jsonl

Usage with PriorAttributer:
    python inference_multievent_forecaster.py \
        --test-data-path ../data/v6/test_polymarket.jsonl \
        --model-path ../saves/multievent_forecaster/checkpoint-best \
        --attributer-path ../saves/prior_attributer/checkpoint-best \
        --output-path ../results/predictions.jsonl
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import jsonlines

from swm.attributer import BasicPriorAttributer
from swm.forecaster import MultiEventForecaster
from swm.utils.utils import load_records, set_seed


def unix_to_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')


def build_prompt_for_display(
    question: str,
    description: str,
    history: List[Dict],
    target: Dict,
    news_items: List[Dict],
) -> str:
    lines = [f'Event: {question}']
    if description:
        lines.append(f'Description: {description}')

    lines.append('\nRecent price history:')
    for day in history:
        lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

    target_date = unix_to_date(target['t'])
    if news_items:
        lines.append('\nRelevant news:')
        for i, news in enumerate(news_items[:5], 1):
            lines.append(f"  [{i}] {news.get('title', '')}")
    lines.append(f'\nPredict the probability on {target_date}:')
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with MultiEventForecaster')
    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)

    parser.add_argument('--attributer-path', type=str, default=None)
    parser.add_argument('--null-gate', action='store_true',
                        help='Two-stage routing: also score the "no relevant news" option; '
                             'if it wins (or exceeds --null-gate-threshold) treat the record '
                             'as null. Otherwise fall back to --score-threshold on the news.')
    parser.add_argument('--null-gate-threshold', type=float, default=None,
                        help='If set, null when no-news score >= this. Unset = null when '
                             'no-news ranks #1 (parameter-free).')

    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--attributer-model-name', type=str, default=None,
                        help='Base model for the attributer (defaults to --model-name).')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-news', type=int, default=30)
    parser.add_argument('--max-history-len', type=int, default=None)
    parser.add_argument('--only-null', action='store_true',
                        help='Score ONLY null records (no oracle-positive news).')
    parser.add_argument('--pooling-method', type=str, default=None,
                        choices=['mean', 'last_token'],
                        help='Override pooling method (auto-detected from checkpoint by default).')

    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--weight-temperature', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading test data from {args.test_data_path}...")
    records = load_records(args.test_data_path)
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records")

    data_with_attr = sum(1 for r in records if r.attributions)
    print(f"Records with precomputed attributions: {data_with_attr}/{len(records)}")

    print(f"Loading forecaster from {args.model_path}...")
    forecaster_kwargs = {
        'model_name': args.model_name,
        'max_seq_length': args.max_seq_length,
        'max_news': args.max_news,
    }
    if args.pooling_method is not None:
        forecaster_kwargs['pooling_method'] = args.pooling_method
    forecaster = MultiEventForecaster(**forecaster_kwargs)
    forecaster.load(args.model_path)

    attributer = None
    if args.attributer_path:
        print(f"Loading attributer from {args.attributer_path}...")
        attributer = BasicPriorAttributer(
            model_name=args.attributer_model_name or args.model_name,
            max_seq_length=args.max_seq_length,
        )
        attributer.load(args.attributer_path)
        attributer.null_gate = args.null_gate
        attributer.null_gate_threshold = args.null_gate_threshold
        if args.null_gate:
            thr = args.null_gate_threshold if args.null_gate_threshold is not None else 'argmax/parameter-free'
            print(f"Null gate ON (threshold={thr})")
    elif data_with_attr == 0:
        raise ValueError(
            "No precomputed attributions found and no attributer provided. "
            "Either use a v6 file with attributions or pass --attributer-path."
        )

    print("Running inference...")
    results = forecaster.predict(
        records=records,
        attributer=attributer,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
    )

    # Build lookup AFTER predict so it includes any newly generated attributions.
    record_lookup = {}
    for r in records:
        t = r.target.get('t')
        if t is None:
            continue
        record_lookup[(r.market_id, t)] = r

    enriched = []
    for result in results:
        key = (result['market_id'], result['t'])
        record = record_lookup.get(key)
        if record is None:
            continue

        news_list = record.news
        attributions = record.attributions
        related_news = []
        for attr in attributions:
            idx = attr.get('news_idx', -1)
            score = attr.get('score') or 0
            if score > 0 and 0 <= idx < len(news_list):
                item = dict(news_list[idx])
                item['attribution_score'] = score
                related_news.append(item)
        related_news.sort(key=lambda x: x.get('attribution_score', 0), reverse=True)

        prompt = build_prompt_for_display(
            question=record.question,
            description=record.description or '',
            history=record.history,
            target=record.target,
            news_items=related_news,
        )

        delta_error = result['pred_delta'] - result['true_delta']
        price_error = result['pred_price'] - result['true_price']

        enriched.append({
            'event_id': result['event_id'],
            'market_id': result['market_id'],
            't': result['t'],
            'date': unix_to_date(result['t']) if result['t'] else None,
            'question': record.question,
            'history': record.history,
            'input_prompt': prompt,
            'news': news_list,
            'attributions': attributions,
            'related_news': related_news[:10],
            'pred_delta': result['pred_delta'],
            'pred_price': result['pred_price'],
            'true_delta': result['true_delta'],
            'true_price': result['true_price'],
            'before_price': result['before_price'],
            'delta_error': delta_error,
            'price_error': price_error,
            'abs_delta_error': abs(delta_error),
            'abs_price_error': abs(price_error),
        })

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(enriched)} predictions to {args.output_path}...")
    with jsonlines.open(output_path, mode='w') as writer:
        for r in enriched:
            writer.write(r)

    if not enriched:
        return

    import numpy as np
    pred_delta = np.array([r['pred_delta'] for r in enriched])
    true_delta = np.array([r['true_delta'] for r in enriched])
    pred_price = np.array([r['pred_price'] for r in enriched])
    true_price = np.array([r['true_price'] for r in enriched])

    delta_mse = float(np.mean((pred_delta - true_delta) ** 2))
    price_mse = float(np.mean((pred_price - true_price) ** 2))
    delta_corr = float(np.corrcoef(pred_delta, true_delta)[0, 1]) if len(enriched) > 1 else float('nan')
    price_corr = float(np.corrcoef(pred_price, true_price)[0, 1]) if len(enriched) > 1 else float('nan')
    direction_acc = float(np.mean((pred_delta > 0) == (true_delta > 0)))

    print(f"\n{'=' * 60}")
    print(f"Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model_path}")
    print(f"  Test records: {len(enriched)}")
    print(f"  Delta MSE: {delta_mse:.6f}  RMSE: {delta_mse ** 0.5:.6f}  corr: {delta_corr:.4f}")
    print(f"  Price MSE: {price_mse:.6f}  RMSE: {price_mse ** 0.5:.6f}  corr: {price_corr:.4f}")
    print(f"  Direction Accuracy: {direction_acc:.2%}")
    print(f"{'=' * 60}")

    metrics = {
        'model_path': args.model_path,
        'test_records': len(enriched),
        'delta_mse': delta_mse,
        'price_mse': price_mse,
        'delta_correlation': None if np.isnan(delta_corr) else delta_corr,
        'price_correlation': None if np.isnan(price_corr) else price_corr,
        'direction_accuracy': direction_acc,
    }
    with open(output_path.with_suffix('.metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
