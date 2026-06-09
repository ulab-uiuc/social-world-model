"""Generate PriorAttributer attributions for v6 records.

Runs a trained PriorAttributer on a v6 jsonl file and writes a copy whose
`attributions` field has been replaced with the attributer's output.

Usage:
    python inference_prior_attribution.py \
        --data-path /path/to/v6.jsonl \
        --attributer-path /path/to/prior_attributer/checkpoint \
        --model-name Qwen/Qwen3-0.6B \
        --output-path /path/to/output.jsonl
"""

import argparse

import jsonlines
from tqdm import tqdm

from swm.attributer import BasicPriorAttributer
from swm.utils.utils import load_records, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate PriorAttributer attributions'
    )
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--attributer-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument(
        '--max-news',
        type=int,
        default=30,
        help='Cap candidate news scored per record. MUST cover all candidates (>=world_model max-news=30) or tail news get silently dropped (default attributer used 20).',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help='Split data into N shards for parallel multi-GPU inference.',
    )
    parser.add_argument(
        '--shard-idx',
        type=int,
        default=0,
        help='Which shard (0..N-1) this process handles.',
    )
    parser.add_argument(
        '--infer-temperature',
        type=float,
        default=None,
        help='Override the softmax temperature at inference only (sharpens output without retraining). Default: keep trained target_temperature.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f'Loading data from {args.data_path}...')
    records = load_records(args.data_path)
    if args.num_shards > 1:
        records = [
            r for i, r in enumerate(records) if i % args.num_shards == args.shard_idx
        ]
        print(f'[shard {args.shard_idx}/{args.num_shards}] -> {len(records)} records')
    print(f'Loaded {len(records)} records')

    print(f'Loading attributer from {args.attributer_path}...')
    attributer = BasicPriorAttributer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )
    attributer.load(args.attributer_path)
    attributer.max_news = args.max_news
    print(
        f'[fix] attributer.max_news forced to {attributer.max_news} (score ALL candidate news, no tail truncation)'
    )
    if args.infer_temperature is not None:
        print(
            f'[temp] overriding inference softmax temperature {attributer.target_temperature} -> {args.infer_temperature} (logits unchanged; sharpening only)'
        )
        attributer.target_temperature = args.infer_temperature

    updated = 0
    for record in tqdm(records, desc='Generating attributions'):
        # CRITICAL: clear any precomputed (posterior/oracle) attributions first
        # so the output carries ONLY this prior attributer's scores — otherwise
        # records the prior skips would silently leak posterior labels into the
        # "prior" file. Mirrors MultiEventWorldModel._generate_attributions.
        record.attributions = []
        if not record.news:
            continue
        attributions = attributer.attribute_record(record)
        if attributions:
            record.attributions = [
                {'news_idx': a['news_idx'], 'score': a['score']} for a in attributions
            ]
            updated += 1

    print(f'Generated attributions for {updated} records')

    print(f'Writing output to {args.output_path}...')
    with jsonlines.open(args.output_path, mode='w') as writer:
        for r in records:
            writer.write(r.model_dump())

    print(f'Done! Output: {args.output_path}')


if __name__ == '__main__':
    main()
