"""Dump per-news world_model predictions mu_i (routing-agnostic).

For each record, run the world_model ONCE on every candidate news (and the
no-news prompt) and store mu_i. The mu_i depend only on (model, news text,
history) — NOT on the attribution/routing weights. So any routing scheme
(direct / odds / L1 / posterior, any rho0 / eps / temp / prior file) becomes a
pure offline weighted sum over these stored mu_i. See scripts/aggregate_offline.py.

Usage (sharded, like the eval scripts):
    CUDA_VISIBLE_DEVICES=g python scripts/dump_per_news_predictions.py \
      --test-data-path <v6.jsonl> --model-path <ckpt> --model-name Qwen/Qwen3-8B \
      --output-path <out_sN.jsonl> --num-shards N --shard-idx s
"""

import argparse
import json

import torch
from tqdm import tqdm

from swm.dataset import MultiEventWorldModelDataset, _pack_prompts
from swm.utils.utils import load_records, set_seed
from swm.world_model import MultiEventWorldModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-data-path', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-name', default='Qwen/Qwen3-8B')
    p.add_argument('--output-path', required=True)
    p.add_argument('--max-seq-length', type=int, default=1024)
    p.add_argument(
        '--max-news',
        type=int,
        default=0,
        help='Cap candidate news per record (0 = ALL, recommended so any '
        'future prior can route to any news).',
    )
    p.add_argument('--max-history-len', type=int, default=None)
    p.add_argument(
        '--predict-delta', action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        '--pooling-method', type=str, default=None, choices=['mean', 'last_token']
    )
    p.add_argument('--chunk-size', type=int, default=8)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-idx', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    records = load_records(args.test_data_path)
    if args.num_shards > 1:
        records = [
            r for i, r in enumerate(records) if i % args.num_shards == args.shard_idx
        ]
        print(f'[shard {args.shard_idx}/{args.num_shards}] -> {len(records)} records')
    if args.max_history_len:
        for r in records:
            if r.history and len(r.history) > args.max_history_len:
                r.history = r.history[-args.max_history_len :]

    fk = {
        'model_name': args.model_name,
        'max_seq_length': args.max_seq_length,
        'max_news': 50,
        'predict_delta': args.predict_delta,
    }
    if args.pooling_method is not None:
        fk['pooling_method'] = args.pooling_method
    fc = MultiEventWorldModel(**fk)
    fc.load(args.model_path)
    fc.model.eval()
    dev = fc.model.llm.device

    # instance only used for its _build_prompt (no datapoints built on empty list)
    ds = MultiEventWorldModelDataset(
        records=[],
        tokenizer=fc.tokenizer,
        max_news=50,
        max_seq_length=args.max_seq_length,
        predict_delta=args.predict_delta,
    )

    n_out = 0
    with open(args.output_path, 'w') as w, torch.no_grad():
        for rec in tqdm(records, desc='Dumping per-news mu'):
            target = rec.target
            before_p = (
                float(rec.history[-1].get('p', 0.5))
                if rec.history
                else float(target.get('p', 0.5))
            )
            target_p = float(target.get('p', 0.5))
            true_delta = (target_p - before_p) if args.predict_delta else target_p

            news_list = rec.news or []
            cap = args.max_news if args.max_news > 0 else len(news_list)
            news_idxs = list(range(min(cap, len(news_list))))
            prompts = [ds._build_prompt(rec, target, news_list[i]) for i in news_idxs]
            prompts.append(ds._build_prompt(rec, target, None))  # no-news prompt last

            mus = []
            packed = _pack_prompts(fc.tokenizer, prompts, args.max_seq_length)
            ids = packed['input_ids'].to(dev)
            am = packed['attention_mask'].to(dev)
            for i in range(0, ids.size(0), args.chunk_size):
                out = fc.model(
                    input_ids=ids[i : i + args.chunk_size],
                    attention_mask=am[i : i + args.chunk_size],
                ).view(-1)
                mus.extend(out.cpu().tolist())

            row = {
                'market_id': rec.market_id,
                'event_id': rec.event_id,
                't': target.get('t'),
                'before_price': before_p,
                'true_delta': true_delta,
                'predict_delta': args.predict_delta,
                'per_news': [
                    {'news_idx': ni, 'mu': mus[k]} for k, ni in enumerate(news_idxs)
                ],
                'no_news_mu': mus[-1],
                # carry this file's attribution scores for convenience (routing input)
                'attributions': [
                    {'news_idx': a.get('news_idx'), 'score': float(a.get('score') or 0)}
                    for a in (rec.attributions or [])
                ],
            }
            w.write(json.dumps(row) + '\n')
            n_out += 1
    print(f'Wrote {n_out} records -> {args.output_path}')


if __name__ == '__main__':
    main()
