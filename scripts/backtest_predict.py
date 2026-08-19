#!/usr/bin/env python3
"""Score a backtest grid with the SWM world model.

Reads the cells written by scripts/backtest_build_grid.py, runs the
MultiEventWorldModel over them, and writes one prediction row per cell -- and
one for every cell too, not just the ones the dataset builder keeps: a cell
whose retrieved headlines all score zero routes to the model's null option and
is emitted with `pred_delta = 0` rather than silently vanishing, so downstream
coverage numbers stay honest.

The `_bt` block from the grid is carried through untouched; nothing in it ever
reaches a prompt.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from swm.data import Record
from swm.world_model import MultiEventWorldModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-name', default='Qwen/Qwen2.5-7B-Instruct',
                   help='tokenizer source; the checkpoint ships weights only')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-seq-length', type=int, default=1024)
    p.add_argument('--max-news', type=int, default=8)
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--null-rho0', type=float, default=1.0)
    p.add_argument('--odds-eps', type=float, default=1e-3)
    p.add_argument('--odds-temp', type=float, default=1.0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-idx', type=int, default=0)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cells = [json.loads(line) for line in open(args.grid) if line.strip()]
    if args.limit:
        cells = cells[: args.limit]
    if args.num_shards > 1:
        cells = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard_idx]
    print(f'[shard {args.shard_idx}/{args.num_shards}] cells={len(cells)}')

    world_model = MultiEventWorldModel(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        max_news=args.max_news,
        predict_delta=True,
    )
    world_model.load(args.model_path)
    world_model.null_rho0 = args.null_rho0
    world_model.odds_eps = args.odds_eps
    world_model.odds_temp = args.odds_temp
    if args.dtype == 'bfloat16':
        # Training ran FSDP mixed precision: bf16 compute over fp32 masters. The
        # checkpoint serialises fp32, so casting the backbone back to bf16 makes
        # inference match the arithmetic the weights were fitted under (and fits
        # a 7B alongside whatever else is on the card). The regression head stays
        # fp32 -- forward() upcasts the pooled vector into it.
        world_model.model.llm = world_model.model.llm.to(torch.bfloat16)
    print(f'loaded: max_news={world_model.max_news} '
          f'predict_delta={world_model.predict_delta} '
          f'pooling={world_model.model.config.pooling_method} dtype={args.dtype}')

    records = [Record.from_dict(c) for c in cells]
    meta = {(str(c['market_id']), int(c['target']['t'])): c for c in cells}

    results = world_model.predict(records=records, batch_size=args.batch_size)
    scored = {(str(r['market_id']), int(r['t'])): r for r in results}
    print(f'scored {len(scored)}/{len(cells)} cells '
          f'({len(cells) - len(scored)} routed to null / had no candidates)')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fout:
        for key, cell in meta.items():
            bt = cell['_bt']
            got = scored.get(key)
            before = cell['history'][-1]['p']
            pred_delta = float(got['pred_delta']) if got else 0.0
            fout.write(
                json.dumps(
                    {
                        'market_id': key[0],
                        'event_id': cell['event_id'],
                        't': key[1],
                        'question': cell['question'],
                        'scored': got is not None,
                        'before_price': before,
                        'pred_delta': pred_delta,
                        'pred_price': before + pred_delta,
                        'true_delta': cell['target']['p'] - before,
                        'true_price': cell['target']['p'],
                        'n_news': bt['n_news'],
                        'top_news': [n.get('title') for n in (cell.get('news') or [])[:3]],
                        '_bt': bt,
                    }
                )
                + '\n'
            )
    print(f'wrote {len(meta)} rows -> {out_path}')


if __name__ == '__main__':
    main()
