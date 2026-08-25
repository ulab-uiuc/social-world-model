#!/usr/bin/env python3
"""Score a regression checkpoint on the streaming test contexts.

Emits the same metric block as `eval_streaming_icl.py` off the same prompts, so
the fine-tune, the in-context baseline and the shipped checkpoint can be read
side by side, and writes rows in the shape `scripts/backtest_report.py` expects
so the same trading evaluation can be run on any of them.

    python scripts/eval_streaming_sft.py --data data/streaming/test.jsonl \\
        --model-path saves/streaming_sft/final-model \\
        --model-name Qwen/Qwen2.5-7B-Instruct --out results/streaming/sft.jsonl
"""

import argparse
import json
import math
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from swm.utils.regressor import LLMRegressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='streaming test.jsonl')
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-name', default='Qwen/Qwen2.5-7B-Instruct',
                   help='tokenizer source')
    p.add_argument('--out', required=True)
    p.add_argument('--max-seq-length', type=int, default=30000)
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--attn-implementation', default='flash_attention_2')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--tag', default='sft')
    return p.parse_args()


def metrics(pred, true):
    n = len(pred)
    if n < 2:
        return {'n': n}
    mp, mt = stats.fmean(pred), stats.fmean(true)
    sp, st = stats.pstdev(pred), stats.pstdev(true)
    cov = sum((a - mp) * (b - mt) for a, b in zip(pred, true)) / n
    mse = stats.fmean([(a - b) ** 2 for a, b in zip(pred, true)])
    baseline = stats.fmean([b * b for b in true])
    moved = [(a, b) for a, b in zip(pred, true) if abs(b) > 1e-9 and abs(a) > 1e-9]
    up = sum(1 for b in true if b > 0)
    return {
        'n': n,
        'pearson': cov / (sp * st) if sp and st else float('nan'),
        'rmse': math.sqrt(mse),
        'no_change_rmse': math.sqrt(baseline),
        'skill_vs_no_change': 1 - mse / baseline if baseline else float('nan'),
        'mae': stats.fmean([abs(a - b) for a, b in zip(pred, true)]),
        'direction_accuracy': (
            sum(1 for a, b in moved if (a > 0) == (b > 0)) / len(moved)
            if moved else float('nan')
        ),
        # The test block is ~68% up-moves, so a constant "up" is a strong
        # directional baseline and any accuracy below it is worse than a guess.
        'majority_direction': max(up, n - up) / n,
        'pred_std': sp,
        'true_std': st,
        'pred_mean': mp,
    }


def main():
    args = parse_args()
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f'{len(rows)} rows')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LLMRegressor.from_pretrained(args.model_path)
    if args.attn_implementation:
        from transformers import AutoModel

        state = model.llm.state_dict()
        # FA2 refuses fp32, and the checkpoint's config declares fp32, so ask
        # for the compute dtype at load time rather than casting afterwards.
        rebuilt = AutoModel.from_pretrained(
            model.config.base_model_name_or_path,
            dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32,
            attn_implementation=args.attn_implementation,
        )
        state = {k: v.to(rebuilt.dtype) for k, v in state.items()}
        rebuilt.load_state_dict(state)
        model.llm = rebuilt
    model = model.to('cuda').eval()
    if args.dtype == 'bfloat16':
        # Backbone only: the head stays fp32 and forward() upcasts into it, so
        # the scalar the metrics are computed from keeps full precision.
        model.llm = model.llm.to(torch.bfloat16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results, n_truncated = [], 0
    with out_path.open('w') as fout, torch.no_grad():
        for row in tqdm(rows, desc=args.tag):
            ids = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
            if len(ids) > args.max_seq_length:
                n_truncated += 1
                ids = ids[-args.max_seq_length :]  # keep the tail: pooling anchor
            input_ids = torch.tensor([ids], device=model.llm.device)
            pred = float(
                model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                ).view(-1)[0]
            )
            record = {
                'market_id': row['market_id'],
                'event_id': row.get('event_id'),
                't': row['t'],
                'question': row.get('question', ''),
                'pred_delta': pred,
                'true_delta': row['label_delta'],
                'before_price': row['before_price'],
                'pred_price': row['before_price'] + pred,
                'true_price': row['target_price'],
                'n_input_tokens': len(ids),
                '_ctx': row.get('_ctx', {}),
            }
            results.append(record)
            fout.write(json.dumps(record) + '\n')

    print(f'\nwrote {len(results)} -> {out_path}   truncated: {n_truncated}')
    scored = metrics([r['pred_delta'] for r in results],
                     [r['true_delta'] for r in results])
    print(f'\n--- {args.tag} (n={scored["n"]}) ---')
    for k, v in scored.items():
        print(f'  {k:22s} {v:.4f}' if isinstance(v, float) else f'  {k:22s} {v}')
    Path(str(out_path) + '.metrics.json').write_text(json.dumps(scored, indent=2))


if __name__ == '__main__':
    main()
