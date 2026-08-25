#!/usr/bin/env python3
"""In-context baseline: an untouched instruct model reads the same streaming prompt.

The fine-tune and this baseline consume byte-identical contexts out of
`build_streaming_data.py`. The only difference is that one had its weights
updated and the other did not, so whatever gap appears is attributable to the
training rather than to prompt engineering.

The model is asked for a bare signed number. Parsing is deliberately strict --
first float in the reply, rejected if it falls outside a plausible delta range --
and refusals are counted rather than silently coerced to zero, because a
baseline that abstains on a third of the set and is scored as "predicted no
change" looks far better than it is.

    python scripts/eval_streaming_icl.py --data data/streaming/test.jsonl \\
        --model Qwen/Qwen2.5-7B-Instruct --out results/streaming/icl.jsonl
"""

import argparse
import json
import math
import re
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = (
    'You are forecasting a prediction market. Above is the market, its recent '
    'daily prices, and how it moved after news on earlier occasions. Given the '
    'headlines at the final decision point, predict the CHANGE in the market '
    'price over the next 24 hours.\n\n'
    'Answer with the change, not the new price. The current price is {price:.3f}, '
    'so your answer plus {price:.3f} must land between 0 and 1. Changes of this '
    'kind are usually between -0.30 and +0.30. Match the format of the '
    '"=> 24h change:" lines above.\n\n'
    'Reply with a single signed decimal number and nothing else. '
    'For example: -0.042'
)
NUMBER = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='streaming test.jsonl')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--out', required=True)
    p.add_argument('--max-input-tokens', type=int, default=30000)
    p.add_argument('--max-new-tokens', type=int, default=12)
    p.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-idx', type=int, default=0)
    p.add_argument('--abstain-as-zero', action='store_true',
                   help='score unparseable replies as a 0.0 delta instead of '
                        'excluding them; reported either way')
    return p.parse_args()


def extract_delta(text: str, before_price: float, floor: float = 0.0):
    """First *physically possible* signed decimal in the reply, or None.

    A delta only makes sense if `before_price + delta` lands inside [0, 1]:
    the instruct model sometimes answers with the new price instead of the
    change, and a +0.95 on a market quoted at 0.635 would imply a price of
    1.585. Rejecting those is not tuning the baseline up -- it is refusing to
    score an answer to a different question. They are counted as abstentions.
    """
    for match in NUMBER.finditer(text):
        try:
            value = float(match.group())
        except ValueError:
            continue
        if -1.0 <= value <= 1.0 and floor <= before_price + value <= 1.0:
            return value
    return None


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
    return {
        'n': n,
        'pearson': cov / (sp * st) if sp and st else float('nan'),
        'rmse': math.sqrt(mse),
        'no_change_rmse': math.sqrt(baseline),
        'skill_vs_no_change': 1 - mse / baseline if baseline else float('nan'),
        'mae': stats.fmean([abs(a - b) for a, b in zip(pred, true)]),
        'direction_accuracy': (
            sum(1 for a, b in moved if (a > 0) == (b > 0)) / len(moved) if moved else float('nan')
        ),
        'pred_std': sp,
        'true_std': st,
        'pred_mean': mp,
    }


def main():
    args = parse_args()
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]
    print(f'[shard {args.shard_idx}/{args.num_shards}] {len(rows)} rows')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map='cuda',
        attn_implementation='sdpa',
    )
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    n_abstain = n_truncated = 0

    with out_path.open('w') as fout, torch.no_grad():
        for row in tqdm(rows, desc='ICL'):
            instruction = INSTRUCTION.format(price=float(row['before_price']))
            messages = [
                {'role': 'user', 'content': f'{row["prompt"]}\n\n{instruction}'}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(ids) > args.max_input_tokens:
                # Same priority as training: drop the oldest demonstrations, keep
                # the current headlines and the question.
                n_truncated += 1
                ids = ids[-args.max_input_tokens :]
            input_ids = torch.tensor([ids], device=model.device)
            generated = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            reply = tokenizer.decode(
                generated[0][input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            pred = extract_delta(reply, float(row['before_price']))
            if pred is None:
                n_abstain += 1
            record = {
                'market_id': row['market_id'],
                't': row['t'],
                'question': row.get('question', ''),
                'raw_reply': reply[:120],
                'pred_delta': pred,
                'true_delta': row['label_delta'],
                'before_price': row['before_price'],
                'pred_price': (row['before_price'] + pred) if pred is not None else None,
                'true_price': row['target_price'],
                'n_input_tokens': len(ids),
                '_ctx': row.get('_ctx', {}),
            }
            results.append(record)
            fout.write(json.dumps(record) + '\n')

    print(f'\nwrote {len(results)} -> {out_path}')
    print(f'unparseable replies: {n_abstain} ({100 * n_abstain / max(len(results), 1):.1f}%)  '
          f'contexts truncated: {n_truncated}')

    scored = [r for r in results if r['pred_delta'] is not None]
    print(f'\n--- parsed only (n={len(scored)}) ---')
    for k, v in metrics([r['pred_delta'] for r in scored],
                        [r['true_delta'] for r in scored]).items():
        print(f'  {k:22s} {v:.4f}' if isinstance(v, float) else f'  {k:22s} {v}')
    if n_abstain:
        print(f'\n--- abstentions scored as 0.0 (n={len(results)}) ---')
        for k, v in metrics([r['pred_delta'] or 0.0 for r in results],
                            [r['true_delta'] for r in results]).items():
            print(f'  {k:22s} {v:.4f}' if isinstance(v, float) else f'  {k:22s} {v}')


if __name__ == '__main__':
    main()
