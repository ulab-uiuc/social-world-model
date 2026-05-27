#!/usr/bin/env python3
"""
Fast end-to-end inference for MultiEventForecaster with BasicPriorAttributer.

Key optimizations vs inference_multievent_forecaster.py:
  1. Dynamic padding (padding=True) instead of padding='max_length' (~5x).
  2. Large attributer batches (default 64) with sequences sorted by length.
  3. Batches span breakpoints - the flat list of all news items is run through
     the attributer in one pass, amortizing per-call overhead.
  4. Forecaster only runs on top-K news per breakpoint (selected by attributer
     score), instead of re-scoring every news item. For top-k=1 the forecaster
     workload drops from ~193k items to ~4.5k items (~40x).

Usage:
    python inference_fast.py \
        --test-data-path ../data/vllm_attributed/combined_test_vllm_attributed.jsonl \
        --model-path ../saves/forecaster_vllm_v17_8b_hlr20/checkpoint-2100 \
        --attributer-path ../saves/prior_attributer_combined_8b/checkpoint-600 \
        --model-name Qwen/Qwen3-8B \
        --output-path ../results/e2e_attr8b_fc8b_combined.jsonl \
        --attributer-batch-size 32 \
        --forecaster-batch-size 16 \
        --top-k 1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Force fast loading: bf16 + FA2 for every underlying LLM load. ---
_orig_from_pretrained = AutoModelForCausalLM.from_pretrained


def _fast_from_pretrained(*args, **kwargs):
    kwargs.setdefault('torch_dtype', torch.bfloat16)
    kwargs.setdefault('attn_implementation', 'flash_attention_2')
    return _orig_from_pretrained(*args, **kwargs)


AutoModelForCausalLM.from_pretrained = _fast_from_pretrained

from swm.attributer import BasicPriorAttributer  # noqa: E402
from swm.forecaster import MultiEventForecaster  # noqa: E402
from swm.utils.utils import load_flat_samples_as_markets, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-data-path', required=True)
    p.add_argument('--model-path', required=True, help='Forecaster checkpoint')
    p.add_argument('--attributer-path', required=True, help='Attributer checkpoint')
    p.add_argument('--model-name', default='Qwen/Qwen3-8B')
    p.add_argument('--output-path', required=True)
    p.add_argument('--cache-dir', default='./cache')
    p.add_argument('--max-seq-length', type=int, default=512)
    p.add_argument('--attributer-batch-size', type=int, default=32)
    p.add_argument('--forecaster-batch-size', type=int, default=16)
    p.add_argument('--max-news-per-bp', type=int, default=50,
                   help='Max news to score per bp with attributer')
    p.add_argument('--top-k', type=int, default=1,
                   help='Number of top-scoring news per bp to feed to forecaster')
    p.add_argument('--weight-temperature', type=float, default=0.1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shard-index', type=int, default=0,
                   help='0-based index of this shard (for parallel multi-GPU runs)')
    p.add_argument('--shard-count', type=int, default=1,
                   help='Total number of shards - split bps uniformly')
    p.add_argument('--merge-lora', action='store_true', default=True,
                   help='Merge LoRA adapter into base weights (~1.2x faster)')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the backbone (~1.3x faster, ~40s warmup)')
    return p.parse_args()


def collect_bp_tasks(markets, max_news_per_bp):
    """Return flat list of bp descriptors + prompt tasks.

    Returns:
        bps: list of dicts, one per breakpoint (kept in processing order).
        news_prompts: list of (bp_idx, local_news_idx) tuples.
    """
    bps = []
    for market in markets:
        if not market.daily_breakpoints:
            continue
        for bp in market.daily_breakpoints:
            news_list = bp.get('news') or []
            if len(news_list) < 2:
                continue
            bps.append({
                'market': market,
                'bp': bp,
                'news': news_list[:max_news_per_bp],
                'scores': None,
                'pred_delta': None,
            })
    return bps


def tokenize_sorted(tokenizer, prompts, max_seq_length):
    """Tokenize prompts and return them sorted ascending by length.

    Returns:
        sorted_ids: list of 1-D int tensors, sorted by length ascending.
        inv_order: tensor mapping sorted index -> original index.
    """
    ids_list = []
    for p in prompts:
        enc = tokenizer(p, truncation=True, max_length=max_seq_length, return_tensors=None)
        ids_list.append(torch.tensor(enc['input_ids'], dtype=torch.long))

    lengths = [int(x.size(0)) for x in ids_list]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    sorted_ids = [ids_list[i] for i in order]
    inv_order = torch.empty(len(order), dtype=torch.long)
    for new_idx, old_idx in enumerate(order):
        inv_order[old_idx] = new_idx
    return sorted_ids, inv_order


def cast_regressor_bf16(model):
    """Cast the LLMRegressor's head to bf16 to match the LLM backbone.

    The LLM is loaded in bf16 via the monkey-patch above, but the regression
    head tensors stored in regression_head.bin are fp32. Cast them so matmul
    dtypes match and we don't fall back to fp32 kernels.
    """
    model.regression_head = model.regression_head.to(dtype=torch.bfloat16)
    return model


def maybe_merge_and_compile(model, merge_lora=True, compile_model=False):
    """Apply PEFT LoRA merge + optional torch.compile to the backbone.

    - merge_and_unload: folds LoRA adapter weights into base Linear layers,
      removing the per-layer lora_A @ lora_B matmul. ~18% speedup observed.
    - torch.compile: graph-optimises the model call. ~32% speedup observed.
      First forward compiles (~40s warmup).
    """
    if merge_lora and hasattr(model.llm, 'merge_and_unload'):
        print("[opt] merging LoRA adapter into base weights...")
        model.llm = model.llm.merge_and_unload()
        model.eval()
    if compile_model:
        print("[opt] torch.compile(mode='default') on backbone...")
        torch.set_float32_matmul_precision('high')
        model.llm = torch.compile(model.llm, mode='default', fullgraph=False)
    return model


def shard_list(xs, shard_index, shard_count):
    """Return xs[shard_index::shard_count] - uniform strided shard."""
    if shard_count <= 1:
        return xs
    return xs[shard_index::shard_count]


def batched_forward_dynamic(model, sorted_ids, pad_id, batch_size, device, desc):
    """Run model forward on sorted tensors with per-batch dynamic padding.

    Within each batch, pad to the batch's longest sequence only. Returns a
    1-D tensor of predictions in SORTED order (caller must un-sort).
    """
    model.eval()
    n = len(sorted_ids)
    preds = torch.empty(n, dtype=torch.float32)
    pbar = tqdm(range(0, n, batch_size), desc=desc)
    with torch.no_grad():
        for start in pbar:
            end = min(start + batch_size, n)
            batch = sorted_ids[start:end]
            max_len = max(t.size(0) for t in batch)
            b_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            b_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, t in enumerate(batch):
                L = t.size(0)
                b_ids[i, :L] = t
                b_mask[i, :L] = 1
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            out = model(input_ids=b_ids, attention_mask=b_mask)
            if isinstance(out, dict):
                out = out['predictions']
            preds[start:end] = out.view(-1).float().cpu()
            pbar.set_postfix(max_len=max_len)
    return preds


def run_attributer(attributer, bps, batch_size, max_seq_length, device):
    """Score every (bp, news) pair in one big batched pass."""
    # Build flat list of prompts and (bp_idx, news_idx) indices
    flat_prompts = []
    flat_keys = []  # (bp_idx, news_idx)
    for bp_idx, item in enumerate(bps):
        market = item['market']
        bp = item['bp']
        window_history = bp.get('window_history', [])
        target = bp.get('after', {})
        for news_idx, news in enumerate(item['news']):
            prompt = attributer._build_prompt_with_news(market, window_history, target, news)
            flat_prompts.append(prompt)
            flat_keys.append((bp_idx, news_idx))

    print(f"[attr] total prompts: {len(flat_prompts)}")
    t0 = time.time()
    sorted_ids, inv_order = tokenize_sorted(
        attributer.tokenizer, flat_prompts, max_seq_length
    )
    lens = [int(t.size(0)) for t in sorted_ids]
    print(f"[attr] tokenized in {time.time()-t0:.1f}s, "
          f"seq len: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")

    pad_id = attributer.tokenizer.pad_token_id or attributer.tokenizer.eos_token_id or 0
    t0 = time.time()
    sorted_preds = batched_forward_dynamic(
        attributer.model, sorted_ids, pad_id, batch_size, device,
        desc='Attr batches',
    )
    dt = time.time() - t0
    print(f"[attr] forward in {dt:.1f}s "
          f"({len(flat_prompts)/max(dt,1e-6):.1f} items/s)")

    preds = torch.empty_like(sorted_preds)
    preds[inv_order] = sorted_preds

    # Attributer is trained with MSE on raw scores (sigmoid head not used here)
    # The checkpoints in use (prior_attributer_combined_8b/*) are raw regression;
    # we just use the raw output as a relative score.
    scores_np = preds.numpy()

    # Assign scores back to each bp
    for item in bps:
        item['scores'] = [None] * len(item['news'])
    for flat_idx, (bp_idx, news_idx) in enumerate(flat_keys):
        bps[bp_idx]['scores'][news_idx] = float(scores_np[flat_idx])

    return bps


def run_forecaster(forecaster, bps, top_k, batch_size, max_seq_length, weight_temperature, device):
    """Run forecaster only on top-K news per breakpoint."""
    from swm.dataset import MultiEventForecasterDataset  # lazy import not needed, but OK

    # Build one forecaster prompt per selected news item
    flat_prompts = []
    flat_keys = []  # (bp_idx, rank) so we know which bp the prompt belongs to

    for bp_idx, item in enumerate(bps):
        scores = item['scores']
        # Top-k by score desc
        ranked = sorted(range(len(scores)), key=lambda i: scores[i] or 0.0, reverse=True)[:top_k]
        item['top_news_indices'] = ranked

        market = item['market']
        bp = item['bp']
        window_history = bp.get('window_history', [])
        target = bp.get('after', {})

        # Use forecaster's dataset prompt builder directly via a throwaway instance
        # method binding: we need access to _build_prompt. Construct a minimal
        # dataset helper on-the-fly.
        for rank, news_idx in enumerate(ranked):
            news = item['news'][news_idx]
            prompt = _fc_build_prompt(
                market, window_history, target, news,
                max_history_len=forecaster.max_history_len,
                predict_absolute_price=forecaster.predict_absolute_price,
            )
            flat_prompts.append(prompt)
            flat_keys.append((bp_idx, rank))

    print(f"[fc] total prompts: {len(flat_prompts)}")
    t0 = time.time()
    sorted_ids, inv_order = tokenize_sorted(
        forecaster.tokenizer, flat_prompts, max_seq_length
    )
    lens = [int(t.size(0)) for t in sorted_ids]
    print(f"[fc] tokenized in {time.time()-t0:.1f}s, "
          f"seq len: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")

    pad_id = forecaster.tokenizer.pad_token_id or forecaster.tokenizer.eos_token_id or 0
    t0 = time.time()
    sorted_preds = batched_forward_dynamic(
        forecaster.model, sorted_ids, pad_id, batch_size, device,
        desc='FC batches',
    )
    dt = time.time() - t0
    print(f"[fc] forward in {dt:.1f}s "
          f"({len(flat_prompts)/max(dt,1e-6):.1f} items/s)")

    preds = torch.empty_like(sorted_preds)
    preds[inv_order] = sorted_preds

    # Aggregate per bp: softmax(attributer scores / T) over top-k, weighted avg
    for flat_idx, (bp_idx, rank) in enumerate(flat_keys):
        item = bps[bp_idx]
        if 'fc_preds' not in item:
            item['fc_preds'] = [None] * len(item['top_news_indices'])
        item['fc_preds'][rank] = float(preds[flat_idx])

    for item in bps:
        fc_preds = item.get('fc_preds')
        ranked = item['top_news_indices']
        if not fc_preds:
            item['pred_delta'] = 0.0
            continue
        raw_scores = [item['scores'][i] or 0.0 for i in ranked]
        s = torch.tensor(raw_scores, dtype=torch.float32)
        w = F.softmax(s / weight_temperature, dim=0) if len(s) > 1 else torch.tensor([1.0])
        p = torch.tensor(fc_preds, dtype=torch.float32)
        item['pred_delta'] = float((w * p).sum())

    return bps


def _fc_build_prompt(market, window_history, target, news, max_history_len=None, predict_absolute_price=False):
    """Replica of MultiEventForecasterDataset._build_prompt (default direction=0)."""
    from swm.utils.utils import unix_to_date

    lines = [f'Event: {market.question}']
    if market.description:
        lines.append(f'Description: {market.description}')

    target_ts = target.get('t')
    if predict_absolute_price:
        history_before_target = [day for day in window_history if day.get('t') != target_ts][-5:]
    else:
        history_before_target = [day for day in window_history if day.get('t') < target_ts]
        if max_history_len is not None and len(history_before_target) > max_history_len:
            history_before_target = history_before_target[-max_history_len:]

    lines.append('\nRecent price history:')
    for day in history_before_target:
        lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

    target_date = unix_to_date(target['t'])
    news_title = news.get('title', '')
    news_desc = news.get('description', '') or ''
    lines.append(f'\nNews: {news_title}')
    if news_desc:
        lines.append(f'{news_desc}')
    lines.append(f'\nPredict the probability on {target_date}:')
    return '\n'.join(lines)


def write_results(bps, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in bps:
            market = item['market']
            bp = item['bp']
            before = bp.get('before', {})
            after = bp.get('after', {})
            before_price = before.get('p', 0.5)
            after_price = after.get('p', 0.5)
            true_delta = after_price - before_price

            ranked = item.get('top_news_indices', [])
            top_news = []
            for idx in ranked:
                n = item['news'][idx].copy()
                n['attribution_score'] = item['scores'][idx]
                top_news.append(n)

            record = {
                'market_id': market.market_id,
                'event_id': getattr(market, 'event_id', market.market_id),
                't': after.get('t'),
                'question': market.question,
                'description': market.description or '',
                'before_price': before_price,
                'after_price': after_price,
                'true_delta': true_delta,
                'pred_delta': item['pred_delta'],
                'pred_price': before_price + item['pred_delta'],
                'top_news': top_news,
            }
            f.write(json.dumps(record) + '\n')


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading test data from {args.test_data_path}...")
    markets = load_flat_samples_as_markets(args.test_data_path)
    if args.limit:
        markets = markets[:args.limit]
    print(f"Loaded {len(markets)} markets")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading attributer from {args.attributer_path}...")
    attributer = BasicPriorAttributer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        max_news_per_bp=args.max_news_per_bp,
    )
    attributer.load(args.attributer_path)
    cast_regressor_bf16(attributer.model)
    maybe_merge_and_compile(
        attributer.model,
        merge_lora=args.merge_lora,
        compile_model=args.compile,
    )

    bps_all = collect_bp_tasks(markets, args.max_news_per_bp)
    bps = shard_list(bps_all, args.shard_index, args.shard_count)
    print(f"Collected {len(bps_all)} breakpoints; "
          f"shard {args.shard_index}/{args.shard_count} → {len(bps)} to score")

    bps = run_attributer(
        attributer, bps,
        batch_size=args.attributer_batch_size,
        max_seq_length=args.max_seq_length,
        device=device,
    )

    # Free attributer VRAM before loading forecaster (same model size)
    del attributer
    torch.cuda.empty_cache()

    print(f"Loading forecaster from {args.model_path}...")
    forecaster = MultiEventForecaster(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        max_news_per_bp=args.top_k,
    )
    forecaster.load(args.model_path)
    cast_regressor_bf16(forecaster.model)
    maybe_merge_and_compile(
        forecaster.model,
        merge_lora=args.merge_lora,
        compile_model=args.compile,
    )

    bps = run_forecaster(
        forecaster, bps,
        top_k=args.top_k,
        batch_size=args.forecaster_batch_size,
        max_seq_length=args.max_seq_length,
        weight_temperature=args.weight_temperature,
        device=device,
    )

    write_results(bps, args.output_path)
    print(f"\nWrote {len(bps)} results to {args.output_path}")

    # Quick eval
    correct = 0
    total = 0
    mse = 0.0
    bl = 0.0
    for item in bps:
        bp = item['bp']
        before_price = bp.get('before', {}).get('p', 0.5)
        after_price = bp.get('after', {}).get('p', 0.5)
        true = after_price - before_price
        pred = item['pred_delta']
        total += 1
        if (pred > 0 and true > 0) or (pred < 0 and true < 0):
            correct += 1
        mse += (pred - true) ** 2
        bl += true * true
    if total:
        print(f"Direction accuracy: {correct/total:.1%} ({correct}/{total})")
        print(f"MSE: {mse/total:.4f}  Baseline: {bl/total:.4f}  Ratio: {mse/bl:.2f}")


if __name__ == '__main__':
    main()
