#!/usr/bin/env python3
"""!!! NEEDS v6 PORT !!!

This script is still wired to the old daily_breakpoints / load_flat_samples_as_markets
shape (see collect_bps and the attribute_breakpoint call sites). It will fail to
import until ported to the v6 Record API (use load_records, drop the bp loop —
one record == one example).

Fast end-to-end inference using vLLM for the LLM backbone.

Architecture:
  1. vLLM loads Qwen/Qwen3-8B once in runner='pooling', convert='embed' mode.
     Two LoRA adapters (attributer + forecaster) are registered with
     enable_lora=True, max_loras=2.
  2. Attributer phase: all (bp, news) prompts are sent to vLLM with the
     attributer LoRA. vLLM returns raw last-token hidden states. We apply
     the attributer regression head (a small MLP loaded from the checkpoint)
     in PyTorch on GPU to get attribution scores.
  3. Top-K selection: for each bp, keep the news with the highest attribution
     score(s).
  4. Forecaster phase: build forecaster-style prompts for the top-K news and
     send them to vLLM with the forecaster LoRA. Apply the forecaster
     regression head to get delta predictions. Weighted aggregate (softmax of
     attributer score with temperature) gives per-bp delta.
  5. Write results JSONL in the same format as inference_fast.py.

Why this is much faster than the HF path:
  - vLLM uses FlashAttention + CUDA graphs + prefix caching + continuous
    batching and gets ~450 items/s on Qwen3-8B with raw pooled hidden states,
    vs ~22 items/s through HuggingFace eager + PEFT. ~20x speedup.
  - The regression head is a tiny MLP (~MB) applied in Python; its cost is
    negligible.

Usage:
    python inference_vllm.py \
        --test-data-path ../data/vllm_attributed/combined_test_vllm_attributed.jsonl \
        --forecaster-path ../saves/forecaster_vllm_v17_8b_hlr20/checkpoint-2100 \
        --attributer-path ../saves/prior_attributer_combined_8b/checkpoint-600 \
        --model-name Qwen/Qwen3-8B \
        --output-path ../results/e2e_vllm_attr8b_fc8b_combined.jsonl \
        --max-news-per-bp 50 --top-k 1 --weight-temperature 0.1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

raise NotImplementedError(
    "inference_vllm.py is wired to the old daily_breakpoints / "
    "load_flat_samples_as_markets shape. Port to the v6 Record API "
    "(use load_records, drop the bp loop — one record == one example) "
    "before running."
)
from swm.utils.utils import load_records, set_seed, unix_to_date  # pylint: disable=unreachable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-data-path', required=True)
    p.add_argument('--forecaster-path', required=True)
    p.add_argument('--attributer-path', default=None,
                   help='Learned prior attributer checkpoint. If omitted, use '
                        '--use-precomputed-attributions from the data file.')
    p.add_argument('--use-precomputed-attributions', action='store_true',
                   help='Use the "attributions" field in the data file (e.g. from '
                        'VLLM Qwen3.5-9B posterior attribution) instead of running a '
                        'learned prior attributer.')
    p.add_argument('--min-max-attribution-score', type=float, default=0.0,
                   help='Only evaluate breakpoints whose best precomputed attribution '
                        'score is at least this. Use 0.95 to match v3_combined baseline.')
    p.add_argument('--score-threshold', type=float, default=0.0,
                   help='News-level threshold: drop news with attribution score below '
                        'this. When a bp has NO qualifying news, the forecaster runs '
                        'with a "no news" prompt (history-only prediction). This matches '
                        'the inference_multievent_forecaster.py score_threshold arg and '
                        'reproduces the 65.4% baseline when set to 0.3.')
    p.add_argument('--model-name', default='Qwen/Qwen3-8B')
    p.add_argument('--output-path', required=True)
    p.add_argument('--max-seq-length', type=int, default=512,
                   help='Truncate tokenized prompts to this length (match HF training)')
    p.add_argument('--vllm-max-model-len', type=int, default=1024,
                   help='vLLM max_model_len (must be >= max-seq-length)')
    p.add_argument('--max-news-per-bp', type=int, default=50)
    p.add_argument('--top-k', type=int, default=1)
    p.add_argument('--weight-temperature', type=float, default=0.1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--max-lora-rank', type=int, default=32)
    p.add_argument('--attr-loss-type', default='kl', choices=['kl', 'mse_rank'],
                   help='How to convert attributer logits to scores. Must match the '
                        'attributer checkpoint training: "kl" applies per-bp softmax '
                        'with --attr-target-temperature (distribution); "mse_rank" '
                        'applies sigmoid (independent [0,1] scores).')
    p.add_argument('--attr-target-temperature', type=float, default=0.5,
                   help='Softmax temperature for KL-mode attributer scoring. Lower = '
                        'sharper. Default 0.5 matches BasicPriorAttributer default.')
    return p.parse_args()


def build_regression_head(has_layernorm: bool, hidden_size: int, mid_dim: int) -> nn.Module:
    """Match the architecture in swm.utils.regressor.LLMRegressor.

    The stored state_dict layer indices are 0, 3, 5 in the no-LayerNorm variant
    (Linear, Linear, Linear separated by ReLU + optional Dropout). With
    LayerNorm, the indices shift by 1.
    """
    if has_layernorm:
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    return nn.Sequential(
        nn.Linear(hidden_size, mid_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(mid_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def load_regression_head(checkpoint_dir: str, hidden_size: int, device: torch.device) -> nn.Module:
    """Load regression_head.bin from a training checkpoint.

    Auto-detects the LayerNorm variant (old format had a LayerNorm as the
    first layer; new format doesn't) and the middle dimension (old 256,
    new variants 1024).
    """
    head_path = Path(checkpoint_dir) / 'regression_head.bin'
    state = torch.load(str(head_path), map_location='cpu', weights_only=True)
    first_key = next(iter(state.keys()))
    has_layernorm = state[first_key].dim() == 1
    # Infer mid_dim from the first Linear weight's output dim
    # For no-LN: '0.weight' is Linear(hidden_size, mid_dim), shape (mid_dim, hidden_size)
    # For LN: '1.weight' is that Linear
    linear_key = '1.weight' if has_layernorm else '0.weight'
    mid_dim = state[linear_key].shape[0]
    head = build_regression_head(has_layernorm, hidden_size, mid_dim)
    head.load_state_dict(state)
    head.to(device=device, dtype=torch.float32)
    head.eval()
    print(f"[head] loaded {checkpoint_dir}: has_layernorm={has_layernorm}, "
          f"mid_dim={mid_dim}, params={sum(p.numel() for p in head.parameters()):,}")
    return head


def build_attributer_prompt(market, window_history, target, news):
    """Replica of BasicPriorAttributer._build_prompt_with_news."""
    lines = [f'Prediction Market: {market.question}']
    if market.description:
        desc = market.description[:200] + '...' if len(market.description or '') > 200 else market.description
        lines.append(f'Description: {desc}')
    target_ts = target.get('t')
    history_before = [d for d in window_history if d.get('t') < target_ts]
    lines.append('\nRecent price history:')
    for day in history_before:
        lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")
    target_date = unix_to_date(target['t'])
    lines.append(f'\nPredicting for: {target_date}')
    lines.append('\nNews article:')
    news_date = news.get('published_at', '')
    if news_date:
        lines.append(f'Date: {news_date}')
    lines.append(f'Title: {news.get("title", "")}')
    desc = news.get('description', '') or ''
    if desc:
        lines.append(f'Content: {desc}')
    lines.append(
        '\nDoes this news have a causal relationship with the price change of this prediction market?'
        ' Rate higher only if the news could directly cause the market price to move.'
        ' News that is merely topically related but would not causally drive a price change'
        ' should receive a low score.'
    )
    return '\n'.join(lines)


def build_forecaster_prompt(market, window_history, target, news, max_history_len=None):
    """Replica of MultiEventForecasterDataset._build_prompt (direction=0)."""
    lines = [f'Event: {market.question}']
    if market.description:
        lines.append(f'Description: {market.description}')
    target_ts = target.get('t')
    history_before = [d for d in window_history if d.get('t') < target_ts]
    if max_history_len is not None and len(history_before) > max_history_len:
        history_before = history_before[-max_history_len:]
    lines.append('\nRecent price history:')
    for day in history_before:
        lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")
    target_date = unix_to_date(target['t'])
    lines.append(f'\nNews: {news.get("title", "")}')
    desc = news.get('description', '') or ''
    if desc:
        lines.append(f'{desc}')
    lines.append(f'\nPredict the probability on {target_date}:')
    return '\n'.join(lines)


def build_no_news_prompt(market, window_history, target, max_history_len=None):
    """Replica of MultiEventForecasterDataset._build_no_news_prompt."""
    lines = [f'Event: {market.question}']
    if market.description:
        lines.append(f'Description: {market.description}')
    target_ts = target.get('t')
    history_before = [d for d in window_history if d.get('t') < target_ts]
    if max_history_len is not None and len(history_before) > max_history_len:
        history_before = history_before[-max_history_len:]
    lines.append('\nRecent price history:')
    for day in history_before:
        lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")
    target_date = unix_to_date(target['t'])
    lines.append('\nNews: No relevant news.')
    lines.append(f'\nPredict the probability on {target_date}:')
    return '\n'.join(lines)


def collect_bps(markets, max_news_per_bp, min_max_score=0.0):
    bps = []
    for m in markets:
        if not m.daily_breakpoints:
            continue
        for bp in m.daily_breakpoints:
            news = bp.get('news') or []
            if len(news) < 2:
                continue
            if min_max_score > 0:
                attrs = bp.get('attributions') or []
                scores = [a.get('score') for a in attrs if a.get('score') is not None]
                if not scores or max(scores) < min_max_score:
                    continue
            bps.append({
                'market': m,
                'bp': bp,
                'news': news[:max_news_per_bp],
            })
    return bps


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading markets from {args.test_data_path}...")
    markets = load_flat_samples_as_markets(args.test_data_path)
    if args.limit:
        markets = markets[:args.limit]
    print(f"Loaded {len(markets)} markets")

    from vllm import LLM, PoolingParams
    from vllm.config.pooler import PoolerConfig
    from vllm.lora.request import LoRARequest
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    print(f"Loading {args.model_name} in vLLM pooling/embed mode...")
    t0 = time.time()
    llm = LLM(
        model=args.model_name,
        runner='pooling',
        convert='embed',
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=2,
        pooler_config=PoolerConfig(pooling_type='LAST'),
    )
    print(f"  vLLM ready in {time.time()-t0:.1f}s")

    # HF tokenizer used for pre-tokenization + truncation (matches training)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get hidden size from the model config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model_name)
    hidden_size = hf_config.hidden_size
    print(f"  hidden_size = {hidden_size}")

    fc_head = load_regression_head(args.forecaster_path, hidden_size, device)
    fc_lora = LoRARequest("forecaster", 2, args.forecaster_path)

    attr_head = None
    attr_lora = None
    if not args.use_precomputed_attributions:
        if args.attributer_path is None:
            raise SystemExit("Must pass --attributer-path or --use-precomputed-attributions")
        attr_head = load_regression_head(args.attributer_path, hidden_size, device)
        attr_lora = LoRARequest("attributer", 1, args.attributer_path)

    bps = collect_bps(markets, args.max_news_per_bp,
                       min_max_score=args.min_max_attribution_score)
    print(f"Collected {len(bps)} breakpoints")

    # ---------------- Attributer phase ----------------
    if args.use_precomputed_attributions:
        print(f"\n[attr] using precomputed attributions from data file")
        for item in bps:
            attrs = item['bp'].get('attributions') or []
            scores = [None] * len(item['news'])
            for a in attrs:
                idx = a.get('news_idx')
                s = a.get('score')
                if idx is not None and s is not None and idx < len(scores):
                    scores[idx] = float(s)
            item['scores'] = scores
    else:
        pp = PoolingParams(use_activation=False)
        flat_prompts = []
        flat_keys = []  # (bp_idx, news_idx)
        for i, item in enumerate(bps):
            wh = item['bp'].get('window_history', [])
            tgt = item['bp'].get('after', {})
            for j, n in enumerate(item['news']):
                flat_prompts.append(build_attributer_prompt(item['market'], wh, tgt, n))
                flat_keys.append((i, j))

        print(f"\n[attr] tokenizing + truncating {len(flat_prompts)} prompts to "
              f"{args.max_seq_length} tokens...")
        t0 = time.time()
        enc = tok(
            flat_prompts,
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=True,
            padding=False,
        )
        attr_token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in enc['input_ids']]
        lens = [len(ids) for ids in enc['input_ids']]
        print(f"[attr] tokenized in {time.time()-t0:.1f}s. "
              f"token lens: min={min(lens)} med={sorted(lens)[len(lens)//2]} max={max(lens)}")

        print(f"[attr] running {len(attr_token_prompts)} prompts through vLLM...")
        t0 = time.time()
        outputs = llm.embed(attr_token_prompts, pooling_params=pp,
                             lora_request=attr_lora, use_tqdm=True)
        dt = time.time() - t0
        print(f"[attr] vLLM forward: {dt:.1f}s ({len(attr_token_prompts)/dt:.1f} items/s)")

        t0 = time.time()
        embs = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs])
        embs_t = torch.from_numpy(embs).to(device)
        with torch.no_grad():
            logits_t = attr_head(embs_t).view(-1).cpu().numpy()
        print(f"[attr] regression head applied in {time.time()-t0:.1f}s")

        # Scatter raw logits to per-bp news slots
        for item in bps:
            item['_logits'] = [None] * len(item['news'])
        for flat_idx, (bp_idx, news_idx) in enumerate(flat_keys):
            bps[bp_idx]['_logits'][news_idx] = float(logits_t[flat_idx])

        # Convert per-bp logits to scores (matching BasicPriorAttributer)
        for item in bps:
            logs = [x for x in item['_logits'] if x is not None]
            if not logs:
                item['scores'] = [None] * len(item['news'])
                continue
            if args.attr_loss_type == 'mse_rank':
                # Independent sigmoid(logit) in [0,1]
                import math
                item['scores'] = [
                    (1.0 / (1.0 + math.exp(-x))) if x is not None else None
                    for x in item['_logits']
                ]
            else:
                # KL: softmax(logits / T) across news within this bp
                t = torch.tensor([x if x is not None else -1e9 for x in item['_logits']],
                                  dtype=torch.float32)
                probs = F.softmax(t / args.attr_target_temperature, dim=0)
                item['scores'] = [
                    float(probs[i]) if item['_logits'][i] is not None else None
                    for i in range(len(item['_logits']))
                ]

    pp = PoolingParams(use_activation=False)

    # ---------------- Top-K selection + forecaster pass ----------------
    # A bp can be in one of two modes:
    #   - "news mode": has at least one news with score >= score_threshold.
    #     Pick top-K qualifying news, each gets its own forecaster prompt.
    #   - "no-news mode": no news passes the threshold. Build a single
    #     "no relevant news" prompt and use history-only prediction.
    fc_prompts = []
    fc_keys = []  # (bp_idx, rank, is_no_news)
    for bp_idx, item in enumerate(bps):
        scores = item['scores']
        wh = item['bp'].get('window_history', [])
        tgt = item['bp'].get('after', {})

        # Filter news indices by threshold, then pick top-K by score
        qualifying = [
            (i, s) for i, s in enumerate(scores)
            if s is not None and s >= args.score_threshold
        ]
        qualifying.sort(key=lambda x: x[1], reverse=True)
        ranked = [i for i, _ in qualifying[:args.top_k]]
        item['top_news_indices'] = ranked
        item['is_no_news'] = (len(ranked) == 0)

        if item['is_no_news']:
            # Single no-news prompt per bp
            fc_prompts.append(build_no_news_prompt(item['market'], wh, tgt))
            fc_keys.append((bp_idx, 0, True))
        else:
            for rank, news_idx in enumerate(ranked):
                n = item['news'][news_idx]
                fc_prompts.append(build_forecaster_prompt(item['market'], wh, tgt, n))
                fc_keys.append((bp_idx, rank, False))

    print(f"\n[fc] tokenizing {len(fc_prompts)} prompts...")
    t0 = time.time()
    fc_enc = tok(
        fc_prompts,
        truncation=True,
        max_length=args.max_seq_length,
        add_special_tokens=True,
        padding=False,
    )
    fc_token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in fc_enc['input_ids']]
    print(f"[fc] tokenized in {time.time()-t0:.1f}s")
    print(f"[fc] running {len(fc_token_prompts)} prompts through vLLM (top-{args.top_k})...")
    t0 = time.time()
    fc_outputs = llm.embed(fc_token_prompts, pooling_params=pp,
                            lora_request=fc_lora, use_tqdm=True)
    dt = time.time() - t0
    print(f"[fc] vLLM forward: {dt:.1f}s ({len(fc_token_prompts)/dt:.1f} items/s)")

    t0 = time.time()
    fc_embs = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in fc_outputs])
    fc_embs_t = torch.from_numpy(fc_embs).to(device)
    with torch.no_grad():
        fc_scores_t = fc_head(fc_embs_t).view(-1).cpu().numpy()
    print(f"[fc] regression head applied in {time.time()-t0:.1f}s")

    # Aggregate per-bp pred_delta
    for item in bps:
        if item.get('is_no_news'):
            item['fc_preds'] = [None]  # single slot for no-news prediction
        else:
            item['fc_preds'] = [None] * len(item['top_news_indices'])
    for flat_idx, (bp_idx, rank, _is_nn) in enumerate(fc_keys):
        bps[bp_idx]['fc_preds'][rank] = float(fc_scores_t[flat_idx])

    for item in bps:
        fc_preds = item['fc_preds']
        if item.get('is_no_news'):
            # History-only prediction
            item['pred_delta'] = fc_preds[0] if fc_preds[0] is not None else 0.0
            continue
        ranked = item['top_news_indices']
        if not fc_preds:
            item['pred_delta'] = 0.0
            continue
        raw = torch.tensor([item['scores'][i] or 0.0 for i in ranked], dtype=torch.float32)
        if raw.numel() == 1:
            w = torch.tensor([1.0])
        else:
            w = F.softmax(raw / args.weight_temperature, dim=0)
        p = torch.tensor(fc_preds, dtype=torch.float32)
        item['pred_delta'] = float((w * p).sum())

    # ---------------- Write results ----------------
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as f:
        for item in bps:
            m = item['market']
            bp = item['bp']
            before = bp.get('before', {})
            after = bp.get('after', {})
            bp_price = before.get('p', 0.5)
            ap = after.get('p', 0.5)
            true_delta = ap - bp_price
            ranked = item.get('top_news_indices', [])
            top_news = []
            for idx in ranked:
                n = item['news'][idx].copy()
                n['attribution_score'] = item['scores'][idx]
                top_news.append(n)
            rec = {
                'market_id': m.market_id,
                'event_id': getattr(m, 'event_id', m.market_id),
                't': after.get('t'),
                'question': m.question,
                'description': m.description or '',
                'before_price': bp_price,
                'after_price': ap,
                'true_delta': true_delta,
                'pred_delta': item['pred_delta'],
                'pred_price': bp_price + item['pred_delta'],
                'top_news': top_news,
            }
            f.write(json.dumps(rec) + '\n')
    print(f"\nWrote {len(bps)} results to {args.output_path}")

    # Quick eval — match the historical inference_multievent_forecaster.py convention
    # `(pred > 0) == (true > 0)` so ties (pred == 0 or true == 0) don't penalize.
    correct = total = 0
    mse = bl = 0.0
    for item in bps:
        bp = item['bp']
        bpp = bp.get('before', {}).get('p', 0.5)
        app = bp.get('after', {}).get('p', 0.5)
        true = app - bpp
        pred = item['pred_delta']
        total += 1
        if (pred > 0) == (true > 0):
            correct += 1
        mse += (pred - true) ** 2
        bl += true * true
    if total:
        print(f"Direction accuracy: {correct/total:.1%} ({correct}/{total})")
        print(f"MSE: {mse/total:.4f}  Baseline: {bl/total:.4f}  Ratio: {mse/bl:.2f}")


if __name__ == '__main__':
    main()
