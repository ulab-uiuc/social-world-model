#!/usr/bin/env python3
"""
Recompute posterior attributions using VLLM-hosted Qwen3.5-9B.

Key difference from old approach:
- Each news article is analyzed INDIVIDUALLY (not batched with others)
- Parallel async requests for high throughput
- More granular and accurate scores (no batch position bias)

Usage:
    # Start VLLM first:
    CUDA_VISIBLE_DEVICES=9 vllm serve Qwen/Qwen3.5-9B --port 8234

    # Then run:
    python recompute_posterior_vllm.py \
        --input_file /path/to/data.jsonl \
        --output_file /path/to/output.jsonl \
        --vllm_url http://localhost:8234/v1 \
        --concurrency 64
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import jsonlines
from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recompute posterior attributions with VLLM (individual news analysis)'
    )
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8234/v1')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-9B')
    parser.add_argument('--max_news', type=int, default=100,
                        help='Max news items to score per sample')
    parser.add_argument('--concurrency', type=int, default=64,
                        help='Max concurrent requests to VLLM')
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--breakpoints_only', action='store_true',
                        help='Only process breakpoint samples')
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def build_breakpoint_prompt(sample: dict, news: dict) -> str:
    """Build prompt for scoring a SINGLE news article against a breakpoint."""
    question = sample.get('question', '')
    description = sample.get('description', '')
    price_before = sample.get('before', {}).get('p', 0.5)
    price_after = sample.get('after', {}).get('p', 0.5)
    time_before = sample.get('before', {}).get('t')
    time_after = sample.get('after', {}).get('t')
    price_change = price_after - price_before
    direction = "increased" if price_change > 0 else "decreased"
    date_before = datetime.fromtimestamp(time_before).strftime('%Y-%m-%d %H:%M') if time_before else 'Unknown'
    date_after = datetime.fromtimestamp(time_after).strftime('%Y-%m-%d %H:%M') if time_after else 'Unknown'

    pub_date = news.get('published_at') or news.get('date') or 'Unknown date'
    title = news.get('title', '')
    desc = news.get('description', '') or ''

    prompt = f"""You are performing POSTERIOR ATTRIBUTION: analyzing a prediction market price change that has ALREADY occurred, and determining whether a specific news article can EXPLAIN this change.

Market Question: {question}
{f'Description: {description}' if description else ''}

OBSERVED Price Change (already happened):
- Before: {price_before:.3f} at {date_before}
- After: {price_after:.3f} at {date_after}
- Change: Price {direction} by {abs(price_change):.3f}

News article to evaluate:
Published: {pub_date}
Title: {title}
Content: {desc}

Score whether this news has a CAUSAL RELATIONSHIP with the observed price change (0-100):
- 90-100: Directly caused the price change (e.g., election result announced → election market moves)
- 70-89: Strong causal link — this news would logically drive traders to move the price in the observed direction
- 40-69: Possible causal contributor, but indirect
- 10-39: Topically related but would NOT causally drive this specific price change
- 0-9: Unrelated, or published AFTER the price change

IMPORTANT:
- Merely being about the same topic is NOT enough for a high score. The news must have a direct causal mechanism that would move the market price in the observed direction. Topically related news without causal impact should score low (0-39).
- If this news would logically push the price in the OPPOSITE direction of the observed change, it should score 0 — it contradicts rather than explains the change.

Respond with ONLY a single integer score (0-100). No explanation needed."""
    return prompt


def build_normal_prompt(sample: dict, news: dict) -> str:
    """Build prompt for scoring a SINGLE news article against a normal point."""
    question = sample.get('question', '')
    description = sample.get('description', '')
    price = sample.get('after', {}).get('p', 0.5)
    timestamp = sample.get('after', {}).get('t')
    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else 'Unknown'

    pub_date = news.get('published_at') or news.get('date') or 'Unknown date'
    title = news.get('title', '')
    desc = news.get('description', '') or ''

    prompt = f"""You are analyzing news relevance for a prediction market on a day with NO significant price change.

Market Question: {question}
{f'Description: {description}' if description else ''}

Date: {date}
Price: {price:.3f} (stable, no significant change)

News article to evaluate:
Published: {pub_date}
Title: {title}
Content: {desc}

Score whether this news could CAUSALLY affect the market price (0-100):
- 90-100: Would directly cause a price change if the market were actively trading
- 70-89: Strong causal potential — contains new information that would shift trader expectations
- 40-69: Some causal potential, but indirect or uncertain
- 10-39: Topically related but would NOT causally move the market price
- 0-9: Unrelated to the market

IMPORTANT: Merely being about the same topic is NOT enough for a high score. The news must contain information that would causally shift trader expectations. Topically related news without causal impact should score low (0-39).

Respond with ONLY a single integer score (0-100). No explanation needed."""
    return prompt


async def score_single_news(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> float | None:
    """Score a single news article. Returns score in [0, 1] or None on failure."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=20,
                    extra_body={
                        "skip_special_tokens": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            content = response.choices[0].message.content.strip()
            # Extract integer from response (handle "85", "Score: 85", etc.)
            # Find the first number in the response
            import re
            numbers = re.findall(r'\b(\d+)\b', content)
            if numbers:
                score = int(numbers[0])
                return max(0, min(100, score)) / 100.0
            else:
                raise ValueError(f"No number in response: {content!r}")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
            else:
                return None
    return None


async def process_sample(
    client: AsyncOpenAI,
    model: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
    max_news: int = 100,
    max_retries: int = 3,
    breakpoints_only: bool = False,
) -> dict:
    """Process a single sample: score all its news articles individually in parallel."""
    sample_type = sample.get('sample_type', '')
    news_list = sample.get('news', [])[:max_news]

    if not news_list:
        sample['attributions'] = []
        return sample

    if breakpoints_only and sample_type != 'breakpoint':
        # Keep existing attributions or set empty
        if 'attributions' not in sample:
            sample['attributions'] = []
        return sample

    # Build prompts for each news article
    if sample_type == 'breakpoint':
        prompts = [build_breakpoint_prompt(sample, n) for n in news_list]
    elif sample_type == 'normal_point':
        prompts = [build_normal_prompt(sample, n) for n in news_list]
    else:
        sample['attributions'] = []
        return sample

    # Score all news in parallel
    tasks = [
        score_single_news(client, model, p, semaphore, max_retries)
        for p in prompts
    ]
    scores = await asyncio.gather(*tasks)

    # Build attributions
    attributions = []
    for i, score in enumerate(scores):
        attributions.append({
            'news_idx': i,
            'score': score,  # None if failed
        })

    sample['attributions'] = attributions
    return sample


async def main_async():
    args = parse_args()

    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(
            input_path.parent / f"{input_path.stem}_vllm_attributed{input_path.suffix}"
        )

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"VLLM:   {args.vllm_url}")
    print(f"Model:  {args.model}")
    print(f"Concurrency: {args.concurrency}")

    # Load processed keys for resume
    processed_keys = set()
    if args.skip_existing and Path(args.output_file).exists():
        with jsonlines.open(args.output_file) as r:
            for item in r:
                key = f"{item.get('market_id')}_{item.get('sample_type')}_{item.get('before', {}).get('t', 0)}"
                processed_keys.add(key)
        print(f"Resuming: {len(processed_keys)} already processed")

    # Load input data
    print(f"Loading data...")
    with jsonlines.open(args.input_file) as r:
        samples = list(r)
    if args.limit:
        samples = samples[:args.limit]

    # Filter out already processed
    to_process = []
    for s in samples:
        key = f"{s.get('market_id')}_{s.get('sample_type')}_{s.get('before', {}).get('t', 0)}"
        if key not in processed_keys:
            to_process.append(s)

    # Count stats
    bp_count = sum(1 for s in to_process if s.get('sample_type') == 'breakpoint')
    np_count = sum(1 for s in to_process if s.get('sample_type') == 'normal_point')
    total_news = sum(min(len(s.get('news', [])), args.max_news) for s in to_process)
    print(f"To process: {len(to_process)} samples ({bp_count} breakpoints, {np_count} normal)")
    print(f"Total individual LLM calls: {total_news}")

    # Init async client
    client = AsyncOpenAI(base_url=args.vllm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process samples and write incrementally
    write_mode = 'a' if processed_keys else 'w'
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    total_scored = 0
    start_time = time.time()

    with jsonlines.open(args.output_file, mode=write_mode) as writer:
        # Process in chunks to show progress and write incrementally
        chunk_size = 10  # Process 10 samples at a time
        for chunk_start in range(0, len(to_process), chunk_size):
            chunk = to_process[chunk_start:chunk_start + chunk_size]

            tasks = [
                process_sample(
                    client, args.model, s, semaphore,
                    args.max_news, args.max_retries, args.breakpoints_only,
                )
                for s in chunk
            ]
            results = await asyncio.gather(*tasks)

            for result in results:
                writer.write(result)
                completed += 1
                n_scored = sum(1 for a in result.get('attributions', []) if a.get('score') is not None)
                total_scored += n_scored

            elapsed = time.time() - start_time
            rate = total_scored / elapsed if elapsed > 0 else 0
            print(
                f"\r[{completed}/{len(to_process)}] "
                f"{total_scored} news scored, {rate:.1f} scores/sec, "
                f"elapsed {elapsed:.0f}s",
                end='', flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\n\nDone! {completed} samples, {total_scored} news scored in {elapsed:.0f}s")
    print(f"Average: {total_scored/elapsed:.1f} scores/sec")
    print(f"Output: {args.output_file}")


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
