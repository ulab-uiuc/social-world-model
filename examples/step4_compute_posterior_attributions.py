#!/usr/bin/env python3
"""
Step 4: Compute POSTERIOR attributions for samples (flat format).

This script uses GPT to score news relevance for each sample.
Works with flat format where each line is a single sample (breakpoint or normal_point).

Features:
    - Incremental writing (saves each sample immediately after processing)
    - Resume support (skips already processed samples with --skip_existing)
    - Works for both breakpoint and normal_point samples

Input format (flat):
{
    "market_id": "...",
    "question": "...",
    "sample_type": "breakpoint" | "normal_point",
    "news": [{"title": "...", "description": "..."}, ...],
    ...
}

Output format (adds attributions):
{
    ...,
    "attributions": [
        {"news_idx": 0, "score": 0.8},
        {"news_idx": 1, "score": 0.3},
        ...
    ]
}

Usage:
    python step4_compute_posterior_attributions.py \
        --input_file ../data/processed_kalshi_v2_0102/kalshi_data_processed_with_news.jsonl \
        --skip_existing
"""
import argparse
import json
import os
import sys
from pathlib import Path

import jsonlines
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute attribution scores for sample news (flat format)'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input flat format data with news')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file. Default: input file with _attributed suffix')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model for scoring (default: gpt-4o-mini)')
    parser.add_argument('--max_news', type=int, default=100,
                        help='Max news items to score per sample (default: 100)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for scoring news (default: 10)')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Max retries on LLM parse failure (default: 3)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already exist in output file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    return parser.parse_args()


def get_sample_key(sample: dict) -> str:
    """Generate a unique key for a sample to track processed ones.
    
    Both breakpoint and normal_point now have the same structure with 'before' field.
    """
    sample_type = sample.get('sample_type', '')
    market_id = sample.get('market_id', '')
    
    # Both types now use 'before' timestamp
    before_t = sample.get('before', {}).get('t', 0)
    return f"{market_id}_{sample_type}_{before_t}"


def load_processed_sample_keys(output_file: str) -> set:
    """Load sample keys that have already been processed."""
    processed_keys = set()
    if Path(output_file).exists():
        try:
            with jsonlines.open(output_file, 'r') as reader:
                for sample in reader:
                    key = get_sample_key(sample)
                    processed_keys.add(key)
        except Exception as e:
            print(f"Warning: Error reading existing output file: {e}")
    return processed_keys


def score_news_batch_breakpoint(
    client: OpenAI,
    model: str,
    question: str,
    description: str,
    price_before: float,
    price_after: float,
    date_before: str,
    date_after: str,
    direction: str,
    price_change: float,
    news_batch: list,
    start_idx: int,
    max_retries: int = 3,
) -> list:
    """Score a batch of news items for a BREAKPOINT (price change occurred)."""
    import time
    
    # Format news with published date
    news_items = []
    for i, n in enumerate(news_batch):
        pub_date = n.get('published_at') or n.get('date') or 'Unknown date'
        title = n.get('title', '')
        desc = n.get('description', '')[:150]
        news_items.append(f"[{i}] ({pub_date}) {title} - {desc}")
    news_text = "\n".join(news_items)
    
    prompt = f"""You are performing POSTERIOR ATTRIBUTION: analyzing a prediction market price change that has ALREADY occurred, and determining which news articles can EXPLAIN or are potentially related to this change.

Market Question: {question}
{f'Description: {description}' if description else ''}

OBSERVED Price Change (already happened):
- Before: {price_before:.3f} at {date_before}
- After: {price_after:.3f} at {date_after}
- Change: Price {direction} by {abs(price_change):.3f}

News articles published around this time:
{news_text}

Your task: For each news article, score how well it EXPLAINS or could have CAUSED this price change (0-100):
- 90-100: This news directly explains the price change (e.g., election result announced → election market moves)
- 70-89: Highly relevant - strongly related to the market topic and timing aligns with the change
- 40-69: Moderately relevant - could have contributed to the change
- 10-39: Weakly related - tangentially connected to the topic
- 0-9: Unrelated or published AFTER the price change occurred

Key considerations:
1. Does the news topic directly relate to the market question?
2. Was the news published BEFORE the price change? (News after the change cannot cause it)
3. Would this news logically cause traders to buy/sell, moving the price in the observed direction?
4. Would this news moving the belief of traders in the direction of the price change and cause the price change?

Respond with ONLY a JSON array of {len(news_batch)} scores, e.g.: [85, 30, 0, 45, ...]
Differentiate scores - not all should be the same value.
"""

    return _call_llm_for_scores(client, model, prompt, news_batch, start_idx, max_retries)


def score_news_batch_normal(
    client: OpenAI,
    model: str,
    question: str,
    description: str,
    price: float,
    date: str,
    news_batch: list,
    start_idx: int,
    max_retries: int = 3,
) -> list:
    """Score a batch of news items for a NORMAL POINT (no significant price change)."""
    import time
    
    # Format news with published date
    news_items = []
    for i, n in enumerate(news_batch):
        pub_date = n.get('published_at') or n.get('date') or 'Unknown date'
        title = n.get('title', '')
        desc = n.get('description', '')[:150]
        news_items.append(f"[{i}] ({pub_date}) {title} - {desc}")
    news_text = "\n".join(news_items)
    
    prompt = f"""You are analyzing news relevance for a prediction market on a day where NO significant price change occurred.

Market Question: {question}
{f'Description: {description}' if description else ''}

Date: {date}
Price: {price:.3f} (stable, no significant change)

News articles published around this time:
{news_text}

Your task: For each news article, score how RELEVANT it is to the market topic (0-100):
- 90-100: Directly about the market topic (but apparently did not cause price movement)
- 70-89: Highly relevant to the market topic
- 40-69: Moderately relevant
- 10-39: Weakly related
- 0-9: Unrelated to the market topic

Note: This is a NORMAL day with no price change, so we're measuring topic relevance, not causal impact.

Respond with ONLY a JSON array of {len(news_batch)} scores, e.g.: [85, 30, 0, 45, ...]
Differentiate scores - not all should be the same value.
"""

    return _call_llm_for_scores(client, model, prompt, news_batch, start_idx, max_retries)


def _call_llm_for_scores(
    client: OpenAI,
    model: str,
    prompt: str,
    news_batch: list,
    start_idx: int,
    max_retries: int,
) -> list:
    """Call LLM and parse scores."""
    import time
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            if '[' in content and ']' in content:
                start = content.index('[')
                end = content.rindex(']') + 1
                scores = json.loads(content[start:end])
            else:
                scores = json.loads(content)
            
            # Validate scores count matches news batch
            if len(scores) != len(news_batch):
                raise ValueError(f"Score count mismatch: got {len(scores)}, expected {len(news_batch)}")
            
            # Normalize scores to 0-1 range and add correct index
            return [
                {"news_idx": start_idx + i, "score": max(0, min(100, s)) / 100.0}
                for i, s in enumerate(scores)
            ]
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}/{max_retries}: JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except ValueError as e:
            print(f"  Attempt {attempt + 1}/{max_retries}: Validation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries}: Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    # All retries failed - return None scores
    print(f"  All {max_retries} attempts failed, returning None scores")
    return [{"news_idx": start_idx + i, "score": None} for i in range(len(news_batch))]


def score_breakpoint_news(
    client: OpenAI,
    model: str,
    sample: dict,
    max_news: int = 100,
    batch_size: int = 10,
    max_retries: int = 3,
) -> list:
    """Score news for a breakpoint sample."""
    from datetime import datetime
    
    news_list = sample.get('news', [])
    if not news_list:
        return []
    
    news_list = news_list[:max_news]
    
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
    
    # Process in batches
    all_attributions = []
    for batch_start in range(0, len(news_list), batch_size):
        batch_end = min(batch_start + batch_size, len(news_list))
        news_batch = news_list[batch_start:batch_end]
        
        batch_attributions = score_news_batch_breakpoint(
            client=client,
            model=model,
            question=question,
            description=description,
            price_before=price_before,
            price_after=price_after,
            date_before=date_before,
            date_after=date_after,
            direction=direction,
            price_change=price_change,
            news_batch=news_batch,
            start_idx=batch_start,
            max_retries=max_retries,
        )
        all_attributions.extend(batch_attributions)
    
    return all_attributions


def score_normal_point_news(
    client: OpenAI,
    model: str,
    sample: dict,
    max_news: int = 100,
    batch_size: int = 10,
    max_retries: int = 3,
) -> list:
    """Score news for a normal_point sample.
    
    Normal points now have the same structure as breakpoints.
    """
    from datetime import datetime
    
    news_list = sample.get('news', [])
    if not news_list:
        return []
    
    news_list = news_list[:max_news]
    
    question = sample.get('question', '')
    description = sample.get('description', '')
    
    # Normal points now have same structure as breakpoints
    price = sample.get('after', {}).get('p', 0.5)
    timestamp = sample.get('after', {}).get('t')
    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else 'Unknown'
    
    # Process in batches
    all_attributions = []
    for batch_start in range(0, len(news_list), batch_size):
        batch_end = min(batch_start + batch_size, len(news_list))
        news_batch = news_list[batch_start:batch_end]
        
        batch_attributions = score_news_batch_normal(
            client=client,
            model=model,
            question=question,
            description=description,
            price=price,
            date=date,
            news_batch=news_batch,
            start_idx=batch_start,
            max_retries=max_retries,
        )
        all_attributions.extend(batch_attributions)
    
    return all_attributions


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set default output file
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_attributed{input_path.suffix}")
    
    print(f'Input file: {args.input_file}')
    print(f'Output file: {args.output_file}')
    
    # Load already processed sample keys for resume support
    processed_keys = set()
    if args.skip_existing and Path(args.output_file).exists():
        processed_keys = load_processed_sample_keys(args.output_file)
        print(f'Found {len(processed_keys)} already processed samples in output file')
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    with jsonlines.open(args.input_file, 'r') as reader:
        samples = list(reader)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} samples")
    
    # Count samples with news
    breakpoints_with_news = sum(1 for s in samples if s.get('sample_type') == 'breakpoint' and s.get('news'))
    normal_with_news = sum(1 for s in samples if s.get('sample_type') == 'normal_point' and s.get('news'))
    print(f"Breakpoints with news: {breakpoints_with_news}")
    print(f"Normal points with news: {normal_with_news}")
    
    # Process each sample and write incrementally
    attributed_count = 0
    skipped_count = 0
    
    # Open output file in append mode if resuming
    write_mode = 'a' if processed_keys else 'w'
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(args.output_file, mode=write_mode) as writer:
        for sample in tqdm(samples, desc='Computing attributions'):
            sample_key = get_sample_key(sample)
            
            # Skip if already processed
            if sample_key in processed_keys:
                skipped_count += 1
                continue
            
            news_list = sample.get('news', [])
            sample_type = sample.get('sample_type', '')
            
            # Skip if no news or already has attributions
            if not news_list:
                sample['attributions'] = []
                writer.write(sample)
                continue
            
            if sample.get('attributions'):
                writer.write(sample)
                continue
            
            # Score based on sample type
            if sample_type == 'breakpoint':
                attributions = score_breakpoint_news(
                    client=client,
                    model=args.model,
                    sample=sample,
                    max_news=args.max_news,
                    batch_size=args.batch_size,
                    max_retries=args.max_retries,
                )
            elif sample_type == 'normal_point':
                attributions = score_normal_point_news(
                    client=client,
                    model=args.model,
                    sample=sample,
                    max_news=args.max_news,
                    batch_size=args.batch_size,
                    max_retries=args.max_retries,
                )
            else:
                # Unknown sample type, skip attribution
                attributions = []
            
            sample['attributions'] = attributions
            writer.write(sample)
            attributed_count += 1
            
            if attributed_count % 10 == 0:
                tqdm.write(f'  Attributed {attributed_count} samples')
    
    print(f"\nDone!")
    print(f"  Attributed: {attributed_count} samples")
    print(f"  Skipped: {skipped_count} (already in output)")
    print(f"  Output: {args.output_file}")


if __name__ == '__main__':
    main()
