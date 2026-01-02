#!/usr/bin/env python3
"""
Step 4: Compute POSTERIOR attributions for breakpoints.

This script uses GPT to score news relevance for each breakpoint.
News must already be embedded in daily_breakpoints (by step3_crawl_breakpoint_news.py).

Features:
    - Incremental writing (saves each market immediately after processing)
    - Resume support (skips already processed markets with --skip_existing)
    - Works for both Polymarket and Kalshi data

Pipeline:
    Step 1/2: converter (raw data → breakpoints with window_history)
    Step 3: crawl_breakpoint_news.py (add news to each breakpoint)
    Step 4: compute_posterior_attributions.py  <-- YOU ARE HERE
            (score each news item → breakpoint.attributions)
    Step 5: split_data.py (split into train/test)

Input breakpoint format:
{
    "before": {"t": ..., "p": ...},
    "after": {"t": ..., "p": ...},
    "window_history": [...],
    "news": [{"title": "...", "description": "..."}, ...]
}

Output breakpoint format (adds attributions):
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
        description='Compute attribution scores for breakpoint news'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input market data with news in breakpoints')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file. Default: input file with _attributed suffix')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model for scoring (default: gpt-4o-mini)')
    parser.add_argument('--max_news', type=int, default=100,
                        help='Max news items to score per breakpoint (default: 100)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for scoring news (default: 10)')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Max retries on LLM parse failure (default: 3)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip markets that already exist in output file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of markets to process (for testing)')
    return parser.parse_args()


def load_processed_market_ids(output_file: str) -> set:
    """Load market IDs that have already been processed."""
    processed_ids = set()
    if Path(output_file).exists():
        try:
            with jsonlines.open(output_file, 'r') as reader:
                for market in reader:
                    market_id = market.get('market_id')
                    if market_id is not None:
                        processed_ids.add(str(market_id))
        except Exception as e:
            print(f"Warning: Error reading existing output file: {e}")
    return processed_ids


def score_news_batch(
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
    """Score a batch of news items (up to 10). Retries on parse failure."""
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
            
            print(scores)
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
                time.sleep(1)  # Brief delay before retry
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


def score_news_relevance(
    client: OpenAI,
    model: str,
    question: str,
    description: str,
    price_before: float,
    price_after: float,
    time_before: float,
    time_after: float,
    news_list: list,
    max_news: int = 100,
    batch_size: int = 10,
    max_retries: int = 3,
) -> list:
    """
    Use LLM to score how relevant each news item is to the price change.
    Processes news in batches of batch_size to avoid token limits.
    
    Returns list of {"news_idx": int, "score": float}
    """
    if not news_list:
        return []
    
    from datetime import datetime
    
    # Limit news to max_news
    news_list = news_list[:max_news]
    
    # Precompute shared context
    price_change = price_after - price_before
    direction = "increased" if price_change > 0 else "decreased"
    date_before = datetime.fromtimestamp(time_before).strftime('%Y-%m-%d %H:%M') if time_before else 'Unknown'
    date_after = datetime.fromtimestamp(time_after).strftime('%Y-%m-%d %H:%M') if time_after else 'Unknown'
    
    # Process in batches
    all_attributions = []
    for batch_start in range(0, len(news_list), batch_size):
        batch_end = min(batch_start + batch_size, len(news_list))
        news_batch = news_list[batch_start:batch_end]
        
        batch_attributions = score_news_batch(
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


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f'Output file: {args.output_file}')
    
    # Load already processed market IDs for resume support
    processed_ids = set()
    if args.skip_existing and Path(args.output_file).exists():
        processed_ids = load_processed_market_ids(args.output_file)
        print(f'Found {len(processed_ids)} already processed markets in output file')
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    with jsonlines.open(args.input_file, 'r') as reader:
        markets = list(reader)
    if args.limit:
        markets = markets[:args.limit]
    print(f"Loaded {len(markets)} markets")
    
    # Count breakpoints with news
    bp_with_news = sum(
        len([bp for bp in m.get('daily_breakpoints', []) if bp.get('news')])
        for m in markets
    )
    print(f"Breakpoints with news: {bp_with_news}")
    
    # Process each market and write incrementally
    attributed_count = 0
    skipped_markets = 0
    
    # Open output file in append mode if resuming
    write_mode = 'a' if processed_ids else 'w'
    
    with jsonlines.open(args.output_file, mode=write_mode) as writer:
        for market in tqdm(markets, desc='Computing attributions'):
            market_id = str(market.get('market_id', ''))
            
            # Skip if already processed
            if market_id in processed_ids:
                skipped_markets += 1
                continue
            
            breakpoints = market.get('daily_breakpoints', [])
            question = market.get('question') or market.get('title', '')
            description = market.get('description', '')
            
            # Process each breakpoint with news
            for bp in breakpoints:
                news_list = bp.get('news', [])
                if not news_list:
                    bp['attributions'] = []
                    continue
                
                # Skip if already has attributions in input data
                if bp.get('attributions'):
                    continue
                
                price_before = bp.get('before', {}).get('p', 0.5)
                price_after = bp.get('after', {}).get('p', 0.5)
                time_before = bp.get('before', {}).get('t')
                time_after = bp.get('after', {}).get('t')
                
                # Score news relevance
                attributions = score_news_relevance(
                    client=client,
                    model=args.model,
                    question=question,
                    description=description,
                    price_before=price_before,
                    price_after=price_after,
                    time_before=time_before,
                    time_after=time_after,
                    news_list=news_list,
                    max_news=args.max_news,
                    batch_size=args.batch_size,
                    max_retries=args.max_retries,
                )
                
                bp['attributions'] = attributions
                attributed_count += 1
                
                if attributed_count % 10 == 0:
                    tqdm.write(f'  Attributed {attributed_count} breakpoints')
            
            # Write market immediately after processing
            writer.write(market)
    
    print(f"\nDone!")
    print(f"  Attributed: {attributed_count} breakpoints")
    print(f"  Skipped markets: {skipped_markets} (already in output)")
    print(f"  Output: {args.output_file}")


if __name__ == '__main__':
    main()
