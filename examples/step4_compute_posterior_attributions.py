"""
Precompute POSTERIOR attributions for breakpoints.

This script uses PosteriorAttributer (GPT-4) to score news relevance for each breakpoint.
News must already be embedded in daily_breakpoints (by crawl_breakpoint_news.py).

Pipeline:
    Step 0: converter (raw data → breakpoints with window_history)
    Step 1: crawl_breakpoint_news.py (add news to each breakpoint)
    Step 2: precompute_posterior_attributions.py  <-- YOU ARE HERE
            (score each news item → breakpoint.attributions)
    Step 3: train_multievent_forecaster.py (train using attributed data)

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
    python precompute_posterior_attributions.py \
        --input_file ../data/with_news/train.jsonl \
        --output_file ../data/attributed/train.jsonl \
        --model gpt-4o-mini
"""
import argparse
import json
import os
from pathlib import Path

import jsonlines
from openai import OpenAI
from tqdm import tqdm

from swm.utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute attribution scores for breakpoint news'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input market data with news in breakpoints')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output market data with attributions')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model for scoring (default: gpt-4o-mini)')
    parser.add_argument('--max_news', type=int, default=10,
                        help='Max news items to score per breakpoint (default: 10)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip breakpoints that already have attributions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of markets to process (for testing)')
    return parser.parse_args()


def score_news_relevance(
    client: OpenAI,
    model: str,
    question: str,
    description: str,
    price_before: float,
    price_after: float,
    news_list: list,
) -> list:
    """
    Use LLM to score how relevant each news item is to the price change.
    
    Returns list of {"news_idx": int, "score": float}
    """
    if not news_list:
        return []
    
    # Build prompt
    price_change = price_after - price_before
    direction = "increased" if price_change > 0 else "decreased"
    
    news_text = "\n".join([
        f"[{i}] {n.get('title', '')} - {n.get('description', '')[:200]}"
        for i, n in enumerate(news_list[:10])  # Limit to 10 news
    ])
    
    prompt = f"""You are analyzing a prediction market price change.

Market Question: {question}
{f'Description: {description}' if description else ''}

The market price {direction} from {price_before:.3f} to {price_after:.3f} (change: {abs(price_change):.3f}).

Here are news articles from around that time:
{news_text}

For each news article, rate its relevance to the price change on a scale of 0-100:
- 100: Directly caused the price change
- 50-99: Likely contributed to the change
- 1-49: Possibly related
- 0: Unrelated

Respond with ONLY a JSON array of scores, e.g.: [85, 30, 0, 45, ...]
One score per news article in the same order.
"""

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
        
        # Normalize scores to 0-1 range
        return [
            {"news_idx": i, "score": max(0, min(100, s)) / 100.0}
            for i, s in enumerate(scores)
            if i < len(news_list)
        ]
    except Exception as e:
        print(f"  Error scoring news: {e}")
        # Fallback: equal scores
        return [{"news_idx": i, "score": 0.5} for i in range(len(news_list))]


def main():
    args = parse_args()
    set_seed(args.seed)
    
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
    
    # Process each market
    attributed_count = 0
    skipped_count = 0
    
    for market in tqdm(markets, desc='Computing attributions'):
        breakpoints = market.get('daily_breakpoints', [])
        if not breakpoints:
            continue
        
        question = market.get('question', '')
        description = market.get('description', '')
        
        for bp in breakpoints:
            news_list = bp.get('news', [])
            if not news_list:
                continue
            
            # Skip if already has attributions
            if args.skip_existing and bp.get('attributions'):
                skipped_count += 1
                continue
            
            price_before = bp.get('before', {}).get('p', 0.5)
            price_after = bp.get('after', {}).get('p', 0.5)
            
            # Score news relevance
            attributions = score_news_relevance(
                client=client,
                model=args.model,
                question=question,
                description=description,
                price_before=price_before,
                price_after=price_after,
                news_list=news_list[:args.max_news],
            )
            
            bp['attributions'] = attributions
            attributed_count += 1
    
    # Save output
    print(f"\nSaving to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(markets)
    
    print(f"\nDone!")
    print(f"  Attributed: {attributed_count} breakpoints")
    print(f"  Skipped: {skipped_count} (already had attributions)")
    print(f"  Output: {args.output_file}")


if __name__ == '__main__':
    main()
