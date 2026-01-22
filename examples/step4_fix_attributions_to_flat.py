#!/usr/bin/env python3
"""
Add attributions to merged flat file.

- For breakpoints: use existing attributions from the attributed file
- For normal_points: set all news attributions to 0 (no price change)

Usage:
    python step4_fix_attributions_to_flat.py \
        --merged_file ../data/kalshi_data_processed_with_news.jsonl \
        --breakpoints_attributed_file ../data/kalshi_data_processed_with_news_attributed.jsonl \
        --output_file ../data/kalshi_data_with_attributions.jsonl
"""
import argparse
from pathlib import Path
from typing import Dict, Optional

import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add attributions to merged flat file'
    )
    parser.add_argument(
        '--merged_file',
        required=True,
        help='Input merged flat file (from step3c_merge_points)',
    )
    parser.add_argument(
        '--breakpoints_attributed_file',
        required=True,
        help='Input file with breakpoints that have attributions (nested format)',
    )
    parser.add_argument(
        '--output_file',
        required=True,
        help='Output flat file with attributions',
    )
    return parser.parse_args()


def build_attribution_lookup(attributed_file: str) -> Dict[str, list]:
    """
    Build a lookup table: (market_id, before_t) -> attributions
    
    This allows us to find attributions for each breakpoint.
    """
    lookup = {}
    
    print(f'Building attribution lookup from {attributed_file}...')
    with jsonlines.open(attributed_file, 'r') as reader:
        for market in tqdm(reader, desc='Loading attributions'):
            market_id = str(market.get('market_id', ''))
            
            for bp in market.get('daily_breakpoints', []):
                before_t = bp.get('before', {}).get('t')
                if before_t is not None:
                    key = f"{market_id}_{before_t}"
                    attributions = bp.get('attributions', [])
                    if attributions:
                        lookup[key] = attributions
    
    print(f'  Found attributions for {len(lookup)} breakpoints')
    return lookup


def get_zero_attributions(news_count: int) -> list:
    """Generate zero attributions for all news items (for normal_points)."""
    return [{"news_idx": i, "score": 0.0} for i in range(news_count)]


def main():
    args = parse_args()
    
    # Build attribution lookup from existing attributed file
    attribution_lookup = build_attribution_lookup(args.breakpoints_attributed_file)
    
    # Process merged flat file and add attributions
    print(f'\nProcessing merged file {args.merged_file}...')
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    bp_count = 0
    bp_with_attr = 0
    np_count = 0
    
    with jsonlines.open(args.merged_file, 'r') as reader:
        samples = list(reader)
    
    print(f'Loaded {len(samples)} samples')
    
    with jsonlines.open(args.output_file, 'w') as writer:
        for sample in tqdm(samples, desc='Adding attributions'):
            sample_type = sample.get('sample_type', '')
            news = sample.get('news', [])
            
            if sample_type == 'breakpoint':
                bp_count += 1
                
                # Look up existing attributions
                market_id = sample.get('market_id', '')
                before_t = sample.get('before', {}).get('t')
                key = f"{market_id}_{before_t}"
                
                attributions = attribution_lookup.get(key, [])
                if attributions:
                    bp_with_attr += 1
                    sample['attributions'] = attributions
                else:
                    # No existing attributions, leave empty or set to None
                    sample['attributions'] = []
            
            elif sample_type == 'normal_point':
                np_count += 1
                
                # Normal points: all news get score 0 (no price change)
                sample['attributions'] = get_zero_attributions(len(news))
            
            writer.write(sample)
    
    # Statistics
    print(f'\nDone!')
    print(f'  Total samples: {len(samples)}')
    print(f'  Breakpoints: {bp_count} ({bp_with_attr} with existing attributions)')
    print(f'  Normal points: {np_count} (all set to 0)')
    print(f'  Output: {args.output_file}')


if __name__ == '__main__':
    main()
