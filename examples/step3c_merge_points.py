#!/usr/bin/env python3
"""
Merge normal_points (negative samples) and breakpoints (positive samples) into a flat format.

Each line in the output is a single sample point with market identifier.

Usage:
    python step3c_merge_normal_points.py \
        --breakpoints_file ../data/kalshi_data_processed_with_news.jsonl \
        --normal_points_file ../data/kalshi_normal_points_with_news.jsonl \
        --output_file ../data/kalshi_merged_with_news.jsonl \
        --flat_output_file ../data/kalshi_flat_samples.jsonl

Output format (flat) - both breakpoints and normal_points have SAME structure:
{
    "market_id": "...",
    "event_id": "...",
    "question": "...",
    "categories": [...],
    "sample_type": "breakpoint" | "normal_point",
    "before": {"t": ..., "p": ...},
    "after": {"t": ..., "p": ...},
    "change": ...,
    "z_score": ...,
    "window_start": ...,
    "window_end": ...,
    "window_history": [...],
    "news": [...]
}
"""
import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge normal_points into processed data with news'
    )
    parser.add_argument(
        '--breakpoints_file',
        required=True,
        help='Input JSONL file with breakpoints and news (positive samples)',
    )
    parser.add_argument(
        '--normal_points_file',
        required=True,
        help='Input JSONL file with normal points and news (negative samples)',
    )
    parser.add_argument(
        '--output_file',
        default=None,
        help='Output JSONL file with merged data (nested format, optional)',
    )
    parser.add_argument(
        '--flat_output_file',
        default=None,
        help='Output JSONL file with flat format (one sample per line)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load normal points into a dict by market_id
    print(f'Loading normal points from {args.normal_points_file}...')
    normal_points_map = {}
    normal_market_info = {}  # Store market info for normal points
    with jsonlines.open(args.normal_points_file, 'r') as reader:
        for item in reader:
            market_id = str(item.get('market_id', ''))
            if market_id:
                normal_points_map[market_id] = item.get('normal_points', [])
                normal_market_info[market_id] = {
                    'event_id': item.get('event_id', ''),
                    'question': item.get('question', ''),
                    'categories': item.get('categories', []),
                }
    print(f'Loaded normal points for {len(normal_points_map)} markets')
    
    # Load breakpoints data and merge
    print(f'Loading breakpoints from {args.breakpoints_file}...')
    
    merged_count = 0
    total_count = 0
    
    with jsonlines.open(args.breakpoints_file, 'r') as reader:
        markets = list(reader)
    
    print(f'Loaded {len(markets)} markets with breakpoints')
    
    # Prepare flat output
    flat_samples = []
    
    # Prepare nested output writer if needed
    nested_writer = None
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        nested_writer = jsonlines.open(args.output_file, 'w')
    
    for market in tqdm(markets, desc='Merging'):
        total_count += 1
        market_id = str(market.get('market_id', ''))
        
        # Market info for flat format
        market_info = {
            'market_id': market_id,
            'event_id': market.get('event_id', ''),
            'question': market.get('question', ''),
            'categories': market.get('categories', []),
        }
        
        # Add breakpoints as flat samples (only if has news)
        for bp in market.get('daily_breakpoints', []):
            news = bp.get('news', [])
            if not news:  # Skip if no news
                continue
            flat_sample = {
                **market_info,
                'sample_type': 'breakpoint',
                'before': bp.get('before'),
                'after': bp.get('after'),
                'change': bp.get('change'),
                'z_score': bp.get('z_score'),
                'window_start': bp.get('window_start'),
                'window_end': bp.get('window_end'),
                'window_history': bp.get('window_history', []),
                'news': news,
            }
            flat_samples.append(flat_sample)
        
        # Add normal_points if available
        if market_id in normal_points_map:
            market['normal_points'] = normal_points_map[market_id]
            merged_count += 1
            
            # Add normal points as flat samples (only if has news)
            # Normal points now have same structure as breakpoints
            for np in normal_points_map[market_id]:
                news = np.get('news', [])
                if not news:  # Skip if no news
                    continue
                flat_sample = {
                    **market_info,
                    'sample_type': 'normal_point',
                    'before': np.get('before'),
                    'after': np.get('after'),
                    'change': np.get('change'),
                    'z_score': np.get('z_score'),
                    'window_start': np.get('window_start'),
                    'window_end': np.get('window_end'),
                    'window_history': np.get('window_history', []),
                    'news': news,
                }
                flat_samples.append(flat_sample)
        else:
            market['normal_points'] = []
        
        if nested_writer:
            nested_writer.write(market)
    
    if nested_writer:
        nested_writer.close()
    
    # Write flat output
    if args.flat_output_file:
        print(f'\nWriting flat samples to {args.flat_output_file}...')
        Path(args.flat_output_file).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(args.flat_output_file, 'w') as writer:
            for sample in flat_samples:
                writer.write(sample)
    
    # Statistics
    print(f'\nDone!')
    print(f'  Total markets: {total_count}')
    print(f'  Markets with normal_points: {merged_count}')
    print(f'  Markets without normal_points: {total_count - merged_count}')
    if args.output_file:
        print(f'  Nested output: {args.output_file}')
    if args.flat_output_file:
        print(f'  Flat output: {args.flat_output_file}')
    
    # Count samples (all have news now)
    breakpoint_count = sum(1 for s in flat_samples if s['sample_type'] == 'breakpoint')
    normal_count = sum(1 for s in flat_samples if s['sample_type'] == 'normal_point')
    
    print(f'\nSample statistics (only samples with news):')
    print(f'  Breakpoints (positive): {breakpoint_count}')
    print(f'  Normal points (negative): {normal_count}')
    print(f'  Total flat samples: {len(flat_samples)}')


if __name__ == '__main__':
    main()
