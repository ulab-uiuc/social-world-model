#!/usr/bin/env python3
"""
Fix normal_points structure to match breakpoints structure.

This script reads existing normal_points data (with news already crawled)
and adds missing fields: before, after, window_history, etc.

Usage:
    python step3b_fix_normal_points_structure.py \
        --normal_points_file ../data/kalshi_normal_points_with_news.jsonl \
        --processed_file ../data/kalshi_data_processed.jsonl \
        --output_file ../data/kalshi_normal_points_with_news_fixed.jsonl
"""
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fix normal_points structure to match breakpoints'
    )
    parser.add_argument(
        '--normal_points_file',
        required=True,
        help='Input JSONL file with normal points (old format with timestamp, price, news)',
    )
    parser.add_argument(
        '--processed_file',
        required=True,
        help='Original processed JSONL file with daily_time_series',
    )
    parser.add_argument(
        '--output_file',
        required=True,
        help='Output JSONL file with fixed normal points structure',
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=15,
        help='Window size for window_history (default: 15)',
    )
    return parser.parse_args()


def find_point_in_timeseries(
    daily_ts: List[Dict],
    timestamp: float,
    tolerance: float = 86400,  # 1 day tolerance
) -> Optional[int]:
    """Find the index of a point in daily_time_series by timestamp."""
    for i, point in enumerate(daily_ts):
        ts = point.get('t')
        if ts is not None and abs(ts - timestamp) < tolerance:
            return i
    return None


def build_window_history(
    daily_ts: List[Dict],
    point_idx: int,
    window_size: int,
) -> List[Dict]:
    """Build window_history for a point."""
    window_start_idx = max(0, point_idx - window_size)
    return [
        {'t': daily_ts[j].get('t'), 'p': daily_ts[j].get('p', daily_ts[j].get('yes_price'))}
        for j in range(window_start_idx, point_idx + 1)
    ]


def calculate_z_score(window_history: List[Dict], price_change: float) -> float:
    """Calculate z_score based on rolling std."""
    if len(window_history) < 3:
        return 0
    
    prices = [p['p'] for p in window_history if p['p'] is not None]
    if len(prices) < 3:
        return 0
    
    try:
        std = statistics.stdev(prices[:-1])  # Exclude current point
        return abs(price_change) / std if std > 0 else 0
    except:
        return 0


def fix_normal_point(
    point: Dict,
    daily_ts: List[Dict],
    window_size: int,
) -> Optional[Dict]:
    """Fix a single normal_point to have the same structure as breakpoints."""
    # Get timestamp from old format
    timestamp = point.get('timestamp')
    if timestamp is None:
        return None
    
    # Find the point in daily_time_series
    point_idx = find_point_in_timeseries(daily_ts, timestamp)
    if point_idx is None or point_idx < 1:
        return None
    
    # Get current and previous points
    curr_point = daily_ts[point_idx]
    prev_point = daily_ts[point_idx - 1]
    
    curr_ts = curr_point.get('t')
    prev_ts = prev_point.get('t')
    curr_price = curr_point.get('p', curr_point.get('yes_price'))
    prev_price = prev_point.get('p', prev_point.get('yes_price'))
    
    if None in (curr_ts, prev_ts, curr_price, prev_price):
        return None
    
    # Build window_history
    window_history = build_window_history(daily_ts, point_idx, window_size)
    
    # Calculate price change and z_score
    price_change = curr_price - prev_price
    z_score = calculate_z_score(window_history, price_change)
    
    # Build new structure (same as breakpoints)
    return {
        'before': {'t': prev_ts, 'p': prev_price},
        'after': {'t': curr_ts, 'p': curr_price},
        'change': price_change,
        'z_score': z_score,
        'window_start': window_history[0]['t'] if window_history else prev_ts,
        'window_end': curr_ts,
        'window_history': window_history,
        'news': point.get('news', []),  # Keep existing news
    }


def main():
    args = parse_args()
    
    # Load original processed data to get daily_time_series
    print(f'Loading processed data from {args.processed_file}...')
    processed_map = {}
    with jsonlines.open(args.processed_file, 'r') as reader:
        for market in reader:
            market_id = str(market.get('market_id', ''))
            if market_id:
                processed_map[market_id] = market.get('daily_time_series', [])
    print(f'Loaded time series for {len(processed_map)} markets')
    
    # Load and fix normal points
    print(f'Loading normal points from {args.normal_points_file}...')
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    fixed_count = 0
    skipped_count = 0
    total_points = 0
    
    with jsonlines.open(args.normal_points_file, 'r') as reader:
        markets = list(reader)
    
    print(f'Loaded {len(markets)} markets with normal points')
    
    with jsonlines.open(args.output_file, 'w') as writer:
        for market in tqdm(markets, desc='Fixing normal points'):
            market_id = str(market.get('market_id', ''))
            daily_ts = processed_map.get(market_id, [])
            
            if not daily_ts:
                # No time series data, skip this market
                skipped_count += len(market.get('normal_points', []))
                continue
            
            # Fix each normal point
            fixed_points = []
            for point in market.get('normal_points', []):
                total_points += 1
                fixed_point = fix_normal_point(point, daily_ts, args.window_size)
                if fixed_point:
                    fixed_points.append(fixed_point)
                    fixed_count += 1
                else:
                    skipped_count += 1
            
            # Write result
            result = {
                'market_id': market_id,
                'event_id': market.get('event_id', ''),
                'question': market.get('question', ''),
                'categories': market.get('categories', []),
                'normal_points': fixed_points,
            }
            writer.write(result)
    
    print(f'\nDone!')
    print(f'  Total points: {total_points}')
    print(f'  Fixed: {fixed_count}')
    print(f'  Skipped: {skipped_count}')
    print(f'  Output: {args.output_file}')


if __name__ == '__main__':
    main()
