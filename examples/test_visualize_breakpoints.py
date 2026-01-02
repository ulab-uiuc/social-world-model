"""
Visualization script for checking breakpoint detection in processed market data.
Displays daily time-series with detected breakpoints highlighted.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def timestamp_to_datetime(ts: float) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(ts)


def plot_market(market: Dict[str, Any], save_dir: Path = None, show: bool = True):
    """
    Plot daily time-series with breakpoints highlighted.
    
    Args:
        market: Market data dict with daily_time_series and daily_breakpoints
        save_dir: Directory to save the plot (optional)
        show: Whether to display the plot
    """
    daily_series = market.get('daily_time_series', [])
    breakpoints = market.get('daily_breakpoints', [])
    question = market.get('question', 'Unknown')
    market_id = market.get('market_id', 'unknown')
    categories = market.get('categories', [])
    
    if not daily_series:
        print(f"No daily time series data for market: {market_id}")
        return
    
    # Sort by timestamp
    daily_series = sorted(daily_series, key=lambda x: x['t'])
    
    # Extract data
    timestamps = [timestamp_to_datetime(p['t']) for p in daily_series]
    prices = [p['p'] for p in daily_series]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot time series
    ax.plot(timestamps, prices, 'b-', linewidth=1.5, label='Price', alpha=0.8)
    ax.scatter(timestamps, prices, c='blue', s=20, alpha=0.6, zorder=5)
    
    # Highlight breakpoints
    if breakpoints:
        bp_before_times = []
        bp_after_times = []
        bp_before_prices = []
        bp_after_prices = []
        
        for bp in breakpoints:
            before = bp['before']
            after = bp['after']
            z_score = bp.get('z_score', 0)
            change = bp.get('change', 0)
            
            before_time = timestamp_to_datetime(before['t'])
            after_time = timestamp_to_datetime(after['t'])
            
            bp_before_times.append(before_time)
            bp_after_times.append(after_time)
            bp_before_prices.append(before['p'])
            bp_after_prices.append(after['p'])
            
            # Draw red line connecting breakpoint pair
            ax.plot([before_time, after_time], [before['p'], after['p']], 
                    'r-', linewidth=2.5, alpha=0.8, zorder=10)
            
            # Add annotation with z-score
            mid_time = before_time + (after_time - before_time) / 2
            mid_price = (before['p'] + after['p']) / 2
            direction = '↑' if after['p'] > before['p'] else '↓'
            ax.annotate(f'{direction} z={z_score:.1f}', 
                        xy=(mid_time, mid_price),
                        xytext=(0, 15 if after['p'] > before['p'] else -15),
                        textcoords='offset points',
                        fontsize=8, color='red', fontweight='bold',
                        ha='center', va='bottom' if after['p'] > before['p'] else 'top')
        
        # Mark breakpoint nodes
        ax.scatter(bp_before_times, bp_before_prices, c='red', s=80, 
                   marker='o', edgecolors='darkred', linewidths=1.5, 
                   label='Breakpoint Start', zorder=15)
        ax.scatter(bp_after_times, bp_after_prices, c='orange', s=80, 
                   marker='s', edgecolors='darkorange', linewidths=1.5,
                   label='Breakpoint End', zorder=15)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (Probability)', fontsize=11)
    ax.set_ylim(0, 1.05)
    
    # Title with question (truncated if too long)
    title_question = question[:80] + '...' if len(question) > 80 else question
    cat_str = ', '.join(categories) if categories else 'N/A'
    ax.set_title(f'{title_question}\n[{market_id}] Categories: {cat_str}', fontsize=10)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    # Add breakpoint count info
    bp_count = len(breakpoints)
    ax.text(0.98, 0.02, f'Breakpoints: {bp_count}', transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_id = market_id.replace('/', '_').replace('\\', '_')
        save_path = save_dir / f'{safe_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize market time-series with breakpoints'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='../data/processed_polymarket_v2_0102/polymarket_data_processed.jsonl',
        help='Path to processed JSONL file'
    )
    parser.add_argument(
        '--market_id',
        type=str,
        default=None,
        help='Specific market ID to visualize (optional)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save plots (optional)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of random samples to visualize (if no market_id specified)'
    )
    parser.add_argument(
        '--only_with_breakpoints',
        action='store_true',
        help='Only show markets that have breakpoints'
    )
    parser.add_argument(
        '--min_breakpoints',
        type=int,
        default=0,
        help='Minimum number of breakpoints to filter'
    )
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display plots (useful for batch saving)'
    )
    args = parser.parse_args()
    
    # Load data
    print(f'Loading data from: {args.input_file}')
    records = load_jsonl(args.input_file)
    print(f'Loaded {len(records)} markets')
    
    # Filter markets
    if args.only_with_breakpoints or args.min_breakpoints > 0:
        min_bp = max(1, args.min_breakpoints)
        records = [r for r in records if len(r.get('daily_breakpoints', [])) >= min_bp]
        print(f'Filtered to {len(records)} markets with >= {min_bp} breakpoints')
    
    # Select markets to visualize
    if args.market_id:
        # Find specific market
        markets = [r for r in records if r.get('market_id') == args.market_id]
        if not markets:
            print(f"Market ID '{args.market_id}' not found!")
            return
    else:
        # Random sample
        if len(records) > args.num_samples:
            indices = np.random.choice(len(records), args.num_samples, replace=False)
            markets = [records[i] for i in indices]
        else:
            markets = records
    
    # Visualize
    print(f'\nVisualizing {len(markets)} markets...\n')
    for i, market in enumerate(markets, 1):
        print(f'[{i}/{len(markets)}] {market.get("market_id", "unknown")}')
        print(f'  Question: {market.get("question", "N/A")[:60]}...')
        print(f'  Breakpoints: {len(market.get("daily_breakpoints", []))}')
        print(f'  Daily points: {len(market.get("daily_time_series", []))}')
        plot_market(market, save_dir=args.save_dir, show=not args.no_show)
        print()


if __name__ == '__main__':
    main()

