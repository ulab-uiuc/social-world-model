#!/usr/bin/env python3
"""
Check split JSONL data files.

Usage:
    python check_split_data.py <jsonl_file>
    python check_split_data.py <jsonl_file> --samples 5
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def check_file(filepath: str, num_samples: int = 3):
    """Check and print statistics for a JSONL file."""
    print(f"\n{'='*60}")
    print(f'File: {filepath}')
    print(f"{'='*60}")

    if not Path(filepath).exists():
        print('ERROR: File not found!')
        return

    # Collect statistics
    total = 0
    point_types = Counter()
    window_history_lengths = Counter()
    window_history_by_type = {}  # {point_type: Counter()}
    has_news = 0
    has_attributions = 0
    news_counts = []
    samples = []
    high_score_samples = []  # Samples with attribution >= 0.7

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            total += 1

            # Point type
            ptype = data.get('sample_type', 'unknown')
            point_types[ptype] += 1

            # Window history length
            wh = data.get('window_history', [])
            window_history_lengths[len(wh)] += 1

            # Window history by point type
            if ptype not in window_history_by_type:
                window_history_by_type[ptype] = Counter()
            window_history_by_type[ptype][len(wh)] += 1

            # News
            news = data.get('news', [])
            if news:
                has_news += 1
                news_counts.append(len(news))

            # Attributions
            attributions = data.get('attributions', [])
            if attributions:
                has_attributions += 1
                # Check for high score attributions (>= 0.7)
                for idx, attr in enumerate(attributions):
                    # Handle both formats: dict {'news_idx': i, 'score': s} or float
                    if isinstance(attr, dict):
                        score = attr.get('score')
                        news_idx = attr.get('news_idx', idx)
                    else:
                        score = attr
                        news_idx = idx

                    # Skip if score is None or not a number
                    if score is None or not isinstance(score, (int, float)):
                        continue

                    if score >= 0.7 and news_idx < len(news):
                        high_score_samples.append(
                            {
                                'question': data.get('question', 'N/A'),
                                'sample_type': ptype,
                                'news_title': news[news_idx].get('title', 'N/A'),
                                'news_description': news[news_idx].get(
                                    'description', 'N/A'
                                ),
                                'attribution_score': score,
                            }
                        )

            # Collect samples
            if len(samples) < num_samples:
                samples.append(data)
    # Print statistics
    print('\n📊 Statistics:')
    print(f'  Total samples: {total}')

    print('\n📌 Point types:')
    for ptype, count in point_types.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f'  {ptype}: {count} ({pct:.1f}%)')

    print('\n📏 Window history lengths (all):')
    for length, count in sorted(window_history_lengths.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f'  length={length}: {count} ({pct:.1f}%)')

    # Calculate average window history length
    if window_history_lengths:
        total_len = sum(
            length * count for length, count in window_history_lengths.items()
        )
        avg_len = total_len / total if total > 0 else 0
        print(f'  Average: {avg_len:.2f}')

    # Window history by point type
    for ptype in sorted(window_history_by_type.keys()):
        type_counter = window_history_by_type[ptype]
        type_total = sum(type_counter.values())
        print(f'\n📏 Window history lengths ({ptype}):')
        for length, count in sorted(type_counter.items()):
            pct = count / type_total * 100 if type_total > 0 else 0
            print(f'  length={length}: {count} ({pct:.1f}%)')
        if type_counter:
            type_total_len = sum(
                length * count for length, count in type_counter.items()
            )
            type_avg = type_total_len / type_total if type_total > 0 else 0
            print(f'  Average: {type_avg:.2f}')

    print('\n📰 News:')
    print(
        f'  Samples with news: {has_news} ({has_news/total*100:.1f}%)'
        if total > 0
        else '  N/A'
    )
    if news_counts:
        print(f'  Avg news per sample: {sum(news_counts)/len(news_counts):.1f}')
        print(f'  Min/Max news: {min(news_counts)}/{max(news_counts)}')

    print('\n🎯 Attributions:')
    print(
        f'  Samples with attributions: {has_attributions} ({has_attributions/total*100:.1f}%)'
        if total > 0
        else '  N/A'
    )
    print(f'  High score news (>= 0.7): {len(high_score_samples)}')

    # Print high score samples
    if high_score_samples:
        print('\n🔥 High Attribution Samples (score >= 0.7):')
        for i, item in enumerate(high_score_samples[:20]):  # Limit to 20
            print(
                f"\n  [{i+1}] Score: {item['attribution_score']:.3f} ({item['sample_type']})"
            )
            print(f"      Question: {item['question'][:80]}...")
            print(f"      News: {item['news_title'][:80]}...")
            if item['news_description']:
                print(f"      Desc: {item['news_description'][:100]}...")
        if len(high_score_samples) > 20:
            print(f'\n  ... and {len(high_score_samples) - 20} more high-score samples')


def main():
    parser = argparse.ArgumentParser(description='Check split JSONL data files')
    parser.add_argument('files', nargs='+', help='JSONL file(s) to check')
    parser.add_argument(
        '--samples', type=int, default=3, help='Number of samples to show (default: 3)'
    )
    args = parser.parse_args()

    for filepath in args.files:
        check_file(filepath, args.samples)

    print(f"\n{'='*60}")
    print('Done!')


if __name__ == '__main__':
    main()
