#!/usr/bin/env python3
"""Rebuild jin10 records with genuinely DAILY price history, and say what it cost.

`swmbench_jin10_attributed_filtered_en_daily.jsonl` on the Hub is misnamed: its
`history` is the original 24-point HOURLY window (consecutive points ~3600s
apart, last point 1h before the move), not daily. Anything that loads it
expecting a daily-history file silently gets hourly data.

This script builds the real thing from the raw price series and reports the
coverage honestly, because the shortfall is not a bug to fix:

    daily points available  ->  age of the market at the move
       1                          1.5 days
       4                          4.6 days
       7                          7.5 days
      16                         32.2 days

A market that opened five days before its breakpoint cannot have sixteen days of
daily history. The `--min-points` cut is therefore a real trade-off between
record count and history length, not a threshold to tune away.

Usage:
    python scripts/build_jin10_daily.py \\
        --src swmbench_jin10_attributed_filtered_en.jsonl \\
        --series polymarket_gap_2025_series.jsonl polymarket_2026_series.jsonl \\
        --out swmbench_jin10_daily_en.jsonl --min-points 8
"""

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

DAY = 86400
N_POINTS = 16  # daily points to sample, matching swm-bench
END_OFFSET = DAY  # last point sits one day before the move


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True,
                   help='swmbench_jin10_attributed_filtered_en.jsonl (the superset)')
    p.add_argument('--series', nargs='+', required=True,
                   help='polymarket_*_series.jsonl files')
    p.add_argument('--out', required=True)
    p.add_argument('--min-points', type=int, default=8,
                   help='drop records with fewer daily points than this')
    p.add_argument('--n-points', type=int, default=N_POINTS)
    p.add_argument('--annotate-from', default=None,
                   help='optional file carrying before_2h / change_2h / '
                        'direction_consistent to copy across')
    p.add_argument('--report', default=None, help='write the coverage report as JSON')
    return p.parse_args()


def move_time(record):
    return record.get('move_hour_t') or (record.get('target') or {}).get('t')


def load_series(paths, wanted):
    """market_id -> (sorted timestamps, prices), restricted to `wanted`."""
    merged = defaultdict(dict)
    for path in paths:
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                mid = str(row.get('market_id'))
                if mid not in wanted:
                    continue
                for point in row.get('series') or []:
                    merged[mid][int(point['t'])] = float(point['p'])
    out = {}
    for mid, bucket in merged.items():
        times = sorted(bucket)
        out[mid] = (times, [bucket[t] for t in times])
    return out


def sample_daily(times, prices, move_t, n_points):
    """`n_points` daily samples ending one day before the move.

    Each sample carries the last price observed at or before its timestamp, so a
    sample is only emitted where the series actually covers that day.
    """
    if not times:
        return []
    end = move_t - END_OFFSET
    out = []
    for i in range(n_points - 1, -1, -1):
        t = end - i * DAY
        if t < times[0] or t > times[-1] + DAY:
            continue
        j = bisect.bisect_right(times, t) - 1
        if j >= 0:
            out.append({'t': t, 'p': prices[j]})
    return out


def main():
    args = parse_args()
    records = [json.loads(line) for line in open(args.src) if line.strip()]
    wanted = {str(r['market_id']) for r in records}
    print(f'source: {len(records)} records / {len(wanted)} markets')

    series = load_series(args.series, wanted)
    print(f'price series found for {len(series)}/{len(wanted)} markets '
          f'({100 * len(series) / max(len(wanted), 1):.0f}%)')

    extra = {}
    if args.annotate_from:
        for line in open(args.annotate_from):
            r = json.loads(line)
            key = (str(r['market_id']), move_time(r))
            extra[key] = {
                k: r[k] for k in ('before_2h', 'change_2h', 'direction_consistent')
                if k in r
            }
        print(f'annotations available for {len(extra)} records')

    reason = Counter()
    point_hist = Counter()
    kept_at = Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0

    with out_path.open('w') as fout:
        for record in records:
            mid = str(record['market_id'])
            move_t = move_time(record)
            if not move_t:
                reason['no move timestamp'] += 1
                continue
            if mid not in series:
                reason['no price series'] += 1
                continue
            times, prices = series[mid]
            daily = sample_daily(times, prices, move_t, args.n_points)
            point_hist[len(daily)] += 1
            for threshold in (2, 4, 6, 8, 10, args.n_points):
                kept_at[threshold] += len(daily) >= threshold
            if not daily:
                reason['series does not cover the move'] += 1
                continue
            if len(daily) < args.min_points:
                reason[f'market younger than {args.min_points} days'] += 1
                continue

            record['history'] = daily
            record['history_step_seconds'] = DAY
            record['n_history'] = len(daily)
            record.update(extra.get((mid, move_t), {}))
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            n_out += 1

    print(f'\nwrote {n_out} records -> {out_path}')
    print('\ndropped:')
    for k, v in reason.most_common():
        print(f'  {k:38s} {v:6d}  ({100 * v / len(records):5.1f}%)')
    print('\nrecords retained at other --min-points values:')
    for threshold in sorted(kept_at):
        print(f'  >= {threshold:2d} points  {kept_at[threshold]:6d}  '
              f'({100 * kept_at[threshold] / len(records):5.1f}%)')

    # The whole point of this script is that the output is daily. Prove it.
    gaps = Counter()
    lengths = Counter()
    with out_path.open() as fh:
        for line in fh:
            hist = json.loads(line)['history']
            lengths[len(hist)] += 1
            for a, b in zip(hist, hist[1:]):
                gaps[b['t'] - a['t']] += 1
    total = sum(gaps.values())
    exact = gaps.get(DAY, 0)
    print(f'\nverification: {exact}/{total} consecutive gaps are exactly {DAY}s '
          f'({100 * exact / max(total, 1):.1f}%)')
    if exact != total:
        raise SystemExit('FAILED: output history is not uniformly daily')
    print(f'history lengths: {dict(sorted(lengths.items()))}')

    if args.report:
        Path(args.report).write_text(json.dumps({
            'source': args.src,
            'source_records': len(records),
            'source_markets': len(wanted),
            'markets_with_series': len(series),
            'written': n_out,
            'min_points': args.min_points,
            'dropped': dict(reason),
            'daily_point_histogram': dict(sorted(point_hist.items())),
            'retained_at_threshold': dict(sorted(kept_at.items())),
        }, indent=2))
        print(f'report -> {args.report}')


if __name__ == '__main__':
    main()
