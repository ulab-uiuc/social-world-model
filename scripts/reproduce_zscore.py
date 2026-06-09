"""Reproduce the `z_score` field stored in the v6 forecaster datasets.

Each record in data/social-world-model-v6-*-clean-dedup/*.jsonl carries a
`z_score` that measures how anomalous the target move is relative to the recent
history. This script recomputes that value from `history` + `target` alone and
verifies it matches the stored field.

The z-score is a robust z-score (median / MAD) of the *absolute* day-over-day
price change, computed over a causal rolling window of the previous changes:

    series   = history + [target]                       # sorted by time
    changes  = |p[i+1] - p[i]|        for consecutive points
    cur      = changes[-1]            # the target move
    local    = changes[-1-window : -1]  # previous `window` changes (causal)
    median   = median(local)
    mad      = median(|c - median| for c in local)      # median absolute deviation
    scaled   = max(mad * 1.4826, 0.01)                  # 1.4826 -> ~std; floor 0.01
    z_score  = (cur - median) / scaled

If there are fewer than `min_history` local changes, z_score is 0.0.

Mirrors PolyMarketDataConverter._compute_robust_z_score /
find_breakpoints in swm/utils/converter.py.

Usage:
    python scripts/reproduce_zscore.py data/.../train.jsonl
    python scripts/reproduce_zscore.py data/.../train.jsonl --tol 5e-3 --limit 10000
"""

import argparse
import json
import statistics
from typing import Dict, List, Optional


WINDOW = 30        # rolling window size (TimeSeriesConfig.rolling_window)
MIN_HISTORY = 10   # need >= 10 local changes for a meaningful statistic
MAD_TO_STD = 1.4826  # MAD -> std scaling for a normal distribution
MAD_FLOOR = 0.01   # floor on scaled MAD, avoids divide-by-zero / blowup


def robust_zscore(
    history: List[Dict[str, float]],
    target: Dict[str, float],
    window: int = WINDOW,
    min_history: int = MIN_HISTORY,
) -> float:
    """Recompute the robust z-score of the target move from history + target."""
    series = list(history) + [target]
    if len(series) < 2:
        return 0.0

    # Causal: sort by time exactly as the converter does.
    series = sorted(series, key=lambda x: x['t'])
    changes = [
        abs(series[i + 1]['p'] - series[i]['p'])
        for i in range(len(series) - 1)
    ]

    cur = changes[-1]                      # the target change (last consecutive move)
    local = changes[max(0, len(changes) - 1 - window):-1]  # previous `window` changes
    if len(local) < min_history:
        return 0.0

    median = statistics.median(local)
    mad = statistics.median([abs(c - median) for c in local])
    scaled = max(mad * MAD_TO_STD, MAD_FLOOR)
    return (cur - median) / scaled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='Path to a v6 *.jsonl file with z_score records.')
    ap.add_argument('--tol', type=float, default=5e-3,
                    help='Abs tolerance vs stored z_score (default 5e-3, covers rounding).')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only check the first N records.')
    ap.add_argument('--window', type=int, default=WINDOW)
    ap.add_argument('--min-history', type=int, default=MIN_HISTORY)
    ap.add_argument('--show', type=int, default=10,
                    help='Number of mismatches to print (default 10).')
    args = ap.parse_args()

    n = matched = skipped = 0
    worst = 0.0
    mismatches = []

    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stored: Optional[float] = r.get('z_score')
            if stored is None or 'history' not in r or 'target' not in r:
                skipped += 1
                continue
            n += 1
            z = robust_zscore(r['history'], r['target'],
                              window=args.window, min_history=args.min_history)
            diff = abs(z - stored)
            worst = max(worst, diff)
            if diff <= args.tol:
                matched += 1
            elif len(mismatches) < args.show:
                mismatches.append((r.get('market_id'), round(z, 4), stored, round(diff, 4)))
            if args.limit and n >= args.limit:
                break

    print(f'file:      {args.path}')
    print(f'checked:   {n} records ({skipped} skipped: missing z_score/history/target)')
    print(f'matched:   {matched}/{n} within tol={args.tol}')
    print(f'worst diff: {worst:.6f}')
    if mismatches:
        print(f'\nmismatches (market_id, recomputed, stored, diff):')
        for m in mismatches:
            print(f'  {m}')
    else:
        print('\nall records reproduced. ✓')


if __name__ == '__main__':
    main()
