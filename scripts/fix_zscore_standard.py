"""Replace the stored robust z_score (median/MAD) with a STANDARD z-score (mean/std).

Mirrors scripts/reproduce_zscore.py exactly except:
    median(local)        -> mean(local)
    max(MAD*1.4826,0.01) -> max(std(local), 0.01)   # population std (ddof=0)

    series  = sorted(history + [target], by t)
    changes = |p[i+1]-p[i]|
    cur     = changes[-1]
    local   = changes[-1-window : -1]   (causal, window=30)
    z       = (cur - mean(local)) / max(std(local), 0.01)   if len(local)>=10 else 0.0

Usage:
    python scripts/fix_zscore_standard.py <dir>          # rewrite 6 files in place (backup first)
    python scripts/fix_zscore_standard.py <dir> --dry    # just report, no write
"""

import argparse
import json
import os
import shutil
import statistics
from typing import Dict, List

WINDOW = 30
MIN_HISTORY = 10
FLOOR = 0.01
FILES = [
    'test_kalshi.jsonl',
    'test_polymarket.jsonl',
    'train.jsonl',
    'train_with_nonzero_attribution.jsonl',
    'valid_with_nonzero_attribution.jsonl',
    'valid_subset150.jsonl',
]


def standard_zscore(
    history: List[Dict],
    target: Dict,
    window: int = WINDOW,
    min_history: int = MIN_HISTORY,
) -> float:
    series = list(history) + [target]
    if len(series) < 2:
        return 0.0
    series = sorted(series, key=lambda x: x['t'])
    changes = [abs(series[i + 1]['p'] - series[i]['p']) for i in range(len(series) - 1)]
    cur = changes[-1]
    local = changes[max(0, len(changes) - 1 - window) : -1]
    if len(local) < min_history:
        return 0.0
    mean = statistics.fmean(local)
    std = statistics.pstdev(local)  # population std of the reference window
    return (cur - mean) / max(std, FLOOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    bak = os.path.join(args.data_dir, '_zscore_robust_backup')
    if not args.dry:
        os.makedirs(bak, exist_ok=True)

    for fname in FILES:
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f'SKIP (missing): {fname}')
            continue
        rows = [json.loads(l) for l in open(path)]
        n = changed = 0
        old_vals, new_vals = [], []
        for r in rows:
            if 'history' not in r or 'target' not in r:
                continue
            n += 1
            old = r.get('z_score')
            new = standard_zscore(r['history'], r['target'])
            if old is not None:
                old_vals.append(old)
            new_vals.append(new)
            if old is None or abs((old or 0) - new) > 1e-9:
                changed += 1
            r['z_score'] = new
        import numpy as np

        ov = np.array(old_vals) if old_vals else np.array([0.0])
        nv = np.array(new_vals)
        print(
            f'{fname}: n={n} changed={changed} | '
            f'OLD |z|>4:{int((np.abs(ov)>4).sum())} mean|z|={np.abs(ov).mean():.3f} | '
            f'NEW |z|>4:{int((np.abs(nv)>4).sum())} mean|z|={np.abs(nv).mean():.3f}'
        )
        if not args.dry:
            shutil.copy2(path, os.path.join(bak, fname))  # backup original
            with open(path, 'w') as w:
                for r in rows:
                    w.write(json.dumps(r) + '\n')
    if not args.dry:
        print(f'\nOriginals backed up to {bak}')


if __name__ == '__main__':
    main()
