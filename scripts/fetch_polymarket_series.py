#!/usr/bin/env python3
"""Fetch daily price series from Polymarket for markets the bundled files miss.

`polymarket_gap_2025_series.jsonl` and `polymarket_2026_series.jsonl` together
leave two holes that the daily rebuild falls into:

  * a time hole -- the 2025 file stops at 2025-11-30 and the 2026 file starts at
    2026-01-01, so **December 2025 is absent entirely**. 73% of the records whose
    move lands in that month cannot be given daily history, against 1-6% for the
    2026 months;
  * a market hole -- 611 polymarket markets referenced by the benchmark were
    never fetched at all.

Both are recoverable: the CLOB prices-history endpoint returns a market's whole
history in one call, so re-fetching a market fills its December gap as a side
effect of fetching it at all.

    gamma-api.polymarket.com/markets/{id}  -> clobTokenIds + outcomes
    clob.polymarket.com/prices-history?market={token}&fidelity=1440&startTs=1

Resumable: results append to the output file, and a re-run skips markets already
present. Kalshi markets are not on this API and are skipped by the caller.

    python scripts/fetch_polymarket_series.py --ids need_fetch.json \\
        --out polymarket_refetch_series.jsonl --workers 8
"""

import argparse
import concurrent.futures as cf
import json
import threading
import time
from pathlib import Path

import requests

GAMMA = 'https://gamma-api.polymarket.com/markets/{}'
CLOB = 'https://clob.polymarket.com/prices-history'
FIDELITY = 1440  # daily candles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ids', required=True, help='JSON list of market_id strings')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def fetch_one(market_id: str, retries: int):
    """(market_id, question, series) -- series is [] when the market has none."""
    for attempt in range(retries):
        try:
            meta = requests.get(GAMMA.format(market_id), timeout=20).json()
            tokens = meta.get('clobTokenIds')
            outcomes = meta.get('outcomes')
            tokens = json.loads(tokens) if isinstance(tokens, str) else (tokens or [])
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else (outcomes or [])
            if not tokens:
                return market_id, meta.get('question'), []
            # Price the YES side, matching the convention in the bundled files.
            yes = outcomes.index('Yes') if 'Yes' in outcomes else 0
            token = tokens[yes] if yes < len(tokens) else tokens[0]
            hist = requests.get(
                CLOB,
                params={'market': token, 'fidelity': FIDELITY, 'startTs': 1},
                timeout=30,
            ).json()
            series = [
                {'t': int(p['t']), 'p': float(p['p'])}
                for p in (hist.get('history') or [])
            ]
            return market_id, meta.get('question'), series
        except Exception:
            if attempt == retries - 1:
                return market_id, None, None  # None series = fetch failed
            time.sleep(1.5 * (attempt + 1))
    return market_id, None, None


def main():
    args = parse_args()
    ids = json.loads(Path(args.ids).read_text())
    out_path = Path(args.out)

    done = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done.add(str(json.loads(line)['market_id']))
                except Exception:
                    continue
    todo = [m for m in ids if str(m) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f'{len(ids)} requested, {len(done)} already fetched, {len(todo)} to go')

    lock = threading.Lock()
    stats = {'ok': 0, 'empty': 0, 'failed': 0, 'points': 0}
    with out_path.open('a') as fout, cf.ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(fetch_one, str(m), args.retries): m for m in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            mid, question, series = fut.result()
            with lock:
                if series is None:
                    stats['failed'] += 1
                elif not series:
                    stats['empty'] += 1
                else:
                    stats['ok'] += 1
                    stats['points'] += len(series)
                if series:
                    fout.write(json.dumps({
                        'market_id': mid, 'question': question, 'series': series,
                    }, ensure_ascii=False) + '\n')
                if i % 200 == 0:
                    fout.flush()
                    print(f'  {i}/{len(todo)}  ok={stats["ok"]} '
                          f'empty={stats["empty"]} failed={stats["failed"]}', flush=True)

    print(f'\ndone: ok={stats["ok"]} empty={stats["empty"]} failed={stats["failed"]}')
    print(f'{stats["points"]:,} daily points -> {out_path}')


if __name__ == '__main__':
    main()
