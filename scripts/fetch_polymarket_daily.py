#!/usr/bin/env python3
"""Fetch DAILY price history for the jin10 polymarket markets from raw APIs.

The jin10 swm-bench file only kept a 24h hourly window per market; to build
swm-bench-style DAILY history (16 pts / ~15 days) we must re-fetch the full
series. Reuses the proven two-step path from backend/polymarket_updater.py:
  gamma-api.polymarket.com/markets/{market_id}  -> clobTokenIds + outcomes
  clob.polymarket.com/prices-history?market={token}&fidelity=1440&startTs=1

fidelity=1440 => daily candles; interval defaults to full history.
Resumable: appends to a cache keyed by market_id; re-runs skip done markets.
NOTE: needs network access to *.polymarket.com (this sandbox blocks it).

Output: data/jin10_daily_history.json  {market_id: [[t, p], ...]}  (Yes side)
"""
import json, time, concurrent.futures as cf
from pathlib import Path
import requests

SRC = Path("data/swmbench_jin10_attributed_filtered_en.jsonl")
OUT = Path("data/jin10_daily_history.json")
FIDELITY = 1440            # daily candles
WORKERS = 12
GAMMA = "https://gamma-api.polymarket.com/markets/{}"
CLOB = "https://clob.polymarket.com/prices-history"


def market_ids():
    ids = []
    seen = set()
    for line in SRC.open():
        r = json.loads(line)
        if r.get("platform") == "kalshi":
            continue
        m = str(r.get("market_id"))
        if m and m not in seen:
            seen.add(m); ids.append(m)
    return ids


def fetch_one(mid, retries=3):
    for attempt in range(retries):
        try:
            md = requests.get(GAMMA.format(mid), timeout=20).json()
            toks = md.get("clobTokenIds")
            outs = md.get("outcomes")
            toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
            outs = json.loads(outs) if isinstance(outs, str) else (outs or [])
            if not toks:
                return mid, None
            yes_idx = outs.index("Yes") if "Yes" in outs else 0
            token = toks[yes_idx] if yes_idx < len(toks) else toks[0]
            h = requests.get(
                CLOB, params={"market": token, "fidelity": FIDELITY, "startTs": 1},
                timeout=30,
            ).json()
            series = [[p["t"], p["p"]] for p in h.get("history", [])]
            return mid, series
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return mid, None


def main():
    cache = {}
    if OUT.exists():
        cache = json.loads(OUT.read_text())
    todo = [m for m in market_ids() if m not in cache]
    print(f"markets: {len(cache)} cached, {len(todo)} to fetch")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for mid, series in ex.map(fetch_one, todo):
            cache[mid] = series or []
            done += 1
            if done % 100 == 0:
                OUT.write_text(json.dumps(cache))
                print(f"  {done}/{len(todo)}", flush=True)
    OUT.write_text(json.dumps(cache))
    ok = sum(1 for v in cache.values() if v)
    print(f"done: {ok}/{len(cache)} markets have series -> {OUT}")


if __name__ == "__main__":
    main()
