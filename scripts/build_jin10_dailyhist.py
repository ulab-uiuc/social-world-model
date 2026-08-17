#!/usr/bin/env python3
"""Build a DAILY-history jin10 dataset from the 2025+2026 raw price series.

For each jin10 record, resample 16 DAILY points (24h step) ending at
move_hour_t - 1 day from the merged real price series, matching swm-bench.
Keeps records with >= MIN_PTS daily points. News/attributions/target unchanged.

Note: with daily history, before_price (= history[-1].p) becomes the price ~1
day before the move (a daily-change target), aligning with swm-bench.
"""
import bisect, json, datetime as dt
from collections import defaultdict
from pathlib import Path

SRC = Path("data/swmbench_jin10_attributed_filtered_en.jsonl")
SERIES = ["data/polymarket_gap_2025_series.jsonl", "data/polymarket_2026_series.jsonl"]
DST = Path("data/swmbench_jin10_dailyhist_en.jsonl")
N, STEP, END_OFF, MIN_PTS = 16, 86400, 86400, 8


def main():
    # jin10 polymarket market ids
    jids = set()
    for line in SRC.open():
        r = json.loads(line)
        if r.get("platform") != "kalshi":
            jids.add(str(r.get("market_id")))

    # merge series for jin10 markets from both years
    raw = defaultdict(dict)   # mid -> {t: p}
    for f in SERIES:
        for line in open(f):
            r = json.loads(line)
            mid = str(r.get("market_id"))
            if mid in jids:
                for x in (r.get("series") or []):
                    raw[mid][x["t"]] = x["p"]
    series = {mid: (sorted(d), [d[t] for t in sorted(d)]) for mid, d in raw.items()}
    print(f"series for {len(series)}/{len(jids)} jin10 markets")

    def sample(ts, ps, move_t):
        if not ts:
            return []
        end = move_t - END_OFF
        out = []
        for i in range(N - 1, -1, -1):
            t = end - i * STEP
            if t < ts[0] or t > ts[-1] + STEP:
                continue
            j = bisect.bisect_right(ts, t) - 1
            if j >= 0:
                out.append({"t": t, "p": ps[j]})
        return out

    n_in = n_out = 0
    from collections import Counter
    ptc = Counter()
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            n_in += 1
            r = json.loads(line)
            mid = str(r.get("market_id"))
            move_t = r.get("move_hour_t") or (r.get("target") or {}).get("t")
            if r.get("platform") == "kalshi" or mid not in series or not move_t:
                continue
            ts, ps = series[mid]
            daily = sample(ts, ps, move_t)
            if len(daily) < MIN_PTS:
                continue
            r["history"] = daily
            r.pop("before_2h", None); r.pop("change_2h", None)  # hourly-only fields
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
            ptc[len(daily)] += 1
    print(f"records: {n_in} -> {n_out}")
    print("daily-point counts:", dict(sorted(ptc.items())))
    print(f"output: {DST}")


if __name__ == "__main__":
    main()
