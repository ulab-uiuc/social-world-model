#!/usr/bin/env python3
"""Rebuild the jin10 swm-bench file with DAILY history from fetched raw series.

Run AFTER scripts/fetch_polymarket_daily.py. Replaces each record's 24-point
hourly `history` with 16 DAILY points (24h step) ending at move_hour_t - 1 day,
matching swm-bench format. News / attributions / target are unchanged.
"""
import bisect, json
from pathlib import Path

SRC = Path("data/swmbench_jin10_attributed_filtered_en.jsonl")
HIST = Path("data/jin10_daily_history.json")
DST = Path("data/swmbench_jin10_daily_en.jsonl")

N = 16              # daily points (match swm-bench)
STEP = 86400       # 1 day
END_OFF = 86400    # end at move - 1 day (exclude the move day)
MIN_PTS = 8        # drop records with too little daily history


def sample_daily(ts, ps, move_t):
    if not ts:
        return []
    end = move_t - END_OFF
    targets = [end - i * STEP for i in range(N - 1, -1, -1)]
    out = []
    for t in targets:
        if t < ts[0]:
            continue
        i = bisect.bisect_right(ts, t) - 1
        if i >= 0:
            out.append({"t": t, "p": ps[i]})
    return out


def main():
    hist = json.loads(HIST.read_text())
    series = {m: (sorted([x[0] for x in v]), [x[1] for x in sorted(v)])
              for m, v in hist.items() if v}
    # rebuild sorted-by-t arrays properly
    series = {}
    for m, v in hist.items():
        if not v:
            continue
        v = sorted(v, key=lambda x: x[0])
        series[m] = ([x[0] for x in v], [x[1] for x in v])

    n_in = n_out = n_nohist = n_short = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            n_in += 1
            r = json.loads(line)
            mid = str(r.get("market_id"))
            move_t = r.get("move_hour_t") or (r.get("target") or {}).get("t")
            if mid not in series or not move_t:
                n_nohist += 1
                continue
            ts, ps = series[mid]
            daily = sample_daily(ts, ps, move_t)
            if len(daily) < MIN_PTS:
                n_short += 1
                continue
            r["history"] = daily
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"records: {n_in} -> {n_out}  (no series: {n_nohist}, too short: {n_short})")
    print(f"output: {DST}")


if __name__ == "__main__":
    main()
