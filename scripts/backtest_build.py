#!/usr/bin/env python3
"""Build a leakage-free, full-natural-universe backtest set from the jin10 daily data.

Spine: data/swmbench_jin10_dailyhist_en.jsonl (daily history already rebuilt from the
2025+2026 price series; history points and news are strictly pre-move, verified).

For a faithful walk-forward we:
  - keep ALL records whose decision time (move_hour_t) is in [--start, --end]
    (NATURAL distribution — NOT the debiased/balanced test subset),
  - replace attributions with a LEAKAGE-FREE HEURISTIC: uniform weight over the
    (up to --max-news most recent) news with published_at <= t. This discards the
    oracle attribution scores (which are outcome-derived) — at trade time we don't
    know which news mattered.
  - default window starts AFTER the model's train cutoff to avoid train/test leakage.

Outputs:
  data/backtest_infer_input.jsonl  — Record-schema lines for backtest_infer.py
  data/backtest_universe.jsonl      — {market_id,event_id,t,entry,move_price,settle,category,z}
"""
import argparse
import json
import datetime as dt
from pathlib import Path

SRC = "data/swmbench_jin10_dailyhist_en.jsonl"


def norm_outcome(o):
    if o is None:
        return None
    o = str(o).strip().lower()
    return 1.0 if o == "yes" else (0.0 if o == "no" else None)


def to_unix(s):
    return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()


def news_ts(n):
    pa = n.get("published_at")
    if not pa:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(pa, fmt).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--start", default="2026-05-24", help="backtest window start (after train cutoff)")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--max-news", type=int, default=8, help="cap news per record (recency); uniform weight")
    ap.add_argument("--require-outcome", action="store_true",
                    help="keep only records with a clean binary outcome (for resolution exit)")
    ap.add_argument("--infer-out", default="data/backtest_infer_input.jsonl")
    ap.add_argument("--univ-out", default="data/backtest_universe.jsonl")
    args = ap.parse_args()

    t0, t1 = to_unix(args.start), to_unix(args.end)
    n_in = n_win = n_out = n_nohist = 0
    fin = open(args.src)
    finfer = open(args.infer_out, "w")
    funiv = open(args.univ_out, "w")
    for line in fin:
        n_in += 1
        r = json.loads(line)
        t = r.get("move_hour_t") or (r.get("target") or {}).get("t")
        if not t or not (t0 <= t <= t1):
            continue
        n_win += 1
        hist = r.get("history") or []
        if len(hist) < 2:
            n_nohist += 1
            continue
        settle = norm_outcome(r.get("outcome"))
        if args.require_outcome and settle is None:
            continue
        news = r.get("news") or []
        # leakage-free heuristic: only news strictly <= t, most-recent max_news, uniform weight
        cand = [(i, news_ts(n)) for i, n in enumerate(news)]
        cand = [(i, ts) for i, ts in cand if ts is not None and ts <= t]
        cand.sort(key=lambda x: -x[1])            # most recent first
        keep = [i for i, _ in cand[: args.max_news]]
        if not keep:
            continue
        attributions = [{"news_idx": i, "score": 1.0} for i in keep]  # uniform, no oracle

        infer_rec = {
            "market_id": str(r.get("market_id")), "event_id": str(r.get("event_id") or r.get("market_id")),
            "question": r.get("question"), "description": r.get("description"),
            "categories": r.get("categories") or [],
            "history": hist, "news": news, "attributions": attributions,
            "target": r.get("target") or {},
        }
        finfer.write(json.dumps(infer_rec, ensure_ascii=False) + "\n")

        entry = hist[-1].get("p")
        funiv.write(json.dumps({
            "market_id": str(r.get("market_id")), "event_id": str(r.get("event_id") or r.get("market_id")),
            "t": t, "entry": entry, "move_price": (r.get("target") or {}).get("p"),
            "settle": settle, "categories": r.get("categories") or [], "z_score": r.get("z_score"),
        }, ensure_ascii=False) + "\n")
        n_out += 1
    fin.close(); finfer.close(); funiv.close()

    with_out = 0
    for line in open(args.univ_out):
        if json.loads(line).get("settle") is not None:
            with_out += 1
    print(f"records: {n_in} total -> {n_win} in window [{args.start}..{args.end}] -> {n_out} usable")
    print(f"  dropped: no/short history {n_nohist}")
    print(f"  with clean binary outcome (resolution-settleable): {with_out}/{n_out}")
    print(f"output: {args.infer_out} , {args.univ_out}")


if __name__ == "__main__":
    main()
