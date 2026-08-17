#!/usr/bin/env python3
"""Convert data/polymarket_cat5_clean.jsonl to swm-bench train schema.

Each breakpoint -> one training record.
"""
import argparse
import json
from pathlib import Path


def convert_bp(market, bp):
    news = [
        {
            "title": (n["content"][:80]),
            "description": n["content"],
            "url": "",
            "published_at": n["time_utc"],
            "source": "jin10",
        }
        for n in bp["news"]
    ]
    attributions = [
        {"news_idx": i, "score": float(n["attribution"]) / 100.0}
        for i, n in enumerate(bp["news"])
    ]
    return {
        "market_id": str(market["market_id"]),
        "event_id": str(market["market_id"]),
        "question": market["question"],
        "description": market.get("event_title") or market["question"],
        "categories": market.get("categories", []),
        "z_score": bp.get("z_score"),
        "news": news,
        "attributions": attributions,
        "history": bp["history_2h"],
        "target": {"t": bp["move_hour_t"], "p": bp["after_p"]},
        "future": [],
        "n_future": 0,
    }


ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/polymarket_cat5_clean.jsonl")
ap.add_argument("--dst", default="data/polymarket_cat5_clean_swmbench.jsonl")
_args = ap.parse_args()
SRC = Path(_args.src)
DST = Path(_args.dst)

n_recs = 0
with SRC.open() as fin, DST.open("w") as fout:
    for line in fin:
        m = json.loads(line)
        for bp in m["breakpoints"]:
            if not bp.get("history_2h"):
                continue  # need at least some pre-jump history
            r = convert_bp(m, bp)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_recs += 1

print(f"wrote {n_recs} records -> {DST}")
