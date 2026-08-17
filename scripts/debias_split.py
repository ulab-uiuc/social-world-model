#!/usr/bin/env python3
"""Direction-debiased temporal split of the jin10 attributed (English) set.

The raw jin10 data is ~72% UP (median delta +0.045) in every split, so a model
just learns a "predict a small up-move" prior and beats nothing. Here we:
  1. temporal split by target.t (train 80 / valid 10 / test 10) — no leakage
  2. WITHIN each split, undersample the majority (up) moves to match the number
     of down moves -> ~50/50 direction balance (majority baseline -> 50%).

delta = target.p - history[-1].p  (same as the model's predict-delta target).
Deterministic (fixed seed) so the balancing is reproducible.
"""
import json, random, datetime as dt, sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/swmbench_jin10_attributed_filtered_en.jsonl")
OUT = Path("data")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "jin10_bal"
TRAIN_FRAC, VALID_FRAC = 0.80, 0.10
SEED = 42
rng = random.Random(SEED)


def tkey(r): return (r.get("target") or {}).get("t") or r.get("move_hour_t") or 0
def d(x): return dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")
def delta(r):
    tp = (r.get("target") or {}).get("p"); h = r.get("history") or []
    if tp is None or not h: return None
    return tp - h[-1].get("p")


recs = []
for line in SRC.open():
    r = json.loads(line)
    if not r.get("event_id"): r["event_id"] = str(r.get("market_id"))
    r["market_id"] = str(r.get("market_id")); r["event_id"] = str(r["event_id"])
    dd = delta(r)
    if dd is None: continue
    r["_delta"] = dd
    recs.append(r)

recs.sort(key=tkey)
n = len(recs)
splits = {
    "train": recs[: int(n * TRAIN_FRAC)],
    "valid": recs[int(n * TRAIN_FRAC): int(n * (TRAIN_FRAC + VALID_FRAC))],
    "test":  recs[int(n * (TRAIN_FRAC + VALID_FRAC)):],
}


def balance(rows):
    up = [r for r in rows if r["_delta"] > 0]
    dn = [r for r in rows if r["_delta"] < 0]
    flat = [r for r in rows if r["_delta"] == 0]
    k = min(len(up), len(dn))
    rng.shuffle(up); rng.shuffle(dn)
    kept = up[:k] + dn[:k] + flat          # keep all minority + equal majority + flat
    kept.sort(key=tkey)                    # restore temporal order
    return kept, len(up), len(dn), k


for name, rows in splits.items():
    kept, nu, nd, k = balance(rows)
    for r in kept: r.pop("_delta", None)
    with (OUT / f"{PREFIX}_{name}.jsonl").open("w") as f:
        for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ts = [tkey(r) for r in kept]
    print(f"{name:5s}: {len(rows)} -> {len(kept)}  (up {nu}->{k}, down {nd}->{k})  "
          f"[{d(min(ts))} .. {d(max(ts))}]")
