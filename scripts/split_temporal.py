#!/usr/bin/env python3
"""Temporal train/valid/test split of the jin10 attributed (English) swm-bench set.

Splitting by prediction time (target.t) — NOT randomly — so the model always
trains on the PAST and is evaluated on the FUTURE (no lookahead leakage).

  train = earliest 80% by target.t
  valid = next 10%  (subsampled to ~250 for fast in-training eval / best-model)
  test  = latest 10% (full, for final corr/MAE)

Also sanitizes null event_id (-> market_id), which the pydantic Record loader
rejects. Reports the time boundaries and train/test market_id overlap.
"""
import json
import datetime as dt
from pathlib import Path

SRC = Path("data/swmbench_jin10_attributed_filtered_en.jsonl")
OUT = Path("data")
TRAIN_FRAC, VALID_FRAC = 0.80, 0.10   # test = remaining 0.10
VALID_SUBSAMPLE = 250


def tkey(r):
    return (r.get("target") or {}).get("t") or r.get("move_hour_t") or 0


def d(x):
    return dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")


recs = []
for line in SRC.open():
    r = json.loads(line)
    if not r.get("event_id"):
        r["event_id"] = str(r.get("market_id"))
    r["market_id"] = str(r.get("market_id"))
    r["event_id"] = str(r["event_id"])
    if tkey(r) and r.get("history") and (r.get("target") or {}).get("p") is not None:
        recs.append(r)

recs.sort(key=tkey)
n = len(recs)
i_tr = int(n * TRAIN_FRAC)
i_va = int(n * (TRAIN_FRAC + VALID_FRAC))
train, valid, test = recs[:i_tr], recs[i_tr:i_va], recs[i_va:]

# deterministic valid subsample (evenly spaced across the valid time block)
step = max(1, len(valid) // VALID_SUBSAMPLE)
valid_sub = valid[::step][:VALID_SUBSAMPLE]


def dump(rows, path):
    with (OUT / path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


dump(train, "jin10_en_train.jsonl")
dump(valid_sub, "jin10_en_valid.jsonl")
dump(test, "jin10_en_test.jsonl")

tr_mkts = {r["market_id"] for r in train}
te_mkts = {r["market_id"] for r in test}
overlap = tr_mkts & te_mkts

print(f"total usable: {n}")
print(f"train: {len(train):5d}  [{d(tkey(train[0]))} .. {d(tkey(train[-1]))}]")
print(f"valid: {len(valid)} -> sub {len(valid_sub):3d}  [{d(tkey(valid[0]))} .. {d(tkey(valid[-1]))}]")
print(f"test : {len(test):5d}  [{d(tkey(test[0]))} .. {d(tkey(test[-1]))}]")
print(f"train/test market_id overlap: {len(overlap)}/{len(te_mkts)} test markets seen in train "
      f"({100*len(overlap)/max(len(te_mkts),1):.0f}%)")
