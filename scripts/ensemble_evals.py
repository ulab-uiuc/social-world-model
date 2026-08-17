#!/usr/bin/env python3
"""Ensemble world-model predictions by averaging pred_delta across runs.

Best result: e_cdpm + pm -> corr 0.2360 (vs 0.2278 best single, 0.2259 pm).
Usage: python scripts/ensemble_evals.py e_cdpm pm  [-> writes results/eval_polymarket/ensemble.jsonl]
"""
import json
import statistics as s
import sys
from pathlib import Path

tags = sys.argv[1:] or ["e_cdpm", "pm"]


def load(tag):
    d = {}
    for line in open(f"results/eval_polymarket/{tag}.jsonl"):
        r = json.loads(line)
        d[(r["market_id"], r["t"])] = r
    return d


data = {t: load(t) for t in tags}
keys = set(data[tags[0]])
for t in tags[1:]:
    keys &= set(data[t])
keys = sorted(keys)

rows = []
for k in keys:
    recs = [data[t][k] for t in tags]
    pd = sum(r["pred_delta"] for r in recs) / len(recs)
    base = recs[0]
    bp = base["before_price"]
    rows.append(
        {**base, "pred_delta": pd, "pred_price": bp + pd,
         "delta_error": pd - base["true_delta"],
         "price_error": (bp + pd) - base["true_price"]}
    )

out = Path("results/eval_polymarket/ensemble.jsonl")
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

preds = [r["pred_delta"] for r in rows]
trues = [data[tags[0]][k]["true_delta"] for k in keys]
n = len(preds)
mp, mt = sum(preds) / n, sum(trues) / n
cov = sum((p - mp) * (t - mt) for p, t in zip(preds, trues)) / n
corr = cov / (s.pstdev(preds) * s.pstdev(trues))
mae = sum(abs(r["price_error"]) for r in rows) / n
print(f"ensemble({'+'.join(tags)}) n={n} corr={corr:.4f} MAE={mae:.5f} -> {out}")
