#!/usr/bin/env python3
"""Score all results/eval_polymarket/*.jsonl: corr(pred,true delta), MAE, std.

The winning metric is correlation with the true delta (pm baseline = 0.226).
"""
import glob
import json
import statistics as s
from pathlib import Path

rows = []
for path in sorted(glob.glob("results/eval_polymarket/*.jsonl")):
    tag = Path(path).stem
    preds, trues, aerr = [], [], []
    for line in open(path):
        r = json.loads(line)
        preds.append(r["pred_delta"]); trues.append(r["true_delta"])
        aerr.append(abs(r["price_error"]))
    n = len(preds)
    if n < 2:
        continue
    mp, mt = sum(preds) / n, sum(trues) / n
    sp, st = s.pstdev(preds), s.pstdev(trues)
    cov = sum((p - mp) * (t - mt) for p, t in zip(preds, trues)) / n
    corr = cov / (sp * st) if sp > 0 and st > 0 else float("nan")
    rows.append((tag, n, corr, sum(aerr) / n, sp))

rows.sort(key=lambda x: (x[2] if x[2] == x[2] else -9))  # nan last
print(f"{'tag':22s} {'n':>5s} {'corr':>8s} {'MAE':>8s} {'pred_std':>9s}")
print("-" * 56)
for tag, n, corr, mae, sp in rows:
    print(f"{tag:22s} {n:5d} {corr:8.4f} {mae:8.5f} {sp:9.5f}")
print("\nbaseline: pm corr=0.2259 MAE=0.04283 | clean(failed) corr=-0.008")
