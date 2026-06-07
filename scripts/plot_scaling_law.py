"""Per-news forecaster scaling law: model size (0.6B/4B/8B) vs end-to-end
metrics (with prior attributor). Reads prediction jsonls, computes attributed
(posterior-nonzero) + all metrics, plots size vs corr/MASE/DA, saves PNG + table.

Usage: pass triples 'size_billions:pred_jsonl:orig_jsonl' for each (model,testset).
We build curves per test set (kalshi, polymarket).
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# config: model size (params in B) -> {kalshi: pred_path, poly: pred_path}
SIZES = [
    (0.6, "results/scaling/kalshi_06b.jsonl", "results/scaling/poly_06b.jsonl"),
    (4.0, "results/scaling/kalshi_4b.jsonl",  "results/scaling/poly_4b.jsonl"),
    (8.0, "results/fc8b/kalshi_prior_v9.jsonl", "results/fc8b/poly_prior_v9.jsonl"),
]
ORIG = {
    "kalshi": "data/social-world-model-v6-qwen-235B-clean-dedup/test_kalshi_final.jsonl",
    "poly":   "data/social-world-model-v6-qwen-235B-clean-dedup/test_polymarket_final.jsonl",
}

def attr_keys(orig):
    ak = set()
    for line in open(orig):
        d = json.loads(line); news = d.get("news") or []
        if any(0 <= x.get("news_idx", -1) < len(news) and float(x.get("score") or 0) > 0
               for x in (d.get("attributions") or [])):
            ak.add((d.get("market_id"), (d.get("target") or {}).get("t")))
    return ak

def metrics(pred_path, orig):
    ak = attr_keys(orig)
    rows = [json.loads(l) for l in open(pred_path)]
    pred = np.array([r["pred_delta"] for r in rows])
    true = np.array([r["true_delta"] for r in rows])
    am = np.array([(r.get("market_id"), r.get("t")) in ak for r in rows])
    def m(p, t):
        err = p - t; mae = np.mean(np.abs(err)); rmse = np.sqrt(np.mean(err**2))
        mase = mae / np.mean(np.abs(t)) if np.mean(np.abs(t)) > 0 else float("nan")
        mv = np.abs(t) > 1e-6
        da = np.mean((p[mv] > 0) == (t[mv] > 0)) if mv.any() else float("nan")
        c = np.corrcoef(p, t)[0, 1] if len(p) > 1 else float("nan")
        return dict(MASE=mase, MAE=mae, RMSE=rmse, DA=da, Corr=c, n=len(p))
    return {"attr": m(pred[am], true[am]), "all": m(pred, true)}

def main():
    data = {}  # (testset, subset, metric) -> [(size, val)]
    table = []
    for size, kp, pp in SIZES:
        for ts, path in [("kalshi", kp), ("poly", pp)]:
            try:
                r = metrics(path, ORIG[ts])
            except FileNotFoundError:
                print(f"SKIP {size}B {ts}: {path} not found"); continue
            for subset in ["attr", "all"]:
                for met in ["MASE", "MAE", "RMSE", "DA", "Corr"]:
                    data.setdefault((ts, subset, met), []).append((size, r[subset][met]))
                table.append((size, ts, subset, r[subset]))

    # print table
    print(f"\n{'size':>5} {'set':>7} {'subset':>5} {'MASE':>7} {'MAE':>7} {'RMSE':>7} {'DA':>6} {'Corr':>7} {'n':>5}")
    for size, ts, subset, r in table:
        print(f"{size:>5} {ts:>7} {subset:>5} {r['MASE']:>7.3f} {r['MAE']:>7.3f} {r['RMSE']:>7.3f} {r['DA']:>6.3f} {r['Corr']:>7.3f} {r['n']:>5}")

    # plot: 2 rows (kalshi, poly) x 3 cols (Corr, MASE, DA), attr+all curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ri, ts in enumerate(["kalshi", "poly"]):
        for ci, met in enumerate(["Corr", "MASE", "DA"]):
            ax = axes[ri][ci]
            for subset, color in [("attr", "C0"), ("all", "C1")]:
                pts = sorted(data.get((ts, subset, met), []))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=color, label=subset)
            ax.set_xscale("log"); ax.set_xticks([0.6, 4, 8]); ax.set_xticklabels(["0.6B","4B","8B"])
            ax.set_title(f"{ts} — {met}"); ax.set_xlabel("model size"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = "results/scaling/scaling_law_pernews.png"
    import os; os.makedirs("results/scaling", exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nSaved plot -> {out}")

if __name__ == "__main__":
    main()
