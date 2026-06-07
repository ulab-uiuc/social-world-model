"""10 forecaster case studies -> markdown. For each selected has-news event:
posterior score + prior(4B nullsub05) score per news (incl. no-news residual),
the news content, and the forecaster's per-news output (run the v9 8B model on
each single-news candidate + the no-news candidate individually).
"""
import json, copy
import numpy as np
from swm.data import Record
from swm.forecaster import MultiEventForecaster

TEST = {
    "kalshi": "data/social-world-model-v6-qwen-235B-clean-dedup/test_kalshi_final.jsonl",
    "poly":   "data/social-world-model-v6-qwen-235B-clean-dedup/test_polymarket_final.jsonl",
}
PRIOR = {
    "kalshi": "/mnt/disk2_from_server2/haofeiy2/swm_prior_attributed/8b_nullsub05/test_kalshi_final_prior_8b_nullsub05.jsonl",
    "poly":   "/mnt/disk2_from_server2/haofeiy2/swm_prior_attributed/8b_nullsub05/test_polymarket_final_prior_8b_nullsub05.jsonl",
}
MODEL = "saves_local/fc8b_v9_pernews_final"
OUT = "results/case_studies_20.md"
N_PER_MARKET = 10

def load_jsonl(p): return [json.loads(l) for l in open(p)]

def prior_map(rows):
    m = {}
    for d in rows:
        k = (d.get("market_id"), (d.get("target") or {}).get("t"))
        n = len(d.get("news") or [])
        m[k] = {a["news_idx"]: float(a.get("score") or 0)
                for a in (d.get("attributions") or []) if 0 <= a.get("news_idx", -1) < n}
    return m

def select(market, n_pick=5):
    """Pick has-news events spread over |true_delta|. Show the FULL news list per
    event (no cap on news count); prefer events with a representative number of
    news (>= the dataset median) so the case studies aren't atypically small."""
    rows = load_jsonl(TEST[market])
    cands = []
    for d in rows:
        news = d.get("news") or []
        pos = [a for a in (d.get("attributions") or []) if 0 <= a.get("news_idx", -1) < len(news) and float(a.get("score") or 0) > 0]
        if not pos or len(news) < 5:   # has-news, with a non-trivial candidate set
            continue
        hist = d.get("history") or []
        bef = hist[-1]["p"] if hist else 0.5
        td = float((d.get("target") or {}).get("p", 0.5)) - bef
        cands.append((abs(td), d))
    cands.sort(key=lambda x: -x[0])
    # spread across the sorted list
    idx = np.linspace(0, len(cands) - 1, n_pick).astype(int)
    return [cands[i][1] for i in idx]

def main():
    pm = {m: prior_map(load_jsonl(PRIOR[m])) for m in TEST}
    picks = []
    for m in ["kalshi", "poly"]:
        for d in select(m, N_PER_MARKET):
            picks.append((m, d))

    # Build per-news + no-news single-candidate records, tagged by unique market_id
    fc = MultiEventForecaster(model_name="Qwen/Qwen3-8B")
    fc.load(MODEL)

    recs = []
    tag_index = {}  # tag -> (pick_idx, news_idx or 'nonews')
    for pi, (mkt, d) in enumerate(picks):
        news = d.get("news") or []
        base = {k: d.get(k) for k in ["question", "description", "categories", "history", "target", "event_id"]}
        # per news
        for i in range(len(news)):
            tag = f"P{pi}__n{i}"
            rd = dict(base); rd["market_id"] = tag; rd["news"] = news
            rd["attributions"] = [{"news_idx": i, "score": 1.0}]
            recs.append(Record.from_dict(rd)); tag_index[tag] = (pi, i)
        # no-news
        tag = f"P{pi}__nonews"
        rd = dict(base); rd["market_id"] = tag; rd["news"] = news; rd["attributions"] = []
        recs.append(Record.from_dict(rd)); tag_index[tag] = (pi, "nonews")

    results = fc.predict(recs, batch_size=16)
    pred = {}
    for r in results:
        pred[r["market_id"]] = r["pred_delta"]

    # Write markdown
    lines = [f"# Forecaster Case Studies ({len(picks)})\n",
             f"- Forecaster: `{MODEL}` (8B per-news, v9)",
             "- Prior attributor: **8B nullsub05**",
             "- Posterior = oracle attribution from the test file; no-news weight = 1 − Σ(scores)",
             "- Forecaster per-news output = model prediction conditioned on that single news (or the no-news prompt)\n"]
    for pi, (mkt, d) in enumerate(picks):
        news = d.get("news") or []
        hist = d.get("history") or []
        bef = hist[-1]["p"] if hist else 0.5
        tp = float((d.get("target") or {}).get("p", 0.5)); td = tp - bef
        k = (d.get("market_id"), (d.get("target") or {}).get("t"))
        post = {a["news_idx"]: float(a.get("score") or 0)
                for a in (d.get("attributions") or []) if 0 <= a.get("news_idx", -1) < len(news)}
        prior = pm[mkt].get(k, {})
        spost, sprior = sum(post.values()), sum(prior.values())

        lines.append(f"\n---\n\n## Case {pi+1} — {mkt.upper()}")
        lines.append(f"**Question:** {d.get('question')}  ")
        if d.get("description"):
            lines.append(f"**Description:** {d.get('description')[:300]}  ")
        lines.append(f"**Category:** {', '.join(d.get('categories') or [])}  ")
        lines.append(f"**z_score:** {d.get('z_score')}  ")
        lines.append(f"**Before price:** {bef:.3f} → **Target:** {tp:.3f}  (**true Δ = {td:+.3f}**)\n")
        lines.append("| # | posterior | prior(8B) | forecaster Δ | news |")
        lines.append("|---|---|---|---|---|")
        for i, nw in enumerate(news):
            title = (nw.get("title") or "").replace("|", "\\|")
            desc = (nw.get("description") or "").replace("|", "\\|")
            content = (title + (" — " + desc if desc else ""))[:180]
            p_i = pred.get(f"P{pi}__n{i}", float("nan"))
            lines.append(f"| {i} | {post.get(i,0):.4f} | {prior.get(i,0):.4f} | {p_i:+.4f} | {content} |")
        # no-news row
        pn = pred.get(f"P{pi}__nonews", float("nan"))
        lines.append(f"| — | {max(0,1-spost):.4f} | {max(0,1-sprior):.4f} | {pn:+.4f} | **NO-NEWS (price unchanged prompt)** |")
        # aggregates
        def wavg(scores):
            tot = sum(scores.get(i, 0) for i in range(len(news)))
            if tot <= 0: return pn
            return sum(scores.get(i, 0) * pred.get(f"P{pi}__n{i}", 0) for i in range(len(news))) / tot
        post_pred = wavg(post)
        prior_norm_pred = wavg(prior)
        soft_pred = min(1.0, sprior) * prior_norm_pred  # 方案A soft routing
        lines.append("")
        lines.append(f"**Aggregated forecaster Δ:**  posterior-weighted = {post_pred:+.4f}  |  "
                     f"prior-weighted(norm) = {prior_norm_pred:+.4f}  |  soft-routing = {soft_pred:+.4f}  "
                     f"(true Δ = {td:+.3f})")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT} with {len(picks)} cases ({len(recs)} per-news predictions)")

if __name__ == "__main__":
    main()
