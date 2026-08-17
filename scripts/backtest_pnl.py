#!/usr/bin/env python3
"""SWM trading backtest — P&L engine (CPU).

Consumes a predictions jsonl (the schema written by
scripts/inference_multievent_world_model.py: per line at least
`market_id, t, before_price, pred_delta, true_delta, true_price`) and turns the
SWM signal into trades, settles them, and reports P&L vs baselines.

Two exit modes:
  --exit move        settle at the realized post-move price (true_price in the
                     preds file). Needs no external data; works on every row.
  --exit resolution  settle at the market's final outcome (1.0/0.0) joined from
                     data/swmbench_jin10_attributed_filtered_en.jsonl by market_id.
                     Rows without a clean binary outcome are dropped.

Trade rule: signal = pred_delta. |signal| < thr -> no trade; signal>0 -> BUY YES
at entry price p (=before_price); signal<0 -> BUY NO. Held to the exit price s.
  YES pnl per $stake = stake*(s-p)/p - cost*stake
  NO  pnl per $stake = stake*(p-s)/(1-p) - cost*stake
Entry price clamped to [pmin, 1-pmin] for the return denominator.

Outputs (under --out, default results/backtest):
  trades.jsonl, summary.json, equity_curve.csv, report.html

--mock-preds {oracle,zero} overwrites pred_delta (= true_delta, or 0) to validate
the engine's sign/cost handling.

Usage:
  python scripts/backtest_pnl.py --preds results/eval_polymarket/jin10d_bal.jsonl \
      --exit move --label "daily-7B (oracle-attr upper bound)"
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

SWMBENCH = "data/swmbench_jin10_attributed_filtered_en.jsonl"


def load_outcomes(path):
    """market_id -> 1.0/0.0 from the `outcome` field (clean binary only)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.open():
        r = json.loads(line)
        o = r.get("outcome")
        if o is None:
            continue
        o = str(o).strip().lower()
        if o == "yes":
            out[str(r["market_id"])] = 1.0
        elif o == "no":
            out[str(r["market_id"])] = 0.0
    return out


def trade_pnl(direction, p, s, stake, cost, pmin):
    """P&L for a $stake trade entered at prob p, settled at s, given direction."""
    p = min(max(p, pmin), 1 - pmin)
    if direction > 0:      # BUY YES
        ret = (s - p) / p
    else:                  # BUY NO
        ret = (p - s) / (1 - p)
    return stake * ret - cost * stake


def run_strategy(rows, mode, thr, sizing, stake, cost, pmin, scale):
    """mode: 'swm' | 'yes' | 'no' | 'random'. Returns list of trade dicts."""
    import hashlib
    trades = []
    for r in rows:
        sig = r["pred_delta"]
        if mode == "swm":
            if abs(sig) < thr:
                continue
            direction = 1 if sig > 0 else -1
        elif mode == "yes":
            direction = 1
        elif mode == "no":
            direction = -1
        else:  # random, deterministic by (market_id,t)
            h = int(hashlib.md5(f"{r['market_id']}_{r['t']}".encode()).hexdigest(), 16)
            direction = 1 if (h % 2 == 0) else -1
        sz = stake
        if sizing == "conf" and mode == "swm":
            sz = stake * min(1.0, abs(sig) / scale)
        pnl = trade_pnl(direction, r["entry"], r["settle"], sz, cost, pmin)
        trades.append({"t": r["t"], "market_id": r["market_id"], "dir": direction,
                       "entry": r["entry"], "settle": r["settle"], "signal": sig,
                       "stake": sz, "pnl": pnl})
    return trades


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "capital": 0.0, "roi": 0.0,
                "win_rate": float("nan"), "avg_ret": float("nan"),
                "sharpe": float("nan")}
    pnl = sum(t["pnl"] for t in trades)
    cap = sum(t["stake"] for t in trades)
    rets = [t["pnl"] / t["stake"] for t in trades]
    wins = sum(1 for t in trades if t["pnl"] > 0)
    sd = st.pstdev(rets) if len(rets) > 1 else 0.0
    return {"n": len(trades), "pnl": pnl, "capital": cap,
            "roi": pnl / cap if cap else 0.0,
            "win_rate": wins / len(trades),
            "avg_ret": sum(rets) / len(rets),
            "sharpe": (sum(rets) / len(rets) / sd) if sd else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--swmbench", default=SWMBENCH)
    ap.add_argument("--exit", choices=["move", "resolution"], default="move")
    ap.add_argument("--thr", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--headline-thr", type=float, default=0.05)
    ap.add_argument("--sizing", choices=["fixed", "conf"], default="fixed")
    ap.add_argument("--conf-scale", type=float, default=0.10)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--cost", type=float, default=0.02)
    ap.add_argument("--pmin", type=float, default=0.02)
    ap.add_argument("--mock-preds", choices=["none", "oracle", "zero"], default="none")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="results/backtest")
    args = ap.parse_args()

    outcomes = load_outcomes(args.swmbench) if args.exit == "resolution" else {}
    rows = []
    n_in = n_drop = 0
    for line in open(args.preds):
        d = json.loads(line)
        n_in += 1
        pd_ = d.get("pred_delta")
        if args.mock_preds == "oracle":
            pd_ = d.get("true_delta")
        elif args.mock_preds == "zero":
            pd_ = 0.0
        entry = d.get("before_price")
        if entry is None or pd_ is None:
            n_drop += 1
            continue
        if args.exit == "move":
            settle = d.get("true_price")
            if settle is None:
                n_drop += 1
                continue
        else:
            mid = str(d.get("market_id"))
            if mid not in outcomes:
                n_drop += 1
                continue
            settle = outcomes[mid]
        rows.append({"market_id": str(d.get("market_id")), "t": d.get("t") or 0,
                     "entry": float(entry), "settle": float(settle),
                     "pred_delta": float(pd_)})
    rows.sort(key=lambda r: r["t"])
    thrs = [float(x) for x in args.thr.split(",")]

    # threshold sweep for SWM
    sweep = []
    for thr in thrs:
        tr = run_strategy(rows, "swm", thr, args.sizing, args.stake, args.cost, args.pmin, args.conf_scale)
        m = metrics(tr)
        m["thr"] = thr
        sweep.append(m)

    # headline strategies at headline-thr, on the SAME universe (rows passing thr for SWM;
    # baselines evaluated on ALL rows so they represent "trade everything")
    hthr = args.headline_thr
    swm_tr = run_strategy(rows, "swm", hthr, args.sizing, args.stake, args.cost, args.pmin, args.conf_scale)
    yes_tr = run_strategy(rows, "yes", 0, "fixed", args.stake, args.cost, args.pmin, args.conf_scale)
    no_tr = run_strategy(rows, "no", 0, "fixed", args.stake, args.cost, args.pmin, args.conf_scale)
    rnd_tr = run_strategy(rows, "random", 0, "fixed", args.stake, args.cost, args.pmin, args.conf_scale)
    strat = {"SWM": (swm_tr, metrics(swm_tr)), "always-YES": (yes_tr, metrics(yes_tr)),
             "always-NO": (no_tr, metrics(no_tr)), "random": (rnd_tr, metrics(rnd_tr))}

    label = args.label or Path(args.preds).stem
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # trades.jsonl (SWM headline)
    with (outdir / "trades.jsonl").open("w") as f:
        for t in swm_tr:
            f.write(json.dumps(t) + "\n")

    # equity_curve.csv (cumulative pnl over time for each strategy)
    def curve(tr):
        c = 0.0; pts = []
        for t in sorted(tr, key=lambda x: x["t"]):
            c += t["pnl"]; pts.append((t["t"], c))
        return pts
    curves = {k: curve(v[0]) for k, v in strat.items()}
    with (outdir / "equity_curve.csv").open("w") as f:
        f.write("strategy,t,cum_pnl\n")
        for k, pts in curves.items():
            for tt, c in pts:
                f.write(f"{k},{tt},{c:.5f}\n")

    summary = {
        "label": label, "preds": args.preds, "exit": args.exit,
        "mock_preds": args.mock_preds, "cost": args.cost, "sizing": args.sizing,
        "stake": args.stake, "headline_thr": hthr,
        "rows_in": n_in, "rows_used": len(rows), "rows_dropped": n_drop,
        "strategies": {k: v[1] for k, v in strat.items()},
        "swm_threshold_sweep": sweep,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_report(outdir / "report.html", summary, curves, label)

    # console
    print(f"[{label}] exit={args.exit} mock={args.mock_preds} cost={args.cost} "
          f"rows {len(rows)}/{n_in} (dropped {n_drop})")
    print(f"{'strategy':12s}{'n':>6s}{'P&L':>10s}{'ROI':>9s}{'win':>7s}{'Sharpe':>8s}")
    for k, (_, m) in strat.items():
        print(f"{k:12s}{m['n']:6d}{m['pnl']:10.2f}{m['roi']:8.1%}{m['win_rate']:7.1%}{m['sharpe']:8.2f}")
    print("SWM threshold sweep:")
    for m in sweep:
        print(f"  thr={m['thr']:.2f}  n={m['n']:5d}  ROI={m['roi']:7.1%}  win={m['win_rate']:6.1%}  Sharpe={m['sharpe']:6.2f}")
    print(f"-> {outdir}/report.html")


def write_report(path, summary, curves, label):
    # dataviz palette: SWM blue, always-YES orange, others muted/aqua
    colors = {"SWM": "#2a78d6", "always-YES": "#eb6834", "always-NO": "#1baf7a", "random": "#898781"}
    # build SVG equity curve
    allpts = [(t, c) for pts in curves.values() for (t, c) in pts]
    svg = "<p class='muted'>no trades</p>"
    if allpts:
        W, H, m = 820, 320, {"l": 56, "r": 12, "t": 12, "b": 28}
        ts = [t for t, _ in allpts]; cs = [c for _, c in allpts]
        tmin, tmax = min(ts), max(ts); cmin, cmax = min(cs + [0]), max(cs + [0])
        pad = (cmax - cmin) * 0.08 or 1
        cmin -= pad; cmax += pad
        def X(t): return m["l"] + (0 if tmax == tmin else (t - tmin) / (tmax - tmin)) * (W - m["l"] - m["r"])
        def Y(c): return m["t"] + (1 - (0.5 if cmax == cmin else (c - cmin) / (cmax - cmin))) * (H - m["t"] - m["b"])
        parts = [f"<svg viewBox='0 0 {W} {H}'>"]
        # zero line + grid
        for i in range(5):
            c = cmin + (cmax - cmin) * i / 4; y = Y(c)
            parts.append(f"<line x1='{m['l']}' x2='{W-m['r']}' y1='{y:.1f}' y2='{y:.1f}' stroke='var(--grid)'/>")
            parts.append(f"<text x='{m['l']-6}' y='{y+3:.1f}' text-anchor='end' class='ax'>{c:.0f}</text>")
        y0 = Y(0)
        parts.append(f"<line x1='{m['l']}' x2='{W-m['r']}' y1='{y0:.1f}' y2='{y0:.1f}' stroke='var(--axis)' stroke-width='1.5'/>")
        for k, pts in curves.items():
            if not pts:
                continue
            d = " ".join(("M" if i == 0 else "L") + f"{X(t):.1f} {Y(c):.1f}" for i, (t, c) in enumerate(pts))
            parts.append(f"<path d='{d}' fill='none' stroke='{colors.get(k,'#888')}' stroke-width='2'/>")
        parts.append("</svg>")
        svg = "".join(parts)
    tiles = ""
    for k, mt in summary["strategies"].items():
        col = colors.get(k, "#888")
        roi = "–" if mt["n"] == 0 else f"{mt['roi']*100:+.1f}%"
        tiles += (f"<div class='tile'><div class='k' style='color:{col}'>{k}</div>"
                  f"<div class='v'>{roi}</div><div class='s'>n={mt['n']} · win {0 if mt['n']==0 else round(mt['win_rate']*100)}% · Sharpe {mt['sharpe']:.2f}</div></div>")
    legend = " ".join(f"<span><i style='background:{colors.get(k)}'></i>{k}</span>" for k in curves)
    sweep_rows = "".join(
        f"<tr><td>{m['thr']:.2f}</td><td>{m['n']}</td><td>{m['roi']*100:+.1f}%</td>"
        f"<td>{0 if m['n']==0 else round(m['win_rate']*100)}%</td><td>{m['sharpe']:.2f}</td></tr>"
        for m in summary["swm_threshold_sweep"])
    caveat = ("⚠ oracle-attribution upper bound" if summary["mock_preds"] == "none" and "oracle" in label.lower()
              else ("MOCK: " + summary["mock_preds"] if summary["mock_preds"] != "none" else ""))
    html = f"""<!doctype html><meta charset=utf-8><title>Backtest — {label}</title>
<style>
:root{{--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.1)}}
@media(prefers-color-scheme:dark){{:root{{--surface-1:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.1)}}}}
body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:24px}}h1{{font-size:18px;margin:0 0 2px}}.sub{{color:var(--muted);margin:0 0 16px;font-size:13px}}
.tiles{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}}.tile{{border:1px solid var(--border);border-radius:10px;padding:10px 14px;min-width:150px;background:var(--surface-1)}}
.tile .k{{font-size:12px;font-weight:650}}.tile .v{{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}}.tile .s{{font-size:11px;color:var(--muted)}}
.card{{border:1px solid var(--border);border-radius:12px;background:var(--surface-1);padding:16px;margin:14px 0}}
svg{{display:block;width:100%;height:320px}}.ax{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
.legend{{display:flex;gap:16px;font-size:12px;color:var(--ink-2);margin-top:6px}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}}td,th{{border:1px solid var(--border);padding:4px 12px;text-align:right}}th{{color:var(--ink-2)}}
h2{{font-size:13px;color:var(--ink-2);margin:18px 0 6px}}
</style>
<div class=wrap>
<h1>SWM Trading Backtest — {label}</h1>
<p class=sub>exit={summary['exit']} · cost={summary['cost']*100:.0f}% round-trip · sizing={summary['sizing']} · headline thr={summary['headline_thr']} · {summary['rows_used']}/{summary['rows_in']} rows · <b style='color:#d03b3b'>{caveat}</b></p>
<div class=tiles>{tiles}</div>
<div class=card><h2 style='margin-top:0'>Equity curve (cumulative P&amp;L over time)</h2>{svg}
<div class=legend>{legend}</div></div>
<div class=card><h2 style='margin-top:0'>SWM threshold sweep</h2>
<table><tr><th>threshold</th><th>#trades</th><th>ROI</th><th>win%</th><th>Sharpe</th></tr>{sweep_rows}</table></div>
</div>"""
    Path(path).write_text(html)


if __name__ == "__main__":
    main()
