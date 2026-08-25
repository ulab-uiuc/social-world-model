#!/usr/bin/env python3
"""Render the backtest JSON into a self-contained HTML report."""

import argparse
import datetime as dt
import json
import math
from pathlib import Path

# ---------------------------------------------------------------- formatting

def pct(x, digits=1, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '&mdash;'
    fmt = f'{{:+.{digits}%}}' if sign else f'{{:.{digits}%}}'
    return fmt.format(x).replace('%', '<span class="unit">%</span>')


def num(x, digits=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '&mdash;'
    return (f'{{:+.{digits}f}}' if sign else f'{{:.{digits}f}}').format(x)


def cents(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '&mdash;'
    return f'{x * 100:+.2f}<span class="unit">c</span>'


def day(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime('%b&nbsp;%-d')


def cls_for(x, good=0.0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    return 'gain' if x > good else ('loss' if x < good else '')


# ---------------------------------------------------------------- svg charts

def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="auto" role="img" '
            f'aria-label="{label}" preserveAspectRatio="xMidYMid meet">')


def _thin(points, limit=400):
    """Downsample a curve to `limit` points, always keeping the last one.

    A grid curve carries one point per trade -- tens of thousands of them. At
    chart width that is far more path data than pixels, and it dominates the
    page weight.
    """
    if len(points) <= limit:
        return points
    step = len(points) / limit
    kept = [points[int(i * step)] for i in range(limit)]
    if kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def equity_chart(series, w=920, h=320):
    """Cumulative return on deployed capital for each named strategy."""
    pad_l, pad_r, pad_t, pad_b = 58, 116, 18, 34
    series = {k: _thin(v) for k, v in series.items()}
    pts = [p for s in series.values() for p in s]
    if not pts:
        return ''
    xs = [p['t'] for p in pts]
    ys = [p['equity'] - 1 for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(min(ys), 0), max(max(ys), 0)
    span = (y1 - y0) or 1
    y0 -= span * 0.08
    y1 += span * 0.08

    def px(t):
        return pad_l + (t - x0) / max(x1 - x0, 1) * (w - pad_l - pad_r)

    def py(v):
        return pad_t + (y1 - v) / (y1 - y0) * (h - pad_t - pad_b)

    out = [svg_open(w, h, 'Cumulative return on deployed capital over the test month')]
    # horizontal grid at round percentages
    step = 0.1 if (y1 - y0) < 0.9 else 0.25
    v = math.floor(y0 / step) * step
    while v <= y1:
        y = py(v)
        out.append(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>')
        v += step
    out.append(f'<line class="zero" x1="{pad_l}" x2="{w - pad_r}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    for t in (x0, (x0 + x1) // 2, x1):
        out.append(f'<text class="axis" x="{px(t):.1f}" y="{h - 12}" text-anchor="middle">{day(t)}</text>')

    for i, (name, pts_) in enumerate(series.items()):
        if not pts_:
            continue
        d = ' '.join(
            f'{"M" if j == 0 else "L"}{px(p["t"]):.1f},{py(p["equity"] - 1):.1f}'
            for j, p in enumerate(pts_)
        )
        out.append(f'<path class="line s{i}" d="{d}"/>')
        last = pts_[-1]
        ly = py(last['equity'] - 1)
        out.append(f'<circle class="dot s{i}" cx="{px(last["t"]):.1f}" cy="{ly:.1f}" r="3.5"/>')
        out.append(
            f'<text class="lbl s{i}" x="{w - pad_r + 10}" y="{ly + 4:.1f}">{name} '
            f'{(last["equity"] - 1) * 100:+.0f}%</text>'
        )
    out.append('</svg>')
    return '\n'.join(out)


def bar_pair_chart(rows, w=920, h=300, ylabel='ROI'):
    """Grouped bars: model vs always-YES across buckets."""
    if not rows:
        return ''
    pad_l, pad_r, pad_t, pad_b = 58, 16, 18, 52
    vals = [v for r in rows for v in (r['a'], r['b'])]
    y1 = max(max(vals), 0)
    y0 = min(min(vals), 0)
    span = (y1 - y0) or 1
    y1 += span * 0.12
    y0 -= span * 0.12
    bw = (w - pad_l - pad_r) / len(rows)

    def py(v):
        return pad_t + (y1 - v) / (y1 - y0) * (h - pad_t - pad_b)

    out = [svg_open(w, h, f'{ylabel} by bucket, model versus always-YES')]
    step = 0.25 if span > 0.9 else 0.1
    v = math.ceil(y0 / step) * step
    while v <= y1:
        y = py(v)
        out.append(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>')
        v += step
    zero = py(0)
    out.append(f'<line class="zero" x1="{pad_l}" x2="{w - pad_r}" y1="{zero:.1f}" y2="{zero:.1f}"/>')
    for i, r in enumerate(rows):
        cx = pad_l + i * bw
        for k, (key, klass) in enumerate((('a', 'model'), ('b', 'base'))):
            val = r[key]
            x = cx + bw * (0.16 + 0.34 * k)
            y = min(py(val), zero)
            hh = abs(py(val) - zero)
            out.append(f'<rect class="bar {klass}" x="{x:.1f}" y="{y:.1f}" '
                       f'width="{bw * 0.3:.1f}" height="{max(hh, 1):.1f}" rx="2"/>')
        out.append(f'<text class="axis" x="{cx + bw / 2:.1f}" y="{h - 30}" text-anchor="middle">{r["label"]}</text>')
        out.append(f'<text class="axis dim" x="{cx + bw / 2:.1f}" y="{h - 14}" text-anchor="middle">n={r["n"]}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def drift_chart(rows, w=440, h=250):
    """ROI as a function of how much of the move has already leaked into the book."""
    if not rows:
        return ''
    pad_l, pad_r, pad_t, pad_b = 54, 18, 18, 40
    ys = [r['roi'] for r in rows]
    y1, y0 = max(max(ys), 0), min(min(ys), 0)
    span = (y1 - y0) or 1
    y1 += span * 0.15
    y0 -= span * 0.15

    def px(i):
        return pad_l + i / max(len(rows) - 1, 1) * (w - pad_l - pad_r)

    def py(v):
        return pad_t + (y1 - v) / (y1 - y0) * (h - pad_t - pad_b)

    out = [svg_open(w, h, 'Return versus assumed execution slippage')]
    out.append(f'<line class="zero" x1="{pad_l}" x2="{w - pad_r}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    step = 0.1 if span < 0.9 else 0.25
    v = math.ceil(y0 / step) * step
    while v <= y1:
        y = py(v)
        out.append(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>')
        v += step
    d = ' '.join(f'{"M" if i == 0 else "L"}{px(i):.1f},{py(r["roi"]):.1f}' for i, r in enumerate(rows))
    out.append(f'<path class="line s0" d="{d}"/>')
    for i, r in enumerate(rows):
        out.append(f'<circle class="dot s0" cx="{px(i):.1f}" cy="{py(r["roi"]):.1f}" r="4"/>')
        out.append(f'<text class="axis" x="{px(i):.1f}" y="{h - 14}" text-anchor="middle">{r["label"]}</text>')
    out.append('</svg>')
    return '\n'.join(out)


# ---------------------------------------------------------------------- css

CSS = """
:root {
  --ground:#F1F5F6; --surface:#FFFFFF; --raised:#F7FAFA;
  --ink:#12212A; --body:#2C3D45; --muted:#5E727B; --faint:#8FA1A8;
  --rule:#D6E0E3; --rule-soft:#E6EDEF;
  --accent:#1A5C64; --accent-soft:#E2EFF0; --gold:#9A6E1C;
  --gain:#1F7350; --loss:#A83535; --gain-soft:#E3F1EA; --loss-soft:#F7E5E5;
  --shadow:0 1px 2px rgba(18,33,42,.05), 0 8px 24px -12px rgba(18,33,42,.16);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0C1518; --surface:#141F23; --raised:#1A272C;
    --ink:#E7F0F2; --body:#C3D2D6; --muted:#8DA2A9; --faint:#63787F;
    --rule:#25353A; --rule-soft:#1D2B30;
    --accent:#5FB3BC; --accent-soft:#123336; --gold:#D2A44E;
    --gain:#4FBC8C; --loss:#E37070; --gain-soft:#102E23; --loss-soft:#33191A;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"] {
  --ground:#0C1518; --surface:#141F23; --raised:#1A272C;
  --ink:#E7F0F2; --body:#C3D2D6; --muted:#8DA2A9; --faint:#63787F;
  --rule:#25353A; --rule-soft:#1D2B30;
  --accent:#5FB3BC; --accent-soft:#123336; --gold:#D2A44E;
  --gain:#4FBC8C; --loss:#E37070; --gain-soft:#102E23; --loss-soft:#33191A;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.7);
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--body);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  font-size:16.5px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1120px; margin:0 auto; padding:0 28px 96px; }
.col { max-width:70ch; }
h1,h2,h3 { font-family:Newsreader,Georgia,"Times New Roman",serif; color:var(--ink);
           text-wrap:balance; font-weight:600; letter-spacing:-.012em; }
h1 { font-size:clamp(2.1rem,4.4vw,3.1rem); line-height:1.08; margin:0 0 .5rem; letter-spacing:-.025em; }
h2 { font-size:clamp(1.45rem,2.4vw,1.85rem); line-height:1.2; margin:0 0 .35rem; }
h3 { font-size:1.08rem; font-family:"IBM Plex Sans",sans-serif; font-weight:600;
     letter-spacing:.005em; margin:0 0 .3rem; }
p { margin:0 0 1.05rem; }
a { color:var(--accent); text-underline-offset:3px; }
strong { color:var(--ink); font-weight:600; }
.mono { font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }
.unit { font-size:.78em; opacity:.62; margin-left:.5px; }

/* ---- masthead ---- */
header.top { border-bottom:1px solid var(--rule); background:var(--surface); }
header.top .wrap { padding-top:52px; padding-bottom:36px; }
.eyebrow { font-family:"IBM Plex Mono",monospace; font-size:.7rem; letter-spacing:.16em;
           text-transform:uppercase; color:var(--accent); margin:0 0 1.1rem; }
.standfirst { font-size:1.16rem; color:var(--muted); max-width:62ch; margin:.3rem 0 0; }
.byline { margin-top:1.6rem; font-family:"IBM Plex Mono",monospace; font-size:.74rem;
          color:var(--faint); letter-spacing:.03em; display:flex; flex-wrap:wrap; gap:.4rem 1.4rem; }

/* ---- verdict ---- */
.verdict { display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
           gap:1px; background:var(--rule); border:1px solid var(--rule);
           border-radius:10px; overflow:hidden; margin:2.6rem 0 0; box-shadow:var(--shadow); }
.vcell { background:var(--surface); padding:20px 22px 22px; }
.vcell .k { font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.13em;
            text-transform:uppercase; color:var(--faint); margin-bottom:.55rem; }
.vcell .v { font-family:Newsreader,serif; font-size:2.15rem; line-height:1;
            color:var(--ink); font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.vcell .v.gain { color:var(--gain); } .vcell .v.loss { color:var(--loss); }
.vcell .n { font-size:.85rem; color:var(--muted); margin-top:.5rem; line-height:1.45; }

section { margin-top:4.4rem; }
section > .col > p:last-child { margin-bottom:0; }
.kicker { font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.15em;
          text-transform:uppercase; color:var(--faint); margin:0 0 .55rem; }
.lede { color:var(--muted); font-size:1.02rem; margin:.15rem 0 1.6rem; max-width:64ch; }

/* ---- panels & tables ---- */
.panel { background:var(--surface); border:1px solid var(--rule); border-radius:10px;
         box-shadow:var(--shadow); margin:1.7rem 0; overflow:hidden; }
.panel > figcaption, .panel > .cap { padding:16px 20px; border-bottom:1px solid var(--rule-soft);
         font-size:.9rem; color:var(--muted); background:var(--raised); }
.panel > .cap strong { color:var(--ink); }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.9rem;
        font-variant-numeric:tabular-nums; }
th,td { padding:9px 14px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--rule-soft); }
th { font-family:"IBM Plex Mono",monospace; font-size:.66rem; letter-spacing:.1em;
     text-transform:uppercase; color:var(--faint); font-weight:500;
     border-bottom:1px solid var(--rule); background:var(--raised); position:sticky; top:0; }
td:first-child, th:first-child { text-align:left; }
tbody tr:last-child td { border-bottom:none; }
tr.hl td { background:var(--accent-soft); }
tr.hl td:first-child { box-shadow:inset 3px 0 0 var(--accent); font-weight:600; color:var(--ink); }
tr.sep td { border-top:1px solid var(--rule); }
.gain { color:var(--gain); } .loss { color:var(--loss); }
.tag { font-family:"IBM Plex Mono",monospace; font-size:.62rem; letter-spacing:.08em;
       text-transform:uppercase; padding:2px 7px; border-radius:99px; border:1px solid var(--rule);
       color:var(--muted); background:var(--raised); }
.tag.model { border-color:var(--accent); color:var(--accent); background:var(--accent-soft); }
.tag.leak  { border-color:var(--gold); color:var(--gold); }
td .sub { display:block; font-size:.74rem; color:var(--faint); font-family:"IBM Plex Mono",monospace; }

/* ---- charts ---- */
figure { margin:0; }
svg .grid { stroke:var(--rule-soft); stroke-width:1; }
svg .zero { stroke:var(--rule); stroke-width:1.5; }
svg .axis { fill:var(--faint); font-family:"IBM Plex Mono",monospace; font-size:11px; }
svg .axis.dim { fill:var(--faint); opacity:.6; font-size:10px; }
svg .line { fill:none; stroke-width:2.2; stroke-linejoin:round; stroke-linecap:round; }
svg .lbl { font-family:"IBM Plex Mono",monospace; font-size:11.5px; }
svg .s0 { stroke:var(--accent); } svg .s0.dot,svg .s0.lbl { fill:var(--accent); stroke:none; }
svg .s1 { stroke:var(--gold); }   svg .s1.dot,svg .s1.lbl { fill:var(--gold); stroke:none; }
svg .s2 { stroke:var(--faint); stroke-dasharray:5 4; }
svg .s2.dot,svg .s2.lbl { fill:var(--faint); stroke:none; }
svg .bar.model { fill:var(--accent); } svg .bar.base { fill:var(--faint); opacity:.5; }
.legend { display:flex; gap:1.3rem; flex-wrap:wrap; padding:12px 20px 16px;
          font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--muted); }
.legend i { display:inline-block; width:16px; height:3px; border-radius:2px; margin-right:7px;
            vertical-align:middle; }
.chartpad { padding:18px 16px 6px; }

/* ---- callout ---- */
.note { border-left:3px solid var(--accent); background:var(--accent-soft);
        padding:16px 20px; border-radius:0 8px 8px 0; margin:1.6rem 0; font-size:.95rem; }
.note.warn { border-left-color:var(--gold); background:color-mix(in srgb,var(--gold) 8%,transparent); }
.note p:last-child { margin-bottom:0; }
.note .h { font-weight:600; color:var(--ink); display:block; margin-bottom:.3rem; }

.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:1.7rem; }
ul.tight { margin:0 0 1.05rem; padding-left:1.15rem; }
ul.tight li { margin-bottom:.45rem; }
code { font-family:"IBM Plex Mono",monospace; font-size:.86em; background:var(--raised);
       border:1px solid var(--rule-soft); border-radius:4px; padding:1px 5px; color:var(--ink); }
pre { background:var(--raised); border:1px solid var(--rule); border-radius:8px;
      padding:16px 18px; overflow-x:auto; font-size:.82rem; line-height:1.65;
      font-family:"IBM Plex Mono",monospace; color:var(--body); }
footer { margin-top:5rem; padding-top:1.6rem; border-top:1px solid var(--rule);
         color:var(--faint); font-size:.82rem; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
@media (prefers-reduced-motion:no-preference) {
  .panel,.verdict { transition:box-shadow .2s ease; }
}
@media (max-width:640px) {
  .wrap { padding:0 18px 64px; }
  header.top .wrap { padding-top:36px; }
  body { font-size:15.5px; }
}
"""


# --------------------------------------------------------------------- body

STRATEGY_COPY = {
    'swm': ('SWM forecast vs quote', 'trade when the model’s price differs from the book'),
    'swm_delta': ('SWM delta', 'trade on the model’s own predicted move'),
    'always_yes': ('Always YES', 'buy YES on every cell'),
    'always_no': ('Always NO', 'buy NO on every cell'),
    'random': ('Coin flip', 'seeded random side'),
    'momentum': ('Momentum', 'follow the last 24h drift'),
    'contrarian': ('Contrarian', 'fade the last 24h drift'),
    'perfect_direction': ('Perfect direction', 'always right about the sign'),
}
ORDER = ['swm', 'swm_delta', 'momentum', 'contrarian', 'always_yes', 'always_no',
         'random', 'perfect_direction']


def strategy_table(strategies, highlight=('swm', 'swm_delta')):
    rows = []
    for key in ORDER:
        s = strategies.get(key)
        if not s or not s.get('n_trades'):
            continue
        label, blurb = STRATEGY_COPY.get(key, (key, ''))
        tag = ''
        if key.startswith('swm'):
            tag = ' <span class="tag model">model</span>'
        ci = ''
        if s.get('roi_ci_low') is not None:
            ci = (f'<span class="sub">95% CI {s["roi_ci_low"] * 100:+.0f} to '
                  f'{s["roi_ci_high"] * 100:+.0f}%</span>')
        rows.append(
            f'<tr class="{"hl" if key in highlight else ""}">'
            f'<td>{label}{tag}<span class="sub">{blurb}</span></td>'
            f'<td>{s["n_trades"]}</td>'
            f'<td class="{cls_for(s["roi"])}">{pct(s["roi"], 1)}{ci}</td>'
            f'<td class="{cls_for(s["edge_per_share"])}">{cents(s["edge_per_share"])}</td>'
            f'<td class="{cls_for(s.get("net_edge_per_share"))}">{cents(s.get("net_edge_per_share"))}</td>'
            f'<td>{pct(s["hit_rate"], 0, sign=False)}</td>'
            f'<td>{num(s.get("sharpe_per_trade"), 2, sign=True)}</td>'
            f'<td>{pct(s["long_share"], 0, sign=False)}</td>'
            f'</tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        '<th>Strategy</th><th>Trades</th><th>Return on capital</th>'
        '<th>Gross / share</th><th>Net / share</th><th>Hit rate</th><th>Sharpe</th><th>Long</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    )


def matched_table(strategies):
    swm = strategies.get('swm') or {}
    matched = swm.get('matched') or {}
    if not matched:
        return ''
    rows = [
        f'<tr class="hl"><td>SWM forecast vs quote <span class="tag model">model</span></td>'
        f'<td>{swm["n_trades"]}</td>'
        f'<td class="{cls_for(swm["roi"])}">{pct(swm["roi"], 1)}</td>'
        f'<td class="{cls_for(swm["edge_per_share"])}">{cents(swm["edge_per_share"])}</td>'
        f'<td class="{cls_for(swm.get("net_edge_per_share"))}">{cents(swm.get("net_edge_per_share"))}</td>'
        f'<td>{pct(swm["hit_rate"], 0, sign=False)}</td></tr>'
    ]
    for key in ('always_yes', 'always_no', 'random', 'perfect_direction'):
        m = matched.get(key)
        if not m or not m.get('n_trades'):
            continue
        label = STRATEGY_COPY[key][0]
        rows.append(
            f'<tr><td>{label}<span class="sub">same cells, no model</span></td>'
            f'<td>{m["n_trades"]}</td>'
            f'<td class="{cls_for(m["roi"])}">{pct(m["roi"], 1)}</td>'
            f'<td class="{cls_for(m["edge_per_share"])}">{cents(m["edge_per_share"])}</td>'
            f'<td class="{cls_for(m.get("net_edge_per_share"))}">{cents(m.get("net_edge_per_share"))}</td>'
            f'<td>{pct(m["hit_rate"], 0, sign=False)}</td></tr>'
        )
    return ('<div class="scroll"><table><thead><tr><th>On the cells SWM chose</th>'
            '<th>Trades</th><th>Return on capital</th><th>Gross / share</th>'
            '<th>Net / share</th><th>Hit rate</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def sweep_table(sweep, drift=0.0):
    costs = sorted({r['cost'] for r in sweep})
    thrs = sorted({r['threshold'] for r in sweep})
    head = ''.join(f'<th>{c * 100:.1f}c spread</th>' for c in costs)
    body = []
    for t in thrs:
        cells = []
        n = None
        for c in costs:
            r = next(x for x in sweep
                     if x['threshold'] == t and x['cost'] == c and x['entry_drift'] == drift)
            n = r['n_trades']
            cells.append(f'<td class="{cls_for(r["roi"])}">{pct(r["roi"], 0)}</td>')
        body.append(f'<tr><td>|edge| &ge; {t:.2f}<span class="sub">{n} trades</span></td>'
                    + ''.join(cells) + '</tr>')
    return ('<div class="scroll"><table><thead><tr><th>Entry threshold</th>'
            + head + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def page(reports, out_path):
    """Assemble the report. `reports` maps a name to a loaded report JSON."""
    notional = reports['notional']
    shares = reports.get('shares')
    grid = reports.get('grid')
    grid_shares = reports.get('grid_shares')

    bp_r = notional['runs']['retrieval']['move']
    bp_o = notional['runs']['oracle']['move']
    bp_res = notional['runs']['retrieval'].get('resolution', {})
    sh_r = shares['runs']['retrieval']['move'] if shares else None

    swm = bp_r['strategies']['swm']
    matched_yes = (swm.get('matched') or {}).get('always_yes', {})
    uni = bp_r['universe']
    fc = bp_r['forecast']
    cfg = notional['config']

    # ---- verdict numbers -------------------------------------------------
    verdict = [
        ('Model, return on capital', pct(swm['roi'], 1), cls_for(swm['roi']),
         f'{swm["n_trades"]} trades on {uni["n_cells"]} test cells, '
         f'{cfg["cost"] * 100:.0f}c spread, exit at the move'),
        ('Always-YES, same cells', pct(matched_yes.get('roi'), 1), cls_for(matched_yes.get('roi')),
         'the trivial rule, run on exactly the cells the model picked'),
        ('Model, equal share sizing',
         pct(sh_r['strategies']['swm']['roi'], 1) if sh_r else '&mdash;',
         cls_for(sh_r['strategies']['swm']['roi']) if sh_r else '',
         'same trades without the longshot leverage'),
        ('Forecast correlation', num(fc['pearson'], 3), '',
         f'Pearson vs the realised move; direction {fc["direction_accuracy"]:.0%} correct'),
    ]
    vcells = ''.join(
        f'<div class="vcell"><div class="k">{k}</div>'
        f'<div class="v {c}">{v}</div>'
        f'<div class="n">{n}</div></div>'
        for k, v, c, n in verdict
    )

    # ---- charts ----------------------------------------------------------
    price_rows = []
    for label, block in (bp_r.get('strata', {}).get('by_entry_price') or {}).items():
        price_rows.append({'label': label, 'n': block['n_cells'],
                           'a': block['swm']['roi'], 'b': block['always_yes']['roi']})
    price_svg = bar_pair_chart(price_rows, ylabel='Return on capital')

    drift_rows = []
    for d in (0.0, 0.25, 0.50):
        r = next((x for x in bp_r['sweep']
                  if x['threshold'] == cfg['threshold'] and x['cost'] == cfg['cost']
                  and x['entry_drift'] == d), None)
        if r:
            drift_rows.append({'label': f'{d:.0%}', 'roi': r['roi']})
    drift_svg = drift_chart(drift_rows)

    # ---- information-value section ---------------------------------------
    info_html = ''
    gi = (grid['runs']['retrieval']['move'].get('information') or {}) if grid else {}
    bi = bp_r.get('information') or {}
    if gi and bi:
        gblock = grid['runs']['retrieval']['move']
        gswm_i = gblock['strategies']['swm']
        gm = gswm_i.get('matched') or {}
        yes_sh = (gm.get('always_yes') or {}).get('edge_per_share')
        no_sh = (gm.get('always_no') or {}).get('edge_per_share')
        best_sh = (gm.get('perfect_direction') or {}).get('edge_per_share')
        capture = (
            f'{gswm_i["edge_per_share"] / best_sh:.0%}'
            if best_sh else '&mdash;'
        )
        rows = ''
        for label, blk in (('Breakpoint cells', bi), ('Full grid', gi)):
            rows += (
                f'<tr><td>{label}<span class="sub">{blk["n"]:,} positions</span></td>'
                f'<td>{num(blk["model_rmse"], 4)}</td>'
                f'<td class="gain">{num(blk["quote_rmse"], 4)}</td>'
                f'<td>{num(blk["anchor_rmse"], 4)}</td>'
                f'<td>{pct(blk["target_overlap"], 0, sign=False)}</td>'
                f'<td>{num(blk["trained_corr"], 3)}</td>'
                f'<td class="{cls_for(blk["residual_corr"])}">{num(blk["residual_corr"], 3)}'
                f'<span class="sub">t = {blk["residual_t"]:+.1f}</span></td></tr>'
            )
        info_html = (
            '<section id="information">\n'
            '  <div class="col">\n'
            '    <p class="kicker">What it actually learned</p>\n'
            '    <h2>The forecast is real, and it is mostly not about the future</h2>\n'
            '    <p class="lede">The model predicts the price 24 hours ahead of a 24-hour-old\n'
            '      anchor. By the time anyone can trade, the book has already moved most of that\n'
            '      distance.</p>\n'
            '  </div>\n'
            '  <figure class="panel">\n'
            '    <div class="cap"><strong>Forecast error against the settlement price</strong>, and\n'
            '      how much of the trained target was public before the trade.</div>\n'
            '    <div class="scroll"><table><thead><tr><th>Universe</th><th>Model RMSE</th>\n'
            '      <th>Live quote RMSE</th><th>24h anchor RMSE</th><th>Target already public</th>\n'
            '      <th>Corr w/ trained target</th><th>Corr w/ tradeable part</th>\n'
            f'      </tr></thead><tbody>{rows}</tbody></table></div>\n'
            '  </figure>\n'
            '  <div class="col">\n'
            '    <p>Read the first two columns together. As an estimate of where the price settles,\n'
            '      the model is <strong>worse than simply reading the current quote</strong> &mdash;\n'
            f'      {num(gi["model_rmse"], 3)} against {num(gi["quote_rmse"], 3)} on the grid &mdash;\n'
            '      and no better than the 24-hour-old price it anchors on. That is less a failure of\n'
            '      training than a statement about the target:\n'
            f'      {pct(gi["target_overlap"], 0, sign=False)} of the 24h move it is graded on is\n'
            '      already visible in the book at decision time. Most of what the objective rewards\n'
            '      is re-deriving public information.</p>\n'
            '    <p>The last column is the part that matters, and it is positive. Against the\n'
            '      residual &mdash; the slice of the move still on the table when the order is\n'
            f'      placed &mdash; the model\u2019s own delta correlates {num(gi["residual_corr"], 3)}\n'
            f'      on the grid (t = {gi["residual_t"]:+.1f} over {gi["n"]:,} positions) and\n'
            f'      {num(bi["residual_corr"], 3)} on the breakpoint cells. Small, but not noise.</p>\n'
            '    <div class="note">\n'
            '      <span class="h">The cleanest evidence that something was learned</span>\n'
            f'      <p>On the {gswm_i["n_trades"]:,} grid positions the model chose to trade, buying\n'
            f'      YES blind earns {cents(yes_sh)} a share and buying NO blind earns\n'
            f'      {cents(no_sh)} &mdash; the selection is directionally neutral, so there is no\n'
            '      drift to ride. The model earns\n'
            f'      <strong>{cents(gswm_i["edge_per_share"])}</strong> on those same positions, and\n'
            f'      perfect direction would earn {cents(best_sh)}. So the forecast captures about\n'
            f'      {capture} of the edge that was there. Real signal &mdash; and roughly half the\n'
            f'      size it needs to be to pay a {cfg["cost"] * 100:.0f}c spread.</p>\n'
            '    </div>\n'
            '    <h3>What would make it tradeable</h3>\n'
            '    <ul class="tight">\n'
            '      <li><strong>Train on the residual, not the 24h move.</strong> The target should\n'
            '        be the settlement price minus the quote at decision time. As set up, three\n'
            '        quarters of the gradient goes into predicting what the book already shows.</li>\n'
            '      <li><strong>Show the model the current quote.</strong> Its history stops 24 hours\n'
            '        before the prediction, so it forecasts without the single most informative\n'
            '        number available &mdash; the live-quote column above is the baseline it is\n'
            '        being denied.</li>\n'
            '      <li><strong>Score against the residual too.</strong> Pearson against the 24h\n'
            f'        delta reads {num(gi["trained_corr"], 3)} on the grid while the tradeable\n'
            f'        correlation is {num(gi["residual_corr"], 3)}; only the second one can be\n'
            '        turned into a position.</li>\n'
            '      <li><strong>Evaluate on a direction-balanced universe.</strong> On the breakpoint\n'
            f'        set \u201calways say up\u201d scores {pct(uni["frac_move_up"], 0, sign=False)}\n'
            '        direction accuracy; any headline direction number below that is worse than a\n'
            '        constant.</li>\n'
            '    </ul>\n'
            '  </div>\n'
            '</section>'
        )

    # ---- grid section ----------------------------------------------------
    grid_html = ''
    if grid:
        g = grid['runs']['retrieval']['move']
        gs = grid_shares['runs']['retrieval']['move'] if grid_shares else None
        gswm = g['strategies']['swm']
        gmatch = (gswm.get('matched') or {}).get('always_yes', {})
        gmatch_no = (gswm.get('matched') or {}).get('always_no', {})
        gperfect = (gswm.get('matched') or {}).get('perfect_direction', {})
        gu = g['universe']
        # Where the spread eats the edge: the cost at which the model breaks even.
        cost_row = sorted(
            (x for x in g['sweep']
             if x['threshold'] == cfg['threshold'] and x['entry_drift'] == 0.0),
            key=lambda x: x['cost'],
        )
        breakeven = next((x['cost'] for x in cost_row if x['roi'] < 0), None)
        gross_free = cost_row[0] if cost_row else None
        strata = g.get('strata', {})
        curves = g.get('equity_curves') or {'SWM': g['equity_curve']}
        cell_rows = ''
        for label, block in (strata.get('by_cell_type') or {}).items():
            cell_rows += (
                f'<tr><td>{"Breakpoint cells" if label == "breakpoint" else "Quiet cells"}'
                f'<span class="sub">{block["n_cells"]} cells</span></td>'
                f'<td>{block["swm"]["n_trades"]}</td>'
                f'<td class="{cls_for(block["swm"]["roi"])}">{pct(block["swm"]["roi"], 1)}</td>'
                f'<td class="{cls_for(block["swm"]["edge_per_share"])}">{cents(block["swm"]["edge_per_share"])}</td>'
                f'<td class="{cls_for(block["always_yes"]["roi"])}">{pct(block["always_yes"]["roi"], 1)}</td>'
                f'</tr>')
        grid_html = f"""
<section id="grid">
  <div class="col">
    <p class="kicker">The unbiased universe</p>
    <h2>Every market, every decision point</h2>
    <p class="lede">The 314 breakpoint cells all sit on a big move. To ask whether the model
      would survive contact with a real book, it has to score the quiet markets too &mdash; and
      decline to trade them.</p>
    <p>At each of the {gu["n_times"]} decision times the grid scores every market with a
      recent enough quote and a later one to settle against: <strong>{gu["n_cells"]:,} cells</strong>
      across {gu["n_markets"]} markets, of which only {gu["n_breakpoint_cells"]} carry a
      breakpoint. The upward tilt is gone &mdash; mean move {cents(gu["mean_move"])},
      {gu["frac_move_up"]:.0%} of cells up &mdash; and with it the free money: always-YES now
      returns {pct(g["strategies"]["always_yes"]["roi"], 1)} and always-NO
      {pct(g["strategies"]["always_no"]["roi"], 1)}. The model routes
      {gu["frac_routed_null"]:.0%} of cells to its null option, which is the behaviour this
      universe was built to test.</p>
    <p>A quiet market is quoted once a day, so many decision times land between the same pair of
      quotes and describe the same position rather than a new one. Those
      {gu["n_cells"]:,} cells are {gu.get("n_distinct_positions", 0):,} distinct positions, and
      each is filled once, at the first decision time that fires on it &mdash; for the model and
      every baseline alike. Counting the repeats as separate trades would inflate the trade
      count fourfold and re-weight the result toward the quiet markets that repeat most.</p>
  </div>
  <figure class="panel">
    <div class="cap"><strong>Full grid, exit at the next quote.</strong>
      {gswm["n_trades"]:,} of {gu["n_cells"]:,} cells cleared the
      |edge|&nbsp;&ge;&nbsp;{cfg["threshold"]:.2f} threshold.</div>
    {strategy_table(g['strategies'])}
  </figure>
  <figure class="panel">
    <div class="cap"><strong>Breakpoint cells versus quiet cells.</strong>
      Where in the grid the model's P&amp;L actually comes from.</div>
    <div class="scroll"><table><thead><tr><th>Cell type</th><th>Trades</th>
      <th>SWM return</th><th>SWM edge / share</th><th>Always-YES return</th>
      </tr></thead><tbody>{cell_rows}</tbody></table></div>
  </figure>
  <figure class="panel">
    <div class="cap"><strong>Cumulative return on deployed capital</strong> over the test month,
      full grid.</div>
    <div class="chartpad">{equity_chart(curves)}</div>
  </figure>
  <div class="col">
    <p>The forecast itself barely survives the move off breakpoints: correlation falls to
      {num(g["forecast"]["pearson"], 3)} and skill against the no-change baseline turns
      negative ({num(g["forecast"]["skill_vs_no_change"], 3, sign=True)}). What is left is a
      small directional edge, and the question is whether it covers the spread.</p>
    <div class="note warn">
      <span class="h">The edge is real, and smaller than the spread</span>
      <p>Gross, the model captures <strong>{cents(gswm['edge_per_share'])}</strong> a share.
      Perfect direction on the same positions would capture
      <strong>{cents(gperfect.get('edge_per_share'))}</strong> &mdash; so the forecast is
      picking up roughly
      {(gswm['edge_per_share'] / gperfect['edge_per_share']):.0%} of the directional edge that
      was there to take. That is not nothing, and it is not enough: net of the
      {cfg['cost'] * 100:.0f}c spread the same trades earn
      <strong class="{cls_for(gswm['net_edge_per_share'])}">{cents(gswm['net_edge_per_share'])}</strong>
      a share, for
      <strong class="{cls_for(gswm['roi'])}">{pct(gswm['roi'], 1)}</strong> on capital
      &mdash; 95% CI {pct(gswm.get('roi_ci_low'), 1)} to {pct(gswm.get('roi_ci_high'), 1)},
      which is a result indistinguishable from zero.</p>
      <p>Break-even sits at a spread of about
      {f"{breakeven * 100:.1f}c" if breakeven else 'under a cent'}: charge nothing and the same
      trades return {pct(gross_free['roi'], 1) if gross_free else '&mdash;'}; charge a full cent
      and they do not. Real books on these markets are wider than that, and a quarter of the move
      leaking in before the fill takes it to
      {pct(next((x['roi'] for x in g['sweep'] if x['threshold'] == cfg['threshold'] and x['cost'] == cfg['cost'] and x['entry_drift'] == 0.25), None), 1)}.</p>
    </div>
  </div>
</section>"""

    # ---- resolution row --------------------------------------------------
    res_html = ''
    if bp_res:
        rs = bp_res['strategies']['swm']
        rm = (rs.get('matched') or {}).get('always_yes', {})
        res_html = (
            f'<p>Holding to resolution instead of closing at the move makes it worse, not better: '
            f'over the {bp_res["universe"]["n_cells"]} test cells whose market has since resolved, '
            f'the model returns <strong class="{cls_for(rs["roi"])}">{pct(rs["roi"], 1)}</strong> '
            f'against <strong>{pct(rm.get("roi"), 1)}</strong> for always-YES on the same cells.</p>'
        )

    generated = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    html = f"""<title>Does SWM Beat Always-YES?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>

<header class="top">
  <div class="wrap">
    <p class="eyebrow">Trading backtest &middot; swm-wm-jin10-daily-7b</p>
    <h1>Does the world model beat always&#8209;YES?</h1>
    <p class="standfirst">On the jin10 test set it looks profitable. Buying YES blind looks
      more profitable. Once the universe stops being selected on big moves, the edge is real but
      smaller than the spread.</p>
    <div class="byline">
      <span>{uni["n_cells"]} held-out cells &middot; {uni["n_markets"]} markets</span>
      <span>{day(uni["t_start"])} &ndash; {day(uni["t_end"])} 2026</span>
      <span>Polymarket &middot; jin10 wire</span>
      <span>{generated}</span>
    </div>
    <div class="verdict">{vcells}</div>
  </div>
</header>

<div class="wrap">

<section id="setup">
  <div class="col">
    <p class="kicker">What was run</p>
    <h2>The setup</h2>
    <p class="lede">A walk-forward backtest on the final 10% of
      <code>swmbench_jin10_dailyhist_en</code>, the slice the checkpoint never trained on.</p>
    <p>The dataset holds 3,134 records over 920 Polymarket markets. Split chronologically by
      <code>target.t</code>, exactly as training did, the last 10% is
      <strong>{uni["n_cells"]} records over {uni["n_markets"]} markets and
      {uni["n_events"]} events</strong>, {day(uni["t_start"])}&ndash;{day(uni["t_end"])} 2026.</p>
    <p>Each record's headlines are published in a one-hour window ending 90 minutes before the
      move. So the decision lands there, on the last quote the book had shown &mdash; typically the
      two-hour-prior price &mdash; and the position closes at the move. Prompt inputs are clipped to
      what was knowable at decision time; the settlement quote is a label and never reaches the
      model. Prompt parity was checked cell by cell: all {uni["n_cells"]} reproduce the training
      pipeline's prompt string exactly.</p>
    <p>The dataset's own <code>attributions</code> field &mdash; which headline caused the move &mdash;
      was scored with the realised move in hand, so it cannot be used live. The headline
      numbers here instead retrieve from the reconstructed jin10 wire (114,316 de-duplicated
      English items) with a bi-encoder calibrated on train-split records only. The oracle
      variant is reported alongside, labelled for what it is.</p>
  </div>
</section>

<section id="skill">
  <div class="col">
    <p class="kicker">First, the good news</p>
    <h2>The forecasts are genuinely informative</h2>
  </div>
  <figure class="panel">
    <div class="cap"><strong>Forecasting metrics on the held-out cells.</strong>
      Predicted 24h price change against the realised one.</div>
    <div class="scroll"><table><thead><tr><th>News source</th><th>Correlation</th>
      <th>Direction</th><th>RMSE</th><th>No-change RMSE</th><th>Skill</th><th>Forecast &sigma;</th>
      </tr></thead><tbody>
      <tr class="hl"><td>Retrieval <span class="tag model">live</span>
        <span class="sub">bi-encoder over the wire</span></td>
        <td>{num(bp_r["forecast"]["pearson"])}</td>
        <td>{pct(bp_r["forecast"]["direction_accuracy"], 0, sign=False)}</td>
        <td>{num(bp_r["forecast"]["rmse"], 4)}</td>
        <td>{num(bp_r["forecast"]["no_change_rmse"], 4)}</td>
        <td class="{cls_for(bp_r["forecast"]["skill_vs_no_change"])}">{num(bp_r["forecast"]["skill_vs_no_change"], 3, sign=True)}</td>
        <td>{num(bp_r["forecast"]["pred_std"], 4)}</td></tr>
      <tr><td>Oracle attributions <span class="tag leak">upper bound</span>
        <span class="sub">uses the answer; not achievable live</span></td>
        <td>{num(bp_o["forecast"]["pearson"])}</td>
        <td>{pct(bp_o["forecast"]["direction_accuracy"], 0, sign=False)}</td>
        <td>{num(bp_o["forecast"]["rmse"], 4)}</td>
        <td>{num(bp_o["forecast"]["no_change_rmse"], 4)}</td>
        <td class="{cls_for(bp_o["forecast"]["skill_vs_no_change"])}">{num(bp_o["forecast"]["skill_vs_no_change"], 3, sign=True)}</td>
        <td>{num(bp_o["forecast"]["pred_std"], 4)}</td></tr>
      </tbody></table></div>
  </figure>
  <div class="col">
    <p>Both beat the no-change baseline, and the model is properly shrunk &mdash; it forecasts a
      {num(bp_r["forecast"]["pred_std"], 3)} standard deviation against a realised
      {num(bp_r["forecast"]["true_std"], 3)}, which is what a well-calibrated regressor on a
      noisy target should do. Replacing oracle attributions with live retrieval costs roughly
      half the correlation, and that gap is the price of not knowing which headline mattered.</p>
    <p>None of that is a trading result.</p>
  </div>
</section>

<section id="bias">
  <div class="col">
    <p class="kicker">The trap</p>
    <h2>This test set pays you to say yes</h2>
    <p class="lede">Every record exists because a large move was detected there &mdash; and
      {uni["frac_move_up"]:.0%} of those moves are upward.</p>
    <p>Mean move across the {uni["n_cells"]} cells is <strong>{cents(uni["mean_move"])}</strong> per
      share, on an average entry of {num(uni["mean_entry_price"], 2)}. Buying YES on everything,
      blind, with no model at all, returns
      <strong class="gain">{pct(bp_r["strategies"]["always_yes"]["roi"], 1)}</strong> on capital.
      A seeded coin flip returns {pct(bp_r["strategies"]["random"]["roi"], 1)}. Any headline
      P&amp;L on this universe has to clear that bar before it means anything.</p>
  </div>
  <figure class="panel">
    <div class="cap"><strong>All strategies, identical cells and identical costs.</strong>
      {cfg["cost"] * 100:.0f}c spread charged on entry; the model trades only when
      |edge|&nbsp;&ge;&nbsp;{cfg["threshold"]:.2f}, the trivial rules trade everything.</div>
    {strategy_table(bp_r['strategies'])}
  </figure>
  <div class="col">
    <p>The model clears the bar on paper. But it trades a subset, and comparing a selective
      strategy against a baseline that trades everything confounds two questions: does it pick
      good cells, and does it call direction better than a coin. Re-running the trivial rules on
      <em>exactly</em> the cells the model chose separates them.</p>
  </div>
  <figure class="panel">
    <div class="cap"><strong>Matched comparison.</strong> Same {swm["n_trades"]} cells,
      same costs, model replaced by a rule.</div>
    {matched_table(bp_r['strategies'])}
  </figure>
  <div class="col">
    <div class="note warn">
      <span class="h">On its own picks, the model loses to buying YES blind</span>
      <p>{pct(swm['roi'], 1)} against {pct(matched_yes.get('roi'), 1)}, with a hit rate of
      {swm['hit_rate']:.0%} where always-YES gets {matched_yes.get('hit_rate', 0):.0%}. The model's
      selection is not adding direction; it is subtracting it.</p>
    </div>
    {res_html}
  </div>
</section>

<section id="sizing">
  <div class="col">
    <p class="kicker">Where the money comes from</p>
    <h2>Twenty-one longshots</h2>
    <p class="lede">Under fixed-notional sizing a dollar buys fifty shares at 2c and one share
      at 95c. That is where the return is.</p>
  </div>
  <figure class="panel">
    <div class="cap"><strong>Return on capital by entry price.</strong>
      <span style="color:var(--accent)">&#9632;</span> SWM &nbsp;
      <span style="color:var(--faint)">&#9632;</span> always-YES, same cells.</div>
    <div class="chartpad">{price_svg}</div>
  </figure>
  <div class="col">
    <p>Every cent of the model's positive return sits in the cheapest bucket. In every other
      price band it is flat or negative, and it loses to always-YES in all of them. Re-run the
      identical trades with equal share counts instead of equal dollars &mdash; stripping the
      leverage out, keeping the direction calls &mdash; and the result inverts.</p>
  </div>
  {'''<figure class="panel">
    <div class="cap"><strong>Equal-share sizing.</strong> One share per trade instead of one
      dollar of capital. Same cells, same signals, same costs.</div>
    ''' + strategy_table(sh_r['strategies']) + '''
  </figure>''' if sh_r else ''}
</section>

<section id="execution">
  <div class="col">
    <p class="kicker">Execution</p>
    <h2>The edge needs a stale quote to survive</h2>
    <p class="lede">The backtest fills at the last price the data shows, up to 30 minutes before
      the decision. A real book would have moved.</p>
  </div>
  <div class="two">
    <figure class="panel">
      <div class="cap"><strong>Return vs assumed slippage.</strong> Share of the eventual move
        already priced in when the order lands.</div>
      <div class="chartpad">{drift_svg}</div>
    </figure>
    <figure class="panel">
      <div class="cap"><strong>Threshold &times; spread.</strong> Return on capital across the
        grid of trading rules, no slippage.</div>
      {sweep_table(bp_r['sweep'])}
    </figure>
  </div>
  <div class="col">
    <p>Assume a quarter of the move has leaked into the book by the time the order lands and the
      return falls to {pct(drift_rows[1]["roi"], 1) if len(drift_rows) > 1 else '&mdash;'};
      assume half and it is
      {pct(drift_rows[2]["roi"], 1) if len(drift_rows) > 2 else '&mdash;'}. Since the signal is
      public wire copy that every other participant reads at the same moment, some leakage is
      the realistic case, not the pessimistic one.</p>
    <p>The threshold sweep is 96 configurations evaluated on {uni["n_cells"]} cells. Read it as a
      robustness check, not a menu &mdash; picking the best cell of it would be fitting the test set.</p>
  </div>
</section>

{info_html}

{grid_html}

<section id="repro">
  <div class="col">
    <p class="kicker">Reproducing</p>
    <h2>How to run it</h2>
  </div>
  <pre># 1. cells the model is asked to score
python scripts/backtest_build_grid.py --data swmbench_jin10_dailyhist_en.jsonl \\
    --mode breakpoint --news retrieval --out results/backtest/grid.jsonl

# 2. score them (shard with --num-shards / --shard-idx)
python scripts/backtest_predict.py --grid results/backtest/grid.jsonl \\
    --model-path swm-wm-jin10-daily-7b --model-name Qwen/Qwen2.5-7B-Instruct \\
    --out results/backtest/preds.jsonl --batch-size 8

# 3. P&amp;L, baselines, sweeps
python scripts/backtest_report.py --preds retrieval=results/backtest/preds.jsonl \\
    --out results/backtest/report.json --threshold {cfg["threshold"]} --cost {cfg["cost"]}</pre>
  <div class="col">
    <h3>What would change the answer</h3>
    <ul class="tight">
      <li><strong>A universe that is not breakpoint-selected.</strong> Every cell here sits on or
        near a detected move, so the model is never asked the question a live system faces most
        of the time: is anything happening at all.</li>
      <li><strong>Order-book data.</strong> Fills are assumed at the quote for the full stake.
        Depth at 2&ndash;10c is thin, and that is exactly where the fixed-notional P&amp;L is.</li>
      <li><strong>Finer price history.</strong> Entry uses a quote up to 30 minutes stale; the
        slippage sweep is a stand-in for measuring it.</li>
      <li><strong>More than one month.</strong> {uni["n_cells"]} cells over {uni["n_events"]} events
        gives wide intervals; the confidence bands above are a cluster bootstrap over
        <code>event_id</code> for that reason.</li>
    </ul>
  </div>
</section>

<footer>
  <div class="col">
    <p>Generated {generated} from <code>results/backtest/*.json</code>.
    Model <code>swmbench/swm-wm-jin10-daily-7b</code>, data
    <code>swmbench/swmbench &middot; swmbench_jin10_dailyhist_en.jsonl</code>.
    Method notes in <code>docs/backtest.md</code>.</p>
  </div>
</footer>
</div>
"""
    Path(out_path).write_text(html)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notional', required=True)
    ap.add_argument('--shares')
    ap.add_argument('--grid')
    ap.add_argument('--grid-shares')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    reports = {'notional': json.load(open(args.notional))}
    for key, path in (('shares', args.shares), ('grid', args.grid),
                      ('grid_shares', args.grid_shares)):
        if path:
            reports[key] = json.load(open(path))
    print('wrote', page(reports, args.out))


if __name__ == '__main__':
    main()
