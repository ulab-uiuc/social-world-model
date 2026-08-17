#!/usr/bin/env python3
"""Build ONE combined HTML report: (A) standard test-set forecasting metrics +
(B) the trading/holdings backtest, with the P&L calculation spelled out.

Reads inference jsonls in results/eval_polymarket/ (schema from
inference_multievent_world_model.py) and computes both the statistical metrics
and the trade P&L in-process, then writes results/backtest/report_full.html.
"""
import json, math, statistics as s, hashlib, os
from pathlib import Path

EVAL = "results/eval_polymarket"
OUT = Path("results/backtest/report_full.html")
OUT.parent.mkdir(parents=True, exist_ok=True)
SWMBENCH = "data/swmbench_jin10_attributed_filtered_en.jsonl"
COST, PMIN, STAKE, HTHR = 0.02, 0.02, 1.0, 0.05

# ---------- load ----------
def load(tag):
    fn = f"{EVAL}/{tag}.jsonl"
    if not os.path.exists(fn):
        return None
    rows = [json.loads(l) for l in open(fn)]
    return rows if len(rows) >= 20 else None

def outcomes():
    m = {}
    if not os.path.exists(SWMBENCH):
        return m
    for line in open(SWMBENCH):
        r = json.loads(line); o = r.get("outcome")
        if o is None: continue
        o = str(o).strip().lower()
        if o == "yes": m[str(r["market_id"])] = 1.0
        elif o == "no": m[str(r["market_id"])] = 0.0
    return m

# ---------- stats ----------
def rk(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r=[0.0]*len(x); i=0
    while i < len(x):
        j=i
        while j+1<len(x) and x[o[j+1]]==x[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=(i+j)/2+1
        i=j+1
    return r
def pear(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))/n; sa=s.pstdev(a); sb=s.pstdev(b)
    return cov/(sa*sb) if sa and sb else float("nan")
def stat_metrics(rows):
    p=[r["pred_delta"] for r in rows]; t=[r["true_delta"] for r in rows]
    n=len(p); mt=sum(t)/n
    ss=sum((y-mt)**2 for y in t); sr=sum((x-y)**2 for x,y in zip(p,t)); r2=1-sr/ss if ss else float("nan")
    mv=[(x,y) for x,y in zip(p,t) if abs(y)>0.01]; cm=[(x,y) for x,y in mv if abs(x)>1e-6]
    cor=sum(1 for x,y in cm if (x>0)==(y>0)); up=sum(1 for x,y in mv if y>0)
    return {"n":n,"pearson":pear(p,t),"spearman":pear(rk(p),rk(t)),"r2":r2,
            "diracc":cor/len(cm) if cm else float("nan"),"base":max(up,len(mv)-up)/len(mv) if mv else 0}

# ---------- trades ----------
def trade(direction,p,ss_,stake):
    p=min(max(p,PMIN),1-PMIN)
    ret=(ss_-p)/p if direction>0 else (p-ss_)/(1-p)
    return stake*ret - COST*stake
def strat(rows, mode, thr):
    tr=[]
    for r in rows:
        sig=r["pred_delta"]
        if mode=="swm":
            if abs(sig)<thr: continue
            d=1 if sig>0 else -1
        elif mode=="yes": d=1
        elif mode=="no": d=-1
        else:
            d=1 if int(hashlib.md5(f"{r['mid']}_{r['t']}".encode()).hexdigest(),16)%2==0 else -1
        tr.append({"t":r["t"],"pnl":trade(d,r["entry"],r["settle"],STAKE)})
    return tr
def tmetrics(tr):
    if not tr: return {"n":0,"roi":0,"win":float("nan"),"sharpe":float("nan"),"pnl":0}
    pnl=sum(x["pnl"] for x in tr); cap=len(tr)*STAKE; rets=[x["pnl"]/STAKE for x in tr]
    sd=s.pstdev(rets) if len(rets)>1 else 0
    return {"n":len(tr),"pnl":pnl,"roi":pnl/cap,"win":sum(1 for x in tr if x["pnl"]>0)/len(tr),
            "sharpe":(sum(rets)/len(rets)/sd) if sd else float("nan")}
def build_rows(preds, exitmode, oc):
    rows=[]
    for d in preds:
        entry=d.get("before_price")
        if entry is None: continue
        if exitmode=="move":
            se=d.get("true_price")
        else:
            se=oc.get(str(d.get("market_id")))
        if se is None: continue
        rows.append({"mid":str(d.get("market_id")),"t":d.get("t") or 0,
                     "entry":float(entry),"settle":float(se),"pred_delta":float(d.get("pred_delta"))})
    rows.sort(key=lambda r:r["t"]); return rows

# ---------- assemble ----------
oc = outcomes()
# (A) 2x2 stat metrics
cells = [("jin10d_bal","daily-7B","jin10 daily test"),
         ("pm7b_on_jin10d","pm-7B","jin10 daily test"),
         ("jin10d_on_swmb","daily-7B","swm-bench test_pm"),
         ("pm_7b","pm-7B","swm-bench test_pm")]
statrows=[]
for tag,mdl,test in cells:
    rows=load(tag)
    statrows.append((mdl,test,stat_metrics(rows) if rows else None))

# (B) backtest on daily-7B (jin10d_bal preds)
bt_preds = load("jin10d_bal") or []
bt = {}
curves = {}
for exitmode in ["move","resolution"]:
    rws = build_rows(bt_preds, exitmode, oc)
    res={}
    res["SWM"]=tmetrics(strat(rws,"swm",HTHR))
    res["always-YES"]=tmetrics(strat(rws,"yes",0))
    res["random"]=tmetrics(strat(rws,"random",0))
    res["_n_used"]=len(rws)
    # sweep
    res["_sweep"]=[(thr,tmetrics(strat(rws,"swm",thr))) for thr in (0.0,0.02,0.05,0.10)]
    bt[exitmode]=res
    # equity curve
    c=0.0; pts=[]
    for x in sorted(strat(rws,"swm",HTHR),key=lambda z:z["t"]): c+=x["pnl"]; pts.append((x["t"],c))
    curves[exitmode]=pts

# equity SVG (move mode SWM)
def svg_curve(pts):
    if not pts: return "<p class='muted'>no trades</p>"
    W,H,m=820,300,{"l":52,"r":14,"t":12,"b":26}
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]+[0.0]
    xmn,xmx=min(xs),max(xs); ymn,ymx=min(ys),max(ys); pad=(ymx-ymn)*.1 or 1; ymn-=pad; ymx+=pad
    X=lambda t: m["l"]+(0 if xmx==xmn else (t-xmn)/(xmx-xmn))*(W-m["l"]-m["r"])
    Y=lambda c: m["t"]+(1-(0.5 if ymx==ymn else (c-ymn)/(ymx-ymn)))*(H-m["t"]-m["b"])
    o=[f"<svg viewBox='0 0 {W} {H}'>"]
    for i in range(5):
        c=ymn+(ymx-ymn)*i/4; y=Y(c)
        o.append(f"<line x1='{m['l']}' x2='{W-m['r']}' y1='{y:.0f}' y2='{y:.0f}' stroke='var(--grid)'/>")
        o.append(f"<text x='{m['l']-6}' y='{y+3:.0f}' text-anchor='end' class='ax'>{c:.0f}</text>")
    o.append(f"<line x1='{m['l']}' x2='{W-m['r']}' y1='{Y(0):.0f}' y2='{Y(0):.0f}' stroke='var(--axis)' stroke-width='1.5'/>")
    d=" ".join(("M" if i==0 else "L")+f"{X(t):.1f} {Y(c):.1f}" for i,(t,c) in enumerate(pts))
    o.append(f"<path d='{d}' fill='none' stroke='#2a78d6' stroke-width='2'/></svg>")
    return "".join(o)

def fmt(x,pct=False,d=3):
    if x is None or (isinstance(x,float) and math.isnan(x)): return "–"
    return f"{x*100:+.1f}%" if pct else f"{x:.{d}f}"
def pct0(x):
    return "–" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{x*100:.0f}%"
def pct1(x):
    return "–" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{x*100:.1f}%"

# ---- HTML ----
def stat_row(mdl,test,mt):
    if not mt:
        return f"<tr><td>{mdl}</td><td>{test}</td><td colspan=5 class=muted>eval 运行中…</td></tr>"
    return (f"<tr><td>{mdl}</td><td>{test}</td><td>{mt['n']}</td>"
            f"<td>{fmt(mt['pearson'])}</td><td>{fmt(mt['spearman'])}</td><td>{fmt(mt['r2'])}</td>"
            f"<td>{pct1(mt['diracc'])} <span class=muted>/ base {pct0(mt['base'])}</span></td></tr>")
statA="".join(stat_row(mdl,test,mt) for mdl,test,mt in statrows)

def bt_table(res):
    return "".join(f"<tr><td>{k}</td><td>{res[k]['n']}</td><td>{fmt(res[k]['roi'],1)}</td>"
                   f"<td>{pct0(res[k]['win'])}</td><td>{fmt(res[k]['sharpe'],d=2)}</td></tr>"
                   for k in ["SWM","always-YES","random"])
def sweep_table(res):
    return "".join(f"<tr><td>{thr:.2f}</td><td>{m['n']}</td><td>{fmt(m['roi'],1)}</td>"
                   f"<td>{pct0(m['win'])}</td><td>{fmt(m['sharpe'],d=2)}</td></tr>"
                   for thr,m in res["_sweep"])

html=f"""<!doctype html><meta charset=utf-8><title>SWM — 预测指标 + 交易回测</title>
<style>
:root{{--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.1)}}
@media(prefers-color-scheme:dark){{:root{{--surface-1:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.1)}}}}
body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:920px;margin:0 auto;padding:26px}}h1{{font-size:20px;margin:0 0 2px}}h2{{font-size:15px;margin:26px 0 8px;border-bottom:2px solid var(--border);padding-bottom:4px}}
h3{{font-size:13px;color:var(--ink-2);margin:16px 0 6px}}.muted{{color:var(--muted)}}.sub{{color:var(--muted);font-size:13px;margin:0 0 6px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;margin:6px 0;width:100%}}
td,th{{border:1px solid var(--border);padding:5px 10px;text-align:right}}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left}}th{{color:var(--ink-2);background:var(--surface-1)}}
.card{{border:1px solid var(--border);border-radius:12px;background:var(--surface-1);padding:14px 18px;margin:12px 0}}
.method{{background:var(--surface-1);border-left:3px solid #2a78d6;border-radius:6px;padding:10px 14px;font-size:13px}}
code{{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12px}}
svg{{display:block;width:100%;height:300px}}.ax{{fill:var(--muted);font-size:11px}}
.warn{{color:#d03b3b;font-weight:600}}
.tiles{{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}}.tile{{border:1px solid var(--border);border-radius:10px;padding:8px 14px;min-width:140px}}
.tile .k{{font-size:11px;color:var(--muted)}}.tile .v{{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}}
</style>
<div class=wrap>
<h1>SWM 评估报告:预测指标 + 交易回测</h1>
<p class=sub>模型:daily-7B(jin10 金十新闻归因数据 → 日级 history → 去偏 + 时间切分 → Qwen2.5-7B)与 pm-7B(swm-bench polymarket 数据)。</p>

<h2>A. 标准预测指标(test set)</h2>
<p class=sub>预测 delta 与真实 delta 的相关/方向准确率。<b>同一 test（同一列）才可直接比。</b>方向准确率的基线=多数类占比。</p>
<table><tr><th>模型</th><th>test set</th><th>n</th><th>Pearson</th><th>Spearman</th><th>R²</th><th>方向准确率</th></tr>{statA}</table>
<p class=sub>swm-bench test 那两行若显示"运行中"是因为 3483 条 7B 推理还没跑完(已重排)。</p>

<h2>B. 交易回测(持仓 P&amp;L)</h2>
<div class=method>
<b>持仓怎么算的(逐笔)</b><br>
对每个决策点(test 里的一条记录):<br>
• <b>信号</b> = <code>pred_delta = pred_price − before_price</code>(模型预测的价格变动)<br>
• <b>方向</b>:<code>pred_delta &gt; +阈值</code> → 买 YES;<code>&lt; −阈值</code> → 买 NO;否则不交易<br>
• <b>入场价 p</b> = <code>before_price</code>(决策时的市场价 = history 最后一点)<br>
• <b>出场价 s</b>:<b>move 模式</b> = <code>true_price</code>(那波新闻移动后的实现价);<b>resolution 模式</b> = 市场最终结算 <code>outcome</code>(Yes→1 / No→0)<br>
• <b>每笔盈亏</b>(每 $1 本金,round-trip 成本 {int(COST*100)}%):<br>
&nbsp;&nbsp;买 YES:<code>pnl = (s − p)/p − 成本</code>&nbsp;&nbsp;买 NO:<code>pnl = (p − s)/(1 − p) − 成本</code><br>
• 入场价夹到 [{PMIN},{1-PMIN}] 防止极端价格放大;每笔固定 $1;权益曲线 = 按时间累加逐笔 pnl。<br>
• <b>基线</b>:always-YES(每条都买 YES)、random(随机方向),用同一批记录对比。
</div>

<h3>结果(daily-7B,阈值 {HTHR})</h3>
<table><tr><th>exit</th><th>策略</th><th>#trades</th><th>ROI</th><th>胜率</th><th>Sharpe</th></tr>
<tr><td rowspan=3>move<br>(短线)</td><td>SWM</td><td>{bt['move']['SWM']['n']}</td><td>{fmt(bt['move']['SWM']['roi'],1)}</td><td>{bt['move']['SWM']['win']*100:.0f}%</td><td>{fmt(bt['move']['SWM']['sharpe'],d=2)}</td></tr>
<tr><td>always-YES</td><td>{bt['move']['always-YES']['n']}</td><td>{fmt(bt['move']['always-YES']['roi'],1)}</td><td>{bt['move']['always-YES']['win']*100:.0f}%</td><td>{fmt(bt['move']['always-YES']['sharpe'],d=2)}</td></tr>
<tr><td>random</td><td>{bt['move']['random']['n']}</td><td>{fmt(bt['move']['random']['roi'],1)}</td><td>{bt['move']['random']['win']*100:.0f}%</td><td>{fmt(bt['move']['random']['sharpe'],d=2)}</td></tr>
<tr><td rowspan=3>resolution<br>(持到结算)</td><td>SWM</td><td>{bt['resolution']['SWM']['n']}</td><td>{fmt(bt['resolution']['SWM']['roi'],1)}</td><td>{bt['resolution']['SWM']['win']*100:.0f}%</td><td>{fmt(bt['resolution']['SWM']['sharpe'],d=2)}</td></tr>
<tr><td>always-YES</td><td>{bt['resolution']['always-YES']['n']}</td><td>{fmt(bt['resolution']['always-YES']['roi'],1)}</td><td>{bt['resolution']['always-YES']['win']*100:.0f}%</td><td>{fmt(bt['resolution']['always-YES']['sharpe'],d=2)}</td></tr>
<tr><td>random</td><td>{bt['resolution']['random']['n']}</td><td>{fmt(bt['resolution']['random']['roi'],1)}</td><td>{bt['resolution']['random']['win']*100:.0f}%</td><td>{fmt(bt['resolution']['random']['sharpe'],d=2)}</td></tr>
</table>

<div class=card><h3 style='margin-top:0'>权益曲线(move 模式,SWM 累计 P&amp;L)</h3>{svg_curve(curves['move'])}</div>

<h3>SWM 阈值扫描(move 模式)</h3>
<table><tr><th>阈值</th><th>#trades</th><th>ROI</th><th>胜率</th><th>Sharpe</th></tr>{sweep_table(bt['move'])}</table>

<h2>C. 重要 caveat(别过度解读)</h2>
<ul class=sub>
<li><span class=warn>oracle 归因上界</span>:这里的预测来自用了"结果信息归因"的 eval 输出,是<b>乐观上界</b>。真正无泄漏版(启发式 pre-t 新闻均权归因、全市场自然分布、按时间 walk-forward)正在搭建/运行。</li>
<li><b>样本小 + 平衡集</b>:daily-7B 预测来自平衡后的 202 条 test(move),resolution 仅 {bt['resolution']['_n_used']} 条有 outcome。</li>
<li><b>绝对 ROI 虚高</b>:平衡集 + 低价 YES 的百分比收益放大,连 always-YES 都很高。<b>重点看 SWM 相对基线的超额</b>,不是绝对数字。</li>
<li><b>只在断点交易</b>:test 记录是预筛的价格大波动事件,非"扫描所有市场所有时刻"。</li>
</ul>
<p class=sub>生成:scripts/make_report.py · 数据 results/eval_polymarket/*.jsonl + {SWMBENCH}(outcome)</p>
</div>"""
OUT.write_text(html)
print("wrote", OUT, f"({OUT.stat().st_size//1024} KB)")
print("A/ stat cells:", sum(1 for _,_,m in statrows if m), "of", len(statrows))
print("B/ move SWM ROI", fmt(bt['move']['SWM']['roi'],1), "| resolution SWM ROI", fmt(bt['resolution']['SWM']['roi'],1))
