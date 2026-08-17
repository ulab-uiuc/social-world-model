#!/usr/bin/env python3
"""Render 5 sample records from a jsonl into a standalone static HTML.

Usage: make_en_daily_viz.py [SRC.jsonl] [OUT.html] [TITLE]
"""
import json, html, sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/swmbench_jin10_attributed_filtered_en_daily.jsonl")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "viz/en_daily_sample.html")
TITLE = sys.argv[3] if len(sys.argv) > 3 else SRC.name
DST.parent.mkdir(exist_ok=True)


def delta(r):
    h = r.get("history") or []
    tp = (r.get("target") or {}).get("p")
    return (tp - h[-1]["p"]) if h and tp is not None else None


def npos(r):
    news = r.get("news") or []
    return sum(1 for a in (r.get("attributions") or [])
               if 0 <= a.get("news_idx", -1) < len(news) and float(a.get("score") or 0) > 0)


# pick 5 varied: fill up/consistent, up/inconsistent, down/consistent, down/inconsistent,
# then pad to 5 with the largest-|delta| remaining candidates.
buckets = {("up", True): None, ("up", False): None, ("dn", True): None, ("dn", False): None}
cands = []
for line in SRC.open():
    r = json.loads(line)
    d = delta(r)
    if d is None or npos(r) < 1 or abs(d) < 0.03:
        continue
    cands.append((abs(d), r))
    key = ("up" if d > 0 else "dn", bool(r.get("direction_consistent")))
    if key in buckets and buckets[key] is None:
        buckets[key] = r
    if len(cands) > 4000:
        break

picked = [v for v in buckets.values() if v]
seen = {id(x) for x in picked}
topics = set()
def topic(r):  # crude de-dup key: first 3 words of the question
    return " ".join((r.get("question") or "").lower().split()[:3])
for r in picked:
    topics.add(topic(r))
# pad to 5 with varied real moves (exclude settlement-like |delta|>0.6), unique topics
for _, r in sorted(cands, key=lambda x: -x[0]):
    if len(picked) >= 5:
        break
    d = abs(delta(r))
    if id(r) in seen or d > 0.6 or topic(r) in topics:
        continue
    picked.append(r); seen.add(id(r)); topics.add(topic(r))
# last resort: if still <5, allow any remaining
for _, r in sorted(cands, key=lambda x: -x[0]):
    if len(picked) >= 5:
        break
    if id(r) not in seen:
        picked.append(r); seen.add(id(r))
picked = picked[:5]


def trim(r):
    news = []
    attr = {a["news_idx"]: float(a.get("score") or 0) for a in (r.get("attributions") or [])
            if a.get("news_idx") is not None}
    for i, n in enumerate(r.get("news") or []):
        news.append({
            "title": (n.get("title") or "")[:200],
            "description": (n.get("description") or "")[:360],
            "published_at": n.get("published_at") or "",
            "source": n.get("source") or "",
            "offset_to_response_sec": n.get("offset_to_response_sec"),
            "score": round(attr.get(i, 0.0), 3),
        })
    news.sort(key=lambda x: -x["score"])
    return {
        "question": r.get("question"), "market_id": r.get("market_id"),
        "categories": r.get("categories") or [], "move_hour_t": r.get("move_hour_t"),
        "history": [{"t": p["t"], "p": p["p"]} for p in (r.get("history") or [])],
        "target": r.get("target") or {}, "before_2h": r.get("before_2h") or {},
        "change": r.get("change"), "change_2h": r.get("change_2h"),
        "z_score": r.get("z_score"), "direction_consistent": r.get("direction_consistent"),
        "news": news[:14],
    }


data = [trim(r) for r in picked]
print(f"selected {len(data)} records")

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — 5 samples</title>
<style>
:root{--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--price:#2a78d6;--target:#eb6834;
--good:#0ca30c;--crit:#d03b3b;--seq:#86b6ef;--chip:#eef1f4;}
@media(prefers-color-scheme:dark){:root{--surface-1:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink-2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--price:#3987e5;--target:#d95926;
--good:#0ca30c;--crit:#d03b3b;--seq:#1c5cab;--chip:#232322;}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:24px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.card{border:1px solid var(--border);border-radius:12px;background:var(--surface-1);padding:16px 18px;margin:16px 0}
.qh{font-size:16px;font-weight:650;margin:0 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.chip{background:var(--chip);border:1px solid var(--border);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--ink-2)}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.tile{border:1px solid var(--border);border-radius:10px;padding:8px 12px;min-width:110px}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.up{color:var(--good)}.dn{color:var(--crit)}
.badge{display:inline-flex;gap:5px;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;border:1px solid var(--border)}
.chartwrap{position:relative}svg{display:block;width:100%;height:280px;overflow:visible}
.axislbl{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 8px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);opacity:0;transition:opacity .08s;font-variant-numeric:tabular-nums;white-space:nowrap}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink-2);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
h2{font-size:13px;font-weight:650;margin:16px 0 8px;color:var(--ink-2)}
.nrow{display:grid;grid-template-columns:56px 1fr;gap:10px;padding:8px 0;border-top:1px solid var(--border)}
.nrow:first-child{border-top:0}.attr{border-left:3px solid var(--price);padding-left:9px;margin-left:-12px}
.score{font-variant-numeric:tabular-nums;font-size:12px}.sbar{height:5px;border-radius:3px;background:var(--seq);margin-top:4px}
.ntitle{font-weight:600}.ndesc{color:var(--ink-2);font-size:13px;margin-top:2px}.nmeta{color:var(--muted);font-size:11px;margin-top:3px}
</style></head><body><div class="wrap">
<h1>__TITLE__ — 5 sample records</h1>
<p class="sub">Price history · target after the move · attributed jin10 news. Hover the line for values.</p>
<div id="root"></div></div>
<script>
const DATA=__DATA__;
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmtT=t=>new Date(t*1000).toISOString().slice(0,16).replace("T"," ");
const fmt=(x,d=3)=>x==null?"–":(+x).toFixed(d);
const NS="http://www.w3.org/2000/svg";
function chart(box,tip,h,tt,tp,moveT,b2){
  const W=box.clientWidth||820,H=280,m={l:44,r:16,t:12,b:26};
  if(!h.length){box.textContent="no history";return;}
  const aT=h.map(p=>p.t).concat(tt?[tt]:[],moveT?[moveT]:[]);
  const aP=h.map(p=>p.p).concat(tp!=null?[tp]:[],b2&&b2.p!=null?[b2.p]:[]);
  const tMin=Math.min(...aT),tMax=Math.max(...aT);let pMin=Math.min(...aP),pMax=Math.max(...aP);
  const pad=(pMax-pMin)*.12||.02;pMin=Math.max(0,pMin-pad);pMax=Math.min(1,pMax+pad);
  const X=t=>m.l+(tMax===tMin?0:(t-tMin)/(tMax-tMin))*(W-m.l-m.r);
  const Y=p=>m.t+(1-(pMax===pMin?.5:(p-pMin)/(pMax-pMin)))*(H-m.t-m.b);
  const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`});
  for(let i=0;i<=4;i++){const p=pMin+(pMax-pMin)*i/4,y=Y(p);
    svg.appendChild(el("line",{x1:m.l,x2:W-m.r,y1:y,y2:y,stroke:"var(--grid)","stroke-width":1}));
    const t=el("text",{x:m.l-6,y:y+3,"text-anchor":"end",class:"axislbl"});t.textContent=p.toFixed(2);svg.appendChild(t);}
  [tMin,(tMin+tMax)/2,tMax].forEach((t,i)=>{const x=el("text",{x:X(t),y:H-8,"text-anchor":i===0?"start":i===2?"end":"middle",class:"axislbl"});x.textContent=fmtT(t).slice(5);svg.appendChild(x);});
  if(moveT)svg.appendChild(el("line",{x1:X(moveT),x2:X(moveT),y1:m.t,y2:H-m.b,stroke:"var(--muted)","stroke-width":1,"stroke-dasharray":"4 4"}));
  svg.appendChild(el("path",{d:h.map((p,i)=>(i?"L":"M")+X(p.t)+" "+Y(p.p)).join(" "),fill:"none",stroke:"var(--price)","stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
  const last=h[h.length-1];
  if(tt&&tp!=null){svg.appendChild(el("line",{x1:X(last.t),y1:Y(last.p),x2:X(tt),y2:Y(tp),stroke:"var(--target)","stroke-width":2,"stroke-dasharray":"3 3",opacity:.85}));
    svg.appendChild(el("circle",{cx:X(tt),cy:Y(tp),r:6,fill:"var(--target)",stroke:"var(--surface-1)","stroke-width":2}));
    const l=el("text",{x:X(tt),y:Y(tp)-10,"text-anchor":"end",class:"axislbl",fill:"var(--target)"});l.textContent="target "+tp.toFixed(3);svg.appendChild(l);}
  svg.appendChild(el("circle",{cx:X(last.t),cy:Y(last.p),r:4,fill:"var(--price)",stroke:"var(--surface-1)","stroke-width":2}));
  const cx=el("line",{x1:0,x2:0,y1:m.t,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,opacity:0});
  const dot=el("circle",{r:5,fill:"var(--price)",stroke:"var(--surface-1)","stroke-width":2,opacity:0});
  svg.appendChild(cx);svg.appendChild(dot);
  const hit=el("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:"transparent"});svg.appendChild(hit);
  hit.addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),sx=(ev.clientX-r.left)*W/r.width;
    let b=h[0],bd=1e9;for(const p of h){const d=Math.abs(X(p.t)-sx);if(d<bd){bd=d;b=p;}}
    cx.setAttribute("x1",X(b.t));cx.setAttribute("x2",X(b.t));cx.setAttribute("opacity",1);
    dot.setAttribute("cx",X(b.t));dot.setAttribute("cy",Y(b.p));dot.setAttribute("opacity",1);
    tip.style.opacity=1;tip.style.left=Math.min(W-120,X(b.t)+10)+"px";tip.style.top=(Y(b.p)-8)+"px";
    tip.innerHTML=`<b>${b.p.toFixed(3)}</b><br><span style="color:var(--muted)">${fmtT(b.t)}</span>`;});
  hit.addEventListener("mouseleave",()=>{cx.setAttribute("opacity",0);dot.setAttribute("opacity",0);tip.style.opacity=0;});
  box.appendChild(svg);
}
const root=document.getElementById("root");
DATA.forEach((r,idx)=>{
  const h=r.history||[],tp=(r.target||{}).p,tt=(r.target||{}).t,bp=h.length?h[h.length-1].p:null;
  const d=(bp!=null&&tp!=null)?tp-bp:null,up=d>0;
  const card=document.createElement("div");card.className="card";
  card.innerHTML=`<div class="qh">${idx+1}. ${esc(r.question)}</div>
    <div class="chips">${(r.categories||[]).map(c=>`<span class="chip">${esc(c)}</span>`).join("")}
      <span class="chip">move ${fmtT(r.move_hour_t||tt)}</span><span class="chip">market ${esc(String(r.market_id))}</span></div>
    <div class="tiles">
      <div class="tile"><div class="k">before → target</div><div class="v">${fmt(bp)} → ${fmt(tp)}</div></div>
      <div class="tile"><div class="k">Δ hist→target</div><div class="v ${up?'up':'dn'}">${d==null?"–":(up?"+":"")+d.toFixed(3)}</div></div>
      <div class="tile"><div class="k">change daily</div><div class="v">${fmt(r.change)}</div></div>
      <div class="tile"><div class="k">change 2h</div><div class="v">${fmt(r.change_2h)}</div></div>
      <div class="tile"><div class="k">z-score</div><div class="v">${fmt(r.z_score,2)}</div></div>
      <div class="tile"><div class="k">consistent</div><div class="v">${r.direction_consistent?"✓":"✗"}</div></div>
    </div>
    <div class="chartwrap"><div class="chart"></div><div class="tip"></div></div>
    <div class="legend"><span><i style="background:var(--price)"></i>hourly price (${h.length} pts)</span>
      <span><i style="background:var(--target)"></i>target (after move)</span>
      <span><i style="border:1px dashed var(--muted)"></i>move time</span></div>
    <h2>News &amp; attribution (${(r.news||[]).filter(n=>n.score>0).length} attributed / ${(r.news||[]).length})</h2>
    <div>${(r.news||[]).map(n=>`<div class="nrow ${n.score>0?'attr':''}">
      <div><div class="score">${n.score>0?n.score.toFixed(2):"–"}</div>${n.score>0?`<div class="sbar" style="width:${Math.max(6,n.score*100)}%"></div>`:""}</div>
      <div><div class="ntitle">${esc(n.title)}</div><div class="ndesc">${esc(n.description)}</div>
      <div class="nmeta">${esc(n.published_at)} · ${esc(n.source)}${n.offset_to_response_sec!=null?" · +"+(n.offset_to_response_sec/3600).toFixed(1)+"h":""}</div></div></div>`).join("")}</div>`;
  root.appendChild(card);
  chart(card.querySelector(".chart"),card.querySelector(".tip"),h,tt,tp,r.move_hour_t,r.before_2h);
});
</script></body></html>"""

DST.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__TITLE__", TITLE))
print(f"wrote {DST}  ({DST.stat().st_size//1024} KB)")
