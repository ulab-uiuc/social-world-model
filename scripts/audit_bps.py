#!/usr/bin/env python3
"""Audit every breakpoint in polymarket_cat5_clean.jsonl:
  - is move_hour_t a real single-hour jump (hourly |Δp| ≥ 0.03)?
  - how relevant is the top news to the question (keyword overlap heuristic)?

Outputs a plain-text report to /tmp/bp_audit.txt.
"""
import json
import bisect
import re
from pathlib import Path

RAW = Path("swm-bench/raw/polymarket/raw/polymarket_merged.jsonl")
CLEAN = Path("data/polymarket_cat5_clean.jsonl")
OUT = Path("/tmp/bp_audit.txt")

# ---------- load raw hourly history ----------
raw = {}
for line in RAW.open():
    r = json.loads(line)
    for m in r.get("markets", []):
        h = m.get("history") or {}
        if not h:
            continue
        outs = m.get("outcomes")
        try:
            outs_list = json.loads(outs) if isinstance(outs, str) else outs
        except Exception:
            outs_list = None
        toks = list(h.keys())
        yi = outs_list.index("Yes") if outs_list and "Yes" in outs_list else 0
        hist = sorted(h[toks[yi]], key=lambda x: x["t"])
        raw[m["id"]] = ([p["t"] for p in hist], [p["p"] for p in hist])


def p_at(mid, t):
    if mid not in raw:
        return None
    ts, ps = raw[mid]
    if not ts or t < ts[0]:
        return None
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return None
    # only accept if the sample is within 1h of t
    if abs(ts[i] - t) > 3600:
        return None
    return ps[i]


# ---------- crude relevance heuristic ----------
# Extract salient tokens from the question: tickers, all-caps words, dollar values,
# proper nouns, and asset/coin names. Then check if any of them appear in the news.

TICKERS = re.compile(
    r"\b(?:AAPL|AMZN|MSFT|GOOGL?|GOOG|META|NVDA|TSLA|NFLX|PLTR|ORCL|INTC|AVGO|VRT|ALNY|OPEN|WBD|RIVN|SPY|IBIT|COIN|MSTR)\b"
)
COINS = re.compile(
    r"\b(?:Bitcoin|BTC|Ethereum|ETH|Solana|SOL|XRP|Dogecoin|DOGE|BNB|Hyperliquid|Chainlink|LINK|Cardano|ADA|Zcash|ZEC|Pump\.fun|Uniswap|UNI|Aster|Plasma|Ethena|Paradex|Extended|Lighter|Theo|Infinex)\b",
    re.IGNORECASE,
)
COMMODITIES = {
    "gold": ["黄金", "金价"], "silver": ["白银", "银价"],
    "oil": ["原油", "油价", "WTI", "布伦特"], "crude": ["原油", "油价"],
    "bitcoin": ["比特币", "BTC"], "btc": ["比特币", "BTC"],
    "ethereum": ["以太", "ETH"], "eth": ["以太", "ETH"],
    "solana": ["Solana", "SOL"], "xrp": ["XRP"], "dogecoin": ["狗狗币", "DOGE"],
    "bnb": ["BNB", "币安"], "gdp": ["GDP", "经济增长", "增速"],
    "inflation": ["通胀", "CPI", "PPI"], "unemployment": ["失业", "非农", "就业"],
    "fed": ["美联储", "FOMC", "鲍威尔", "利率决议", "点阵图"],
    "ecb": ["欧洲央行", "欧央行", "ECB", "拉加德"],
    "bank of england": ["英国央行", "英央行", "贝利"],
    "bank of japan": ["日本央行", "日央行", "植田"],
    "brazil": ["巴西"], "colombia": ["哥伦比亚"], "australia": ["澳洲联储", "澳大利亚"],
    "mexico": ["墨西哥"], "canada": ["加拿大央行"], "korea": ["韩国"],
    "trump": ["特朗普", "Trump"], "putin": ["普京"], "zelenskyy": ["泽连斯基"],
    "netanyahu": ["内塔尼亚胡"], "modi": ["莫迪"],
    "israel": ["以色列", "以军", "以国防"],
    "gaza": ["加沙", "哈马斯"], "ukraine": ["乌克兰", "乌军", "乌方", "泽连斯基"],
    "russia": ["俄罗斯", "俄军", "俄方", "普京"],
    "iran": ["伊朗"], "venezuela": ["委内瑞拉", "马杜罗"],
    "maduro": ["马杜罗", "委内瑞拉"], "china": ["中国"],
    "openai": ["OpenAI"], "spacex": ["SpaceX", "马斯克"],
    "amazon": ["亚马逊", "AMZN"], "apple": ["苹果", "AAPL"],
    "microsoft": ["微软", "MSFT"], "google": ["谷歌", "GOOGL", "Alphabet"],
    "meta": ["Meta", "扎克伯格"], "nvidia": ["英伟达", "NVDA"],
    "tesla": ["特斯拉", "TSLA", "马斯克"],
    "palantir": ["Palantir", "PLTR"], "opendoor": ["Opendoor"],
    "alibaba": ["阿里巴巴", "阿里"], "abraham accords": ["亚伯拉罕协议"],
    "hegseth": ["赫格塞斯", "Hegseth"], "cook": ["库克", "Cook"],
    "noem": ["诺姆"], "hassett": ["哈塞特"], "leavitt": ["莱维特"],
    "powell": ["鲍威尔"], "bolsonaro": ["博索纳罗"],
    "lebanon": ["黎巴嫩"], "syria": ["叙利亚"],
}


def relevance(question: str, news_content: str) -> int:
    """Return 0/1/2: 0 no signal, 1 topical match, 2 strong (asset/name match)."""
    q = question.lower()
    n = news_content or ""
    hits = 0
    # asset / person name hits (strong)
    for key, patterns in COMMODITIES.items():
        if key in q and any(p in n for p in patterns):
            return 2
    # ticker
    m = TICKERS.search(question)
    if m and m.group(0) in n:
        return 2
    # crypto ticker
    m = COINS.search(question)
    if m and m.group(0).lower() in n.lower():
        return 2
    # weak: any 3+ char English word from question that also appears in news
    words = [w for w in re.findall(r"[A-Za-z]{4,}", question)
             if w.lower() not in {"will", "have", "with", "date", "week", "month",
                                  "hour", "over", "under", "before", "after",
                                  "reach", "settle", "close", "high", "level",
                                  "year", "the", "and", "for", "was", "his"}]
    for w in words:
        if w in n:
            hits = max(hits, 1)
    return hits


# ---------- audit loop ----------
recs = [json.loads(l) for l in CLEAN.open()]

counts = {"jump": 0, "smooth": 0, "unknown": 0}
rel_counts = {0: 0, 1: 0, 2: 0}
lines = []

for r in recs:
    mid = r["market_id"]
    for bp in r["breakpoints"]:
        mh = bp["move_hour_t"]
        p_prev = p_at(mid, mh - 3600)
        p_now = p_at(mid, mh)
        if p_prev is None or p_now is None:
            jump_tag = "UNK"
            counts["unknown"] += 1
            dp_hr = None
        else:
            dp_hr = p_now - p_prev
            if abs(dp_hr) >= 0.03:
                jump_tag = "JUMP"; counts["jump"] += 1
            else:
                jump_tag = "SMOOTH"; counts["smooth"] += 1

        top = bp["news"][0]
        rel = relevance(r["question"], top["content"])
        rel_counts[rel] += 1
        rel_tag = ["NO-MATCH", "TOPIC-MATCH", "STRONG-MATCH"][rel]

        dp_str = "  N/A " if dp_hr is None else f"{dp_hr:+.3f}"
        lines.append(
            f"[{jump_tag:6}][{rel_tag:12}] Δhr={dp_str:>7} "
            f"Δday={bp['change']:+.3f} attr={top['attribution']:>3} "
            f"| {r['question'][:75]}\n"
            f"    news: {top['content'][:100]}\n"
        )

with OUT.open("w") as f:
    f.write(f"total bps: {sum(counts.values())}\n")
    f.write(f"jump/smooth/unknown: {counts}\n")
    f.write(f"relevance (0=no,1=topic,2=strong): {rel_counts}\n")
    f.write("\n" + "="*80 + "\n\n")
    f.writelines(lines)

print(f"wrote {OUT}")
print(f"total bps: {sum(counts.values())}")
print(f"jump/smooth/unknown: {counts}")
print(f"relevance: {rel_counts}")

# cross-tab
print("\nJump × Relevance cross-tab:")
xtab = {}
for r in recs:
    for bp in r["breakpoints"]:
        mid = r["market_id"]
        mh = bp["move_hour_t"]
        p_prev = p_at(mid, mh - 3600); p_now = p_at(mid, mh)
        if p_prev is None or p_now is None:
            j = "unknown"
        elif abs(p_now - p_prev) >= 0.03:
            j = "jump"
        else:
            j = "smooth"
        rel = relevance(r["question"], bp["news"][0]["content"])
        xtab[(j, rel)] = xtab.get((j, rel), 0) + 1
print(f"{'':10}  {'no':>6}  {'topic':>6}  {'strong':>6}")
for j in ["jump", "smooth", "unknown"]:
    row = [xtab.get((j, r), 0) for r in [0, 1, 2]]
    print(f"  {j:8}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}")
