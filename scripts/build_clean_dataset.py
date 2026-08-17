#!/usr/bin/env python3
"""Build a cleaned dataset from polymarket_cat5_attributed_2h.jsonl.

Filters:
  - keep bps where at least one news has attribution >= 25
  - drop settlement-like bps (before or after in [0, 0.02] or [0.98, 1])
  - drop title-party / digest news (keywords below) even from the news list
  - after news filtering, re-check the >=25 requirement on remaining news

History:
  - replaces bp.window_history (daily) with a 2h-granularity series drawn from
    raw polymarket history (hourly). 24 points, ending at move_hour_t - 2h so
    the series contains only pre-jump information.

Output schema (per line = one market):
  {
    "market_id", "question", "categories", "event_title",
    "breakpoints": [
      {
        "move_hour_t",           # unix seconds, the hour of the jump
        "before_p", "after_p",   # daily comparison (24h apart)
        "change", "z_score",
        "history_2h": [          # 24 pre-jump samples, 2h apart
          {"t": unix_sec, "p": prob}
        ],
        "news": [
          {"time_utc", "content", "offset_to_response_sec", "attribution"}
        ]
      }
    ]
  }
"""
import bisect
import json
import re
from pathlib import Path

SRC = Path("data/polymarket_cat5_attributed_2h.jsonl")
DST = Path("data/polymarket_cat5_clean.jsonl")
RAW = Path("swm-bench/raw/polymarket/raw/polymarket_merged.jsonl")

ATTRIBUTION_MIN = 25
SETTLE_LO, SETTLE_HI = 0.02, 0.98

HISTORY_N = 24            # number of samples
HISTORY_STEP = 2 * 3600   # step (default 2h)
HISTORY_END_OFFSET = 2 * 3600  # end at move_hour_t - step (exclude jump hour)

DIGEST_PATTERNS = [
    "金十图示", "每日汇总", "整理：每日", "整理:每日",
    "重点关注的财经数据与事件", "重点关注", "预告：", "预告:",
    "本周重要事件", "要闻速递", "要闻汇总",
    "24小时局势跟踪", "投行/机构观点梳理", "过去24小时都忙了什么",
]
DIGEST_RE = re.compile("|".join(re.escape(p) for p in DIGEST_PATTERNS))


def is_digest(content: str) -> bool:
    return bool(DIGEST_RE.search(content or ""))


def is_settlement(p_before: float, p_after: float) -> bool:
    def extreme(p):
        return p <= SETTLE_LO or p >= SETTLE_HI
    return extreme(p_before) or extreme(p_after)


def load_raw_history() -> dict:
    """market_id -> sorted list of (t, p) for the Yes side."""
    out = {}
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
            tokens = list(h.keys())
            yes_idx = 0
            if outs_list and "Yes" in outs_list:
                yes_idx = outs_list.index("Yes")
            hist = h[tokens[yes_idx]] if yes_idx < len(tokens) else h[tokens[0]]
            hist = sorted(hist, key=lambda x: x["t"])
            out[m["id"]] = ([p["t"] for p in hist], [p["p"] for p in hist])
    return out


def sample_history_2h(hist_ts, hist_ps, move_hour_t):
    """Return N points ending at move_hour_t - HISTORY_END_OFFSET, step 2h back."""
    if not hist_ts:
        return []
    end_t = move_hour_t - HISTORY_END_OFFSET
    targets = [end_t - i * HISTORY_STEP for i in range(HISTORY_N - 1, -1, -1)]
    out = []
    for t in targets:
        if t < hist_ts[0]:
            continue  # before the market existed
        i = bisect.bisect_right(hist_ts, t) - 1
        if i < 0:
            continue
        out.append({"t": t, "p": hist_ps[i]})
    return out


def clean_bp(bp: dict, hist_ts, hist_ps) -> dict | None:
    if is_settlement(bp["before"]["p"], bp["after"]["p"]):
        return None

    news = [
        {
            "time_utc": n["time_utc"],
            "content": n["content"],
            "offset_to_response_sec": n["offset_to_response_sec"],
            "attribution": n["attribution"],
        }
        for n in bp.get("news", [])
        if not is_digest(n.get("content", ""))
    ]
    if not any(n["attribution"] >= ATTRIBUTION_MIN for n in news):
        return None
    news.sort(key=lambda n: -n["attribution"])

    history_2h = sample_history_2h(hist_ts, hist_ps, bp["move_hour_t"])

    return {
        "move_hour_t": bp["move_hour_t"],
        "before_p": bp["before"]["p"],
        "after_p": bp["after"]["p"],
        "change": bp["change"],
        "z_score": bp.get("z_score"),
        "history_2h": history_2h,
        "news": news,
    }


def main():
    global DST, HISTORY_N, HISTORY_STEP, HISTORY_END_OFFSET
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--history-n", type=int, default=HISTORY_N)
    ap.add_argument("--history-step-hours", type=float, default=HISTORY_STEP / 3600)
    _a = ap.parse_args()
    DST = Path(_a.dst)
    HISTORY_N = _a.history_n
    HISTORY_STEP = int(_a.history_step_hours * 3600)
    HISTORY_END_OFFSET = HISTORY_STEP
    print(f"history: N={HISTORY_N} step={HISTORY_STEP}s -> {DST}")

    print(f"loading raw history from {RAW} ...")
    raw = load_raw_history()
    print(f"  {len(raw)} markets loaded")

    n_in = n_bps_in = n_bps_out = n_out = n_missing_hist = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            n_in += 1
            r = json.loads(line)
            mid = r["market_id"]
            hist_ts, hist_ps = raw.get(mid, ([], []))
            if not hist_ts:
                n_missing_hist += 1
            kept = []
            for bp in r["daily_breakpoints"]:
                n_bps_in += 1
                cleaned = clean_bp(bp, hist_ts, hist_ps)
                if cleaned:
                    kept.append(cleaned)
                    n_bps_out += 1
            if not kept:
                continue
            out = {
                "market_id": mid,
                "question": r["question"],
                "categories": r.get("categories", []),
                "event_title": r.get("event_title"),
                "breakpoints": kept,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"markets:     {n_in} -> {n_out}   (missing raw history: {n_missing_hist})")
    print(f"breakpoints: {n_bps_in} -> {n_bps_out}")
    print(f"output: {DST}")


if __name__ == "__main__":
    main()
