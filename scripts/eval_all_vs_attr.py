#!/usr/bin/env python3
"""Compute baseline metrics on two subsets and emit a combined TSV:

  all  = every datapoint in the test file
  attr = datapoints whose news is NOT all-zero attribution
         (defined from the clean-semdedup test_*_final.jsonl files,
          matched to each baseline file by (market_id, history fingerprint))

Metric logic is identical to run_gpt55_baseline.compute_metrics_for_file
(no z_score filter).  n_moved_* = count with z_score > 4 (auxiliary).

Output column order:
  split baseline method n_all n_moved_all n_attr n_moved_attr
  MASE_all MASE_attr MAE_all MAE_attr DA_all DA_attr Corr_all Corr_attr
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "social-world-model-v6-qwen3.5-397B-clean-semdedup"
# "moved" subset: keep records whose market actually moved, i.e. |delta| >= DELTA_MIN
# where delta = target.p - last_price (filter out |delta| < 1e-6).
# DA is computed over this moved subset; MASE/MAE/Corr stay on the full subset.
DELTA_MIN = 1e-6
METHOD_RENAME = {"Qwen3-8B-NewsForecast": "fnf"}


def is_moved(r):
    delta = target_p(r) - float(r["history"][-1]["p"])
    return abs(delta) > DELTA_MIN


def load(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def hist_fp(history):
    return tuple((round(float(h["t"]), 0), round(float(h["p"]), 6)) for h in history)


def corr(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def split_of(path):
    name = path.stem
    if name.endswith("_kalshi"):
        return "kalshi"
    if name.endswith("_polymarket"):
        return "polymarket"
    return None


def baseline_of(path):
    name = path.stem
    for suf in ("_kalshi", "_polymarket"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def method_of(records, path):
    if records and records[0].get("method"):
        return METHOD_RENAME.get(records[0]["method"], records[0]["method"])
    return baseline_of(path)


def target_p(r):
    t = r["target"]
    return float(t["p"]) if isinstance(t, dict) else float(t)


def sign_tol(x, tol=1e-6):
    """Three-way sign with a dead band: |x| <= tol counts as 0 (no change)."""
    return 1 if x > tol else -1 if x < -tol else 0


def _da(records):
    """Directional accuracy over the given record list.
    Match rule = sign_tol(pred_delta) == sign_tol(true_delta), where a delta
    within +/-1e-6 counts as 'no change' (0)."""
    if not records:
        return float("nan")
    preds = [float(r["prediction"]) for r in records]
    targets = [target_p(r) for r in records]
    last_prices = [float(r["history"][-1]["p"]) for r in records]
    pred_delta = [p - lp for p, lp in zip(preds, last_prices)]
    true_delta = [t - lp for t, lp in zip(targets, last_prices)]
    return sum(1 for a, b in zip(pred_delta, true_delta) if sign_tol(a) == sign_tol(b)) / len(records)


def metrics(records):
    """MASE/MAE/Corr over the full record list (compute_metrics_for_file logic);
    DA over the 'moved' subset (z_score >= Z_MIN)."""
    if not records:
        return None
    preds = [float(r["prediction"]) for r in records]
    targets = [target_p(r) for r in records]
    last_prices = [float(r["history"][-1]["p"]) for r in records]
    n = len(records)
    errors = [p - t for p, t in zip(preds, targets)]
    p_errors = [lp - t for lp, t in zip(last_prices, targets)]
    mae = sum(abs(e) for e in errors) / n
    p_mae = sum(abs(e) for e in p_errors) / n
    pred_delta = [p - lp for p, lp in zip(preds, last_prices)]
    true_delta = [t - lp for t, lp in zip(targets, last_prices)]
    moved = [r for r in records if is_moved(r)]
    return {
        "n": n,
        "n_moved": len(moved),
        "mae": mae,
        "mase": mae / p_mae if p_mae else float("nan"),
        "da": _da(moved),                      # DA over moved subset only
        "corr": corr(pred_delta, true_delta),  # Corr over full subset
    }


def load_attr_keys():
    """Return {split: set of (market_id, history_fp) that are in the attr subset}."""
    keys = {}
    for split in ("kalshi", "polymarket"):
        rows = load(CLEAN / f"test_{split}_final.jsonl")
        s = set()
        for r in rows:
            attrs = r.get("attributions") or []
            if any(float(a.get("score", 0)) != 0 for a in attrs):
                s.add((r["market_id"], hist_fp(r["history"])))
        keys[split] = s
    return keys


def fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.10f}"
    return str(v)


def main():
    attr_keys = load_attr_keys()
    FIELDS = [
        "split", "baseline", "method",
        "n_all", "n_moved_all", "n_attr", "n_moved_attr",
        "MASE_all", "MASE_attr", "MAE_all", "MAE_attr",
        "DA_all", "DA_attr", "Corr_all", "Corr_attr",
    ]
    rows = []
    for p in sorted(ROOT.glob("*_*.jsonl")):
        split = split_of(p)
        if split is None:
            continue
        recs = load(p)
        recs = [r for r in recs if r.get("prediction") is not None]
        method = method_of(recs, p)
        baseline = baseline_of(p)

        keyset = attr_keys[split]
        attr_recs = [r for r in recs if (r["market_id"], hist_fp(r["history"])) in keyset]

        m_all = metrics(recs)
        m_attr = metrics(attr_recs)
        rows.append({
            "split": split,
            "baseline": baseline,
            "method": method,
            "n_all": m_all["n"],
            "n_moved_all": m_all["n_moved"],
            "n_attr": m_attr["n"] if m_attr else 0,
            "n_moved_attr": m_attr["n_moved"] if m_attr else 0,
            "MASE_all": m_all["mase"],
            "MASE_attr": m_attr["mase"] if m_attr else float("nan"),
            "MAE_all": m_all["mae"],
            "MAE_attr": m_attr["mae"] if m_attr else float("nan"),
            "DA_all": m_all["da"],
            "DA_attr": m_attr["da"] if m_attr else float("nan"),
            "Corr_all": m_all["corr"],
            "Corr_attr": m_attr["corr"] if m_attr else float("nan"),
        })

    split_order = {"kalshi": 0, "polymarket": 1}
    rows.sort(key=lambda r: (split_order.get(r["split"], 9), r["method"].lower()))

    out_path = ROOT / "metrics_all_vs_attr.tsv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in FIELDS})
    print(f"wrote {out_path} ({len(rows)} rows)")
    for split in ("kalshi", "polymarket"):
        sub = [r for r in rows if r["split"] == split]
        if sub:
            print(f"  {split}: {len(sub)} baselines | n_all={sub[0]['n_all']} n_attr={sub[0]['n_attr']} "
                  f"n_moved_attr={sub[0]['n_moved_attr']}")


if __name__ == "__main__":
    main()
