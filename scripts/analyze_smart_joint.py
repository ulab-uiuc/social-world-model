"""SMART joint evaluation: best-of-pipeline forecaster on K+P full test.

SMART joint = v42b (or v44) on has-news events + copy-p_before on null events.
Beats single-model approaches because v43/v43b null-LM is worse than copy.

Usage:
    python analyze_smart_joint.py
"""
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent  # social-world-model/


def load(path):
    p = Path(path)
    if not p.exists():
        return {}
    return {(r['market_id'], r.get('t')): r for r in
            [json.loads(l) for l in open(p)]}


def main():
    for label, oracle, v42bf, v44f in [
        ('KALSHI (1120)',
            BASE / 'data/vllm_attributed/combined_test_kalshi.jsonl',
            BASE / 'results/v42b_ck500_K.jsonl',
            BASE / 'results/v44_ck600_K.jsonl'),
        ('POLYMARKET (3692)',
            BASE / 'data/vllm_attributed/combined_test_polymarket.jsonl',
            BASE / 'results/v42b_ck500_P.jsonl',
            BASE / 'results/v44_ck600_P.jsonl'),
    ]:
        print(f'\n{"=" * 70}\n=== {label} ===\n{"=" * 70}')
        v42b = load(v42bf)
        v44 = load(v44f)

        pred_smart_v42b, pred_smart_v44, pred_copy, true_d, is_hn = [], [], [], [], []
        for r in [json.loads(l) for l in open(oracle)]:
            k = (r['market_id'], r.get('after', {}).get('t'))
            bp = r.get('before', {}).get('p', 0.5)
            ap = r.get('after', {}).get('p', 0.5)
            td = ap - bp
            true_d.append(td)
            hn = any((a.get('score') or 0) > 0 for a in (r.get('attributions') or []))
            is_hn.append(hn)
            pred_smart_v42b.append(v42b[k]['pred_delta'] if (hn and k in v42b) else 0.0)
            pred_smart_v44.append(v44[k]['pred_delta'] if (hn and k in v44) else 0.0)
            pred_copy.append(0.0)
        true_d = np.array(true_d)
        is_hn = np.array(is_hn)

        print(f'\n  {"model":35s}  {"MAE":>7s}  {"dirAcc":>8s}')
        for name, pd in [('copy-p_before baseline', pred_copy),
                          ('SMART joint v42b + copy', pred_smart_v42b),
                          ('SMART joint v44 + copy', pred_smart_v44)]:
            pd = np.array(pd)
            mae = np.abs(pd - true_d).mean()
            diracc = ((pd >= 0) == (true_d >= 0)).mean() * 100 if pd.std() > 1e-6 else 50.0
            print(f'  {name:35s}  {mae:>7.4f}  {diracc:>7.1f}%')

        print(f'\n  --- has-news segment only (n={int(is_hn.sum())}) ---')
        for name, lookup in [('v42b', v42b), ('v44', v44)]:
            preds, td_hn = [], []
            for r in [json.loads(l) for l in open(oracle)]:
                if any((a.get('score') or 0) > 0 for a in (r.get('attributions') or [])):
                    k = (r['market_id'], r.get('after', {}).get('t'))
                    preds.append(lookup[k]['pred_delta'] if k in lookup else 0.0)
                    td_hn.append(r.get('after', {}).get('p', 0.5) - r.get('before', {}).get('p', 0.5))
            preds = np.array(preds)
            td_hn = np.array(td_hn)
            mae = np.abs(preds - td_hn).mean()
            diracc = ((preds >= 0) == (td_hn >= 0)).mean() * 100 if preds.std() > 1e-6 else 50.0
            print(f'    {name:8s}  MAE={mae:.4f}  dirAcc={diracc:.1f}%')


if __name__ == '__main__':
    main()
