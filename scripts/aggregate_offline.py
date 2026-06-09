"""Offline aggregation over stored per-news mu_i (no GPU).

pred_delta = sum_i w_i * mu_i  (+ optional no-news term).
Routing schemes turn attribution scores -> weights w_i:
  direct : w_i = score_i               (prior softmax pi_i used as-is; no-news = 1-sum -> 0)
  odds   : o_i=((s+eps)/(1-s+eps))^(1/T); w_i = o_i/(rho0 + sum o)   (posterior口径)
  l1     : w_i = score_i / sum score_i (renormalize to 1; no no-news shrinkage)
Scores come from --scores-file's `attributions` (swap priors here!); default = the
mu-dump's own attributions. attr-subset is defined by ORACLE positives in --oracle-file.

Example:
  python scripts/aggregate_offline.py --mu-dump dumps/fc8b_kalshi.jsonl \
     --oracle-file data/.../test_kalshi_final.jsonl --scheme direct
"""

import argparse
import json

import numpy as np


def load_scores(path):
    """(market_id,t) -> {news_idx: score} from a file's attributions."""
    d = {}
    for l in open(path):
        r = json.loads(l)
        t = r.get('t') if 't' in r else (r.get('target') or {}).get('t')
        d[(r.get('market_id'), t)] = {
            a['news_idx']: float(a.get('score') or 0)
            for a in (r.get('attributions') or [])
        }
    return d


def attr_set(path):
    ak = set()
    for l in open(path):
        d = json.loads(l)
        n = d.get('news') or []
        if any(
            0 <= x.get('news_idx', -1) < len(n) and float(x.get('score') or 0) > 0
            for x in (d.get('attributions') or [])
        ):
            ak.add((d.get('market_id'), (d.get('target') or {}).get('t')))
    return ak


def weights(scores, scheme, rho0, eps, temp):
    s = np.clip(np.array(scores, float), 0.0, 1.0)
    if scheme == 'direct':
        return s
    if scheme == 'l1':
        tot = s.sum()
        return s / tot if tot > 0 else np.full_like(s, 1.0 / len(s)) if len(s) else s
    if scheme == 'odds':
        a = np.clip(s, 0.0, 1.0 - 1e-6)
        o = ((a + eps) / (1.0 - a + eps)) ** (1.0 / temp)
        return o / (rho0 + o.sum())
    raise ValueError(scheme)


def metrics(P, T):
    p = np.array(P, float)
    t = np.array(T, float)
    mae = np.mean(np.abs(p - t))
    b = np.mean(np.abs(t))
    mase = mae / b if b > 0 else float('nan')
    mv = np.abs(t) > 1e-6
    da = np.mean((p[mv] > 0) == (t[mv] > 0)) if mv.any() else float('nan')
    c = np.corrcoef(p, t)[0, 1] if np.std(p) > 0 else 0.0
    return mase, mae, da, c, len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu-dump', required=True)
    ap.add_argument(
        '--oracle-file', required=True, help='semdedup test file: defines attr-subset'
    )
    ap.add_argument(
        '--scores-file',
        default=None,
        help="attributions source for routing (default: the mu-dump's own).",
    )
    ap.add_argument('--scheme', default='direct', choices=['direct', 'odds', 'l1'])
    ap.add_argument('--rho0', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument(
        '--nonews',
        default='zero',
        choices=['zero', 'mu'],
        help='no-news contribution: zero (route-to-0) or its stored mu.',
    )
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.mu_dump)]
    scores_by = load_scores(args.scores_file) if args.scores_file else None
    ak = attr_set(args.oracle_file)

    res = {'all': ([], []), 'attr': ([], [])}
    for r in rows:
        k = (r.get('market_id'), r.get('t'))
        mu = {pn['news_idx']: pn['mu'] for pn in r['per_news']}
        if scores_by is not None:
            sc = scores_by.get(k, {})
        else:
            sc = {
                a['news_idx']: float(a.get('score') or 0)
                for a in (r.get('attributions') or [])
            }
        # route over POSITIVE-score news only (matches world_model _top_positives)
        idxs = [i for i in mu.keys() if i in sc and sc[i] > 0] if sc else []
        if idxs:
            w = weights(
                [sc[i] for i in idxs], args.scheme, args.rho0, args.eps, args.temp
            )
            pred = float(np.dot(w, [mu[i] for i in idxs]))
            if args.nonews == 'mu':
                wsum = float(np.sum(w))
                pred += max(0.0, 1.0 - wsum) * r.get('no_news_mu', 0.0)
        else:
            pred = r.get('no_news_mu', 0.0) if args.nonews == 'mu' else 0.0
        for sub in ['all', 'attr'] if k in ak else ['all']:
            res[sub][0].append(pred)
            res[sub][1].append(r['true_delta'])

    tag = (
        args.label
        or f'{args.scheme}(rho0={args.rho0},T={args.temp},nonews={args.nonews})'
    )
    print(
        f"\n== {tag} ==  (scores={'own' if not args.scores_file else args.scores_file.split('/')[-1]})"
    )
    print(f"{'subset':>7}{'MASE':>8}{'MAE':>9}{'DA':>7}{'Corr':>8}{'n':>7}")
    for sub in ['all', 'attr']:
        m = metrics(*res[sub])
        print(f'{sub:>7}{m[0]:>8.3f}{m[1]:>9.4f}{m[2]:>7.3f}{m[3]:>8.3f}{m[4]:>7}')


if __name__ == '__main__':
    main()
