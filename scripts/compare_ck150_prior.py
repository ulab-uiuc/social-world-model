"""Compare prior computed 3 ways, all camera-ready forecaster + ck150 prior scores:
  A) GPU-fresh dump on ck150 news, idx-aligned scores, ALL ck150 news (zero matching)
  B) GPU-fresh dump on ck150 news, restricted to TEST news (content-matched)
  C) offline reuse of dumps/ mu (test news) + ck150 content-matched scores  [current table]
direct soft routing, nonews=zero. If B ~= C, reusing dumps/ mu is validated.
"""

import json

import numpy as np

DD = 'data/social-world-model-v6-qwen3.5-397B-clean-semdedup'
PD = '/mnt/disk2_from_server2/haofeiy2/swm_prior_attributed/v6_397bsem_8b_5ep_ckptscan'


def st(x, t=1e-6):
    return 1 if x > t else -1 if x < -t else 0


def mets(P, T):
    P = np.array(P)
    T = np.array(T)
    mv = np.abs(T) > 1e-6
    mae = np.mean(np.abs(P - T))
    base = np.mean(np.abs(T))
    mase = mae / base if base > 0 else float('nan')
    da = (
        np.mean([st(a) == st(b) for a, b in zip(P[mv], T[mv])])
        if mv.any()
        else float('nan')
    )
    ic = np.corrcoef(P, T)[0, 1] if len(P) > 1 and np.std(P) > 0 else 0.0
    return mase, mae, da, ic


def nkey(x):
    return x.get('url') or (x.get('title', '') + '|' + x.get('description', '')[:40])


def direct(pairs):  # pairs: list of (score, mu)
    pairs = [(s, m) for s, m in pairs if s > 0]
    if not pairs:
        return 0.0
    w = np.clip([s for s, _ in pairs], 0, 1)
    return float(np.dot(w, [m for _, m in pairs]))


for plat, tf, pf in [
    ('poly', 'test_polymarket_final.jsonl', 'test_polymarket_final_ck150.jsonl'),
    ('kalshi', 'test_kalshi_final.jsonl', 'test_kalshi_final_ck150.jsonl'),
]:
    allk = set()
    attrk = set()
    testnews = {}
    for l in open(f'{DD}/{tf}'):
        d = json.loads(l)
        k = (d['market_id'], d['target']['t'])
        allk.add(k)
        testnews[k] = set(nkey(x) for x in (d.get('news') or []))
        if any(float(x.get('score') or 0) != 0 for x in (d.get('attributions') or [])):
            attrk.add(k)
    # fresh GPU dump on ck150: per_news mu (idx = ck150 news idx) + ck150 scores in attributions
    cr = {}
    for l in open(f'dumps_ck150_cr/fc8b_{plat}.jsonl'):
        r = json.loads(l)
        cr[(r['market_id'], r['t'])] = r
    # ck150 file: idx -> news nkey  (to know which ck150 idx is a test news)
    cknews = {}
    for l in open(f'{PD}/{pf}'):
        d = json.loads(l)
        k = (d['market_id'], (d.get('target') or {}).get('t'))
        cknews[k] = {i: nkey(x) for i, x in enumerate(d.get('news') or [])}
    # offline current: dumps/ mu (test news) + ck150 content-matched scores
    duf = {}
    for l in open(f'dumps/fc8b_{plat}.jsonl'):
        r = json.loads(l)
        duf[(r['market_id'], r['t'])] = {
            pn['news_idx']: pn['mu'] for pn in r['per_news']
        }
    ckscore_by_nkey = {}
    for l in open(f'{PD}/{pf}'):
        d = json.loads(l)
        k = (d['market_id'], (d.get('target') or {}).get('t'))
        nl = d.get('news') or []
        sc = {
            a['news_idx']: float(a.get('score') or 0)
            for a in (d.get('attributions') or [])
        }
        ckscore_by_nkey[k] = {nkey(nl[i]): sc.get(i, 0) for i in range(len(nl))}
    testnews_idx = {}
    for l in open(f'{DD}/{tf}'):
        d = json.loads(l)
        k = (d['market_id'], d['target']['t'])
        testnews_idx[k] = {i: nkey(x) for i, x in enumerate(d.get('news') or [])}

    RES = {m: {'all': ([], []), 'attr': ([], [])} for m in ('A', 'B', 'C')}
    for k in allk:
        if k not in cr:
            continue
        r = cr[k]
        td = r['true_delta']
        mu_ck = {pn['news_idx']: pn['mu'] for pn in r['per_news']}
        sc_ck = {
            a['news_idx']: float(a.get('score') or 0)
            for a in (r.get('attributions') or [])
        }
        # A: all ck150 news, idx-aligned
        predA = direct([(sc_ck.get(i, 0), mu_ck[i]) for i in mu_ck])
        # B: only ck150 news that are test news (content), idx-aligned
        tn = testnews.get(k, set())
        cn = cknews.get(k, {})
        predB = direct([(sc_ck.get(i, 0), mu_ck[i]) for i in mu_ck if cn.get(i) in tn])
        # C: dumps/ mu (test news) + content-matched ck150 score
        mu_du = duf.get(k, {})
        sm = ckscore_by_nkey.get(k, {})
        tni = testnews_idx.get(k, {})
        predC = direct([(sm.get(tni.get(i), 0), mu_du[i]) for i in mu_du])
        for m, p in (('A', predA), ('B', predB), ('C', predC)):
            for sub in ['all', 'attr'] if k in attrk else ['all']:
                RES[m][sub][0].append(p)
                RES[m][sub][1].append(td)
    print(f"\n##### {plat} (n_all={len(RES['A']['all'][0])}) #####")
    lab = {
        'A': 'GPU ck150 全news(零匹配)',
        'B': 'GPU ck150 仅test news',
        'C': '离线 dumps/+内容匹配(当前表)',
    }
    print(
        f"{'方法':>26}{'MASE_all':>9}{'MASE_at':>8}{'DA_all':>7}{'DA_at':>7}{'Corr_all':>9}{'Corr_at':>8}"
    )
    for m in ('A', 'B', 'C'):
        a = mets(*RES[m]['all'])
        t = mets(*RES[m]['attr'])
        print(
            f'{lab[m]:>26}{a[0]:>9.3f}{t[0]:>8.3f}{a[2]:>7.3f}{t[2]:>7.3f}{a[3]:>9.3f}{t[3]:>8.3f}'
        )
