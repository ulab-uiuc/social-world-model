"""给 clean-semdedup 各文件加 future 字段(target 之后最多16个日点, 从 raw 取)。"""

import json
import os
import shutil

import numpy as np

P = 'data/prediction-market-data'
DD = 'data/social-world-model-v6-qwen3.5-397B-clean-semdedup'
FILES = [
    'test_kalshi.jsonl',
    'test_polymarket.jsonl',
    'train.jsonl',
    'train_with_nonzero_attribution.jsonl',
    'valid_with_nonzero_attribution.jsonl',
    'valid_subset150.jsonl',
]
DAY = 86400
K = 16
TOL = 43200

# 1. 收集需要的 market_id
need = set()
for f in FILES:
    for l in open(f'{DD}/{f}'):
        need.add(str(json.loads(l)['market_id']))
print(f'需要 market_id: {len(need)}')

# 2. raw poly (hourly, Yes token) — 只存需要的
rawp = {}
for l in open(f'{P}/polymarket/raw/polymarket_merged.jsonl'):
    d = json.loads(l)
    for m in d.get('markets') or []:
        mid = str(m.get('id'))
        if mid not in need:
            continue
        hist = m.get('history')
        if not isinstance(hist, dict) or not hist:
            continue
        try:
            key = json.loads(m['clobTokenIds'])[0]
        except Exception:
            key = list(hist.keys())[0]
        key = key if key in hist else list(hist.keys())[0]
        h = sorted(hist[key], key=lambda x: x['t'])
        rawp[mid] = (
            np.array([x['t'] for x in h], float),
            np.array([float(x['p']) for x in h]),
        )
# 3. raw kalshi (daily)
rawk = {}
for l in open(f'{P}/kalshi/raw/kalshi_data_raw.jsonl'):
    d = json.loads(l)
    mid = str(d['market_id'])
    if mid not in need:
        continue
    ts = sorted(d.get('time_series') or [], key=lambda x: x['t'])
    if ts:
        rawk[mid] = (
            np.array([x['t'] for x in ts], float),
            np.array([float(x['p']) for x in ts]),
        )
print(f'raw 命中: poly={len(rawp)} kalshi={len(rawk)}')


def future(mid, t0):
    src = rawk.get(mid) or rawp.get(mid)
    if src is None:
        return []
    ts, ps = src
    out = []
    for i in range(1, K + 1):
        tt = t0 + i * DAY
        j = int(np.argmin(np.abs(ts - tt)))
        if abs(ts[j] - tt) <= TOL:
            out.append({'t': int(tt), 'p': round(float(ps[j]), 6)})
        else:
            break
    return out


bak = os.path.join(DD, '_pre_future_backup')
os.makedirs(bak, exist_ok=True)
for f in FILES:
    rows = [json.loads(l) for l in open(f'{DD}/{f}')]
    lens = []
    for r in rows:
        fut = future(str(r['market_id']), float(r['target']['t']))
        r['future'] = fut
        r['n_future'] = len(fut)
        lens.append(len(fut))
    shutil.copy2(f'{DD}/{f}', os.path.join(bak, f))
    with open(f'{DD}/{f}', 'w') as w:
        for r in rows:
            w.write(json.dumps(r) + '\n')
    import numpy as _np

    print(
        f'{f}: n={len(rows)} future中位={int(_np.median(lens))} 满16={sum(1 for x in lens if x==16)} 无={sum(1 for x in lens if x==0)}'
    )
print('backup ->', bak)
