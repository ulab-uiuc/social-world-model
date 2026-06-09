"""Semantic dedup of candidate news per record (on top of exact-title dedup).
Embed each news (title+desc) with all-MiniLM-L6-v2, greedily cluster within each
record at cosine>=TAU, keep the highest-posterior-score representative per cluster,
and remap attributions (news_idx -> rep, score = max over cluster members).
Writes a new dataset dir. Usage: python semantic_dedup.py SRC DST [TAU]
"""

import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

SRC, DST = sys.argv[1], sys.argv[2]
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
os.makedirs(DST, exist_ok=True)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')


def greedy_cluster(sim, tau):
    n = len(sim)
    lab = [-1] * n
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = c
        for j in range(i + 1, n):
            if lab[j] < 0 and sim[i, j] >= tau:
                lab[j] = c
        c += 1
    return lab


FILES = [f for f in os.listdir(SRC) if f.endswith('.jsonl')]
print(f'TAU={TAU}')
for fn in sorted(FILES):
    recs = [json.loads(l) for l in open(os.path.join(SRC, fn))]
    # batch-encode all news across the file
    texts, span = [], []
    for d in recs:
        nw = d.get('news') or []
        s0 = len(texts)
        for n in nw:
            texts.append(
                (n.get('title', '') + '. ' + (n.get('description', '') or ''))[:400]
            )
        span.append((s0, len(texts)))
    emb = (
        model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if texts
        else np.zeros((0, 384))
    )
    tot_old = tot_new = 0
    out = []
    for d, (s0, s1) in zip(recs, span):
        news = d.get('news') or []
        attrs = d.get('attributions') or []
        score_by_idx = {
            a['news_idx']: float(a.get('score') or 0)
            for a in attrs
            if 0 <= a.get('news_idx', -1) < len(news)
        }
        tot_old += len(news)
        if len(news) <= 1:
            out.append(d)
            tot_new += len(news)
            continue
        E = emb[s0:s1]
        sim = E @ E.T
        lab = greedy_cluster(sim, TAU)
        # per cluster: pick representative = highest posterior score (tie -> lowest idx)
        clusters = {}
        for i, l in enumerate(lab):
            clusters.setdefault(l, []).append(i)
        reps = []  # (orig_idx) chosen rep per cluster, keep cluster order by min idx
        cluster_score = {}
        for l, members in clusters.items():
            rep = max(members, key=lambda i: (score_by_idx.get(i, 0.0), -i))
            reps.append((min(members), rep, members))
        reps.sort()  # stable order by first appearance
        new_news = []
        old2newrep = {}
        new_scores = {}
        for _, rep, members in reps:
            ni = len(new_news)
            new_news.append(news[rep])
            mx = max((score_by_idx.get(m, 0.0) for m in members), default=0.0)
            if mx > 0:
                new_scores[ni] = mx
            for m in members:
                old2newrep[m] = ni
        d2 = dict(d)
        d2['news'] = new_news
        d2['attributions'] = [
            {'news_idx': i, 'score': s} for i, s in sorted(new_scores.items())
        ]
        out.append(d2)
        tot_new += len(new_news)
    with open(os.path.join(DST, fn), 'w') as fo:
        fo.write('\n'.join(json.dumps(x) for x in out) + '\n')
    print(
        f'{fn:<28} records={len(recs):>5}  news {tot_old} -> {tot_new}  (-{100*(tot_old-tot_new)/max(tot_old,1):.0f}%)'
    )
print('DONE')
