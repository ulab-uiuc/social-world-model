"""Build social-world-model-v6-qwen3-32B-clean-semdedup:
clone of the 397B-clean-semdedup (B), with ONLY each attribution's score replaced
by the Qwen3-32B score (from social-world-model-v6 = A), matched news-by-news on
(title, description) since A and B order news differently. Everything else
(z_score, news, history, target, attribution news_idx structure, files) identical to B.
"""
import json, os

A="data/social-world-model-v6"
B="data/social-world-model-v6-qwen3.5-397B-clean-semdedup"
OUT="data/social-world-model-v6-qwen3-32B-clean-semdedup"
BFILES=["test_kalshi_final.jsonl","test_polymarket_final.jsonl","train.jsonl",
        "train_clean.jsonl","valid_clean.jsonl","valid_subset150.jsonl"]

def nk(n): return ((n.get("title") or "").strip(), (n.get("description") or "").strip())

# A union: key -> {(title,desc): 32B score (max over dups)}
Amap={}
for af in ["train.jsonl","test_kalshi.jsonl","test_polymarket.jsonl"]:
    for l in open(f"{A}/{af}"):
        r=json.loads(l); k=(r["market_id"],(r.get("target") or {}).get("t"))
        an=r.get("news") or []; m={}
        for a in (r.get("attributions") or []):
            i=a.get("news_idx")
            if i is None or not(0<=i<len(an)): continue
            key=nk(an[i]); s=float(a.get("score") or 0.0)
            m[key]=max(m.get(key,0.0), s)        # dup content -> max score
        Amap[k]=m

os.makedirs(OUT, exist_ok=True)
for bf in BFILES:
    bp=f"{B}/{bf}"
    if not os.path.exists(bp): print("SKIP",bf); continue
    n=0; miss_key=0; miss_news=0; tot_news=0; pos=0
    with open(f"{OUT}/{bf}","w") as w:
        for l in open(bp):
            r=json.loads(l); k=(r["market_id"],(r.get("target") or {}).get("t"))
            bn=r.get("news") or []; m=Amap.get(k); 
            if m is None: miss_key+=1; m={}
            new_attr=[]
            for a in (r.get("attributions") or []):
                j=a.get("news_idx")
                if j is None or not(0<=j<len(bn)):
                    new_attr.append({"news_idx":j,"score":0.0}); continue
                tot_news+=1
                s=m.get(nk(bn[j]))
                if s is None: miss_news+=1; s=0.0
                if s>0: pos+=1
                new_attr.append({"news_idx":j,"score":s})
            r["attributions"]=new_attr
            w.write(json.dumps(r)+"\n"); n+=1
    print(f"{bf}: n={n} miss_key={miss_key} news未匹配={miss_news}/{tot_news} 正分条目={pos}")
print("DONE ->",OUT)
