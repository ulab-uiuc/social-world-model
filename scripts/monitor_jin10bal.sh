#!/bin/bash
# Watch jin10bal training; kill teardown after save; then score the balanced
# test with the full metric suite (Pearson/Spearman/R2/MAE/dir-acc vs 50% base).
set -uo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model
J=$(cat logs/.jin10bal_train_jobid); LOG="logs/slurm-worldmodel-$J.out"
RES="results/eval_polymarket/jin10bal.jsonl"

for i in $(seq 1 240); do
  grep -q "Best model saved" "$LOG" 2>/dev/null && { echo "[monitor] best-model saved"; break; }
  squeue -h -j "$J" >/dev/null 2>&1 || { echo "[monitor] job gone"; break; }
  sleep 60
done
squeue -h -j "$J" -o "%T" 2>/dev/null | grep -q RUNNING && { echo "[monitor] killing teardown $J"; scancel "$J"; }

for i in $(seq 1 40); do
  n=$(wc -l < "$RES" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 60
done

python3 - <<'PY'
import json, math, statistics as s
def load(tag):
    ps=[];ts=[]
    for line in open(f'results/eval_polymarket/{tag}.jsonl'):
        r=json.loads(line); ps.append(r['pred_delta']); ts.append(r['true_delta'])
    return ps,ts
def pear(p,t):
    n=len(p);mp=sum(p)/n;mt=sum(t)/n
    cov=sum((a-mp)*(b-mt) for a,b in zip(p,t))/n;sp=s.pstdev(p);st=s.pstdev(t)
    return cov/(sp*st) if sp and st else float('nan')
def rank(x):
    o=sorted(range(len(x)),key=lambda i:x[i]);r=[0.0]*len(x);i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[o[j+1]]==x[o[i]]:j+=1
        for k in range(i,j+1):r[o[k]]=(i+j)/2+1
        i=j+1
    return r
p,t=load('jin10bal'); n=len(p)
mt=sum(t)/n; ss=sum((b-mt)**2 for b in t); sr=sum((a-b)**2 for a,b in zip(p,t))
mv=[(a,b) for a,b in zip(p,t) if abs(b)>0.01]; cm=[(a,b) for a,b in mv if abs(a)>1e-6]
cor=sum(1 for a,b in cm if (a>0)==(b>0)); up=sum(1 for a,b in mv if b>0)
print('===== jin10bal (DEBIASED, balanced temporal test) =====')
print(f'n={n}  n_move={len(mv)}')
print(f'Pearson  = {pear(p,t):.4f}')
print(f'Spearman = {pear(rank(p),rank(t)):.4f}')
print(f'R2       = {1-sr/ss if ss else float("nan"):.4f}')
print(f'dir_acc  = {cor/len(cm) if cm else float("nan"):.3f}  (coverage {len(cm)/len(mv) if mv else 0:.0%}, baseline {max(up,len(mv)-up)/len(mv) if mv else 0:.3f})')
PY
