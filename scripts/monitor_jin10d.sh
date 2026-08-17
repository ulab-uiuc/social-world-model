#!/bin/bash
# Monitor jin10d_bal (daily-history, 7B). Waits (patiently, GPU-contended) for
# best-model save; surfaces early failure (OOM/etc); kills teardown; scores the
# balanced daily test with full metrics.
set -uo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model
J=$(cat logs/.jin10d_bal_train_jobid); LOG="logs/slurm-worldmodel-$J.out"; ERR="logs/slurm-worldmodel-$J.err"
RES="results/eval_polymarket/jin10d_bal.jsonl"

for i in $(seq 1 720); do   # up to ~12h (contended queue)
  if grep -q "Best model saved" "$LOG" 2>/dev/null; then echo "[m] best-model saved"; break; fi
  if ! squeue -h -j "$J" >/dev/null 2>&1; then
    if ! grep -q "Best model saved" "$LOG" 2>/dev/null; then
      echo "[m] job $J ended WITHOUT saving — likely error:"
      grep -iE "error|oom|out of memory|traceback|cuda" "$ERR" 2>/dev/null | grep -viE "warn" | tail -8
      exit 0
    fi
  fi
  sleep 60
done
squeue -h -j "$J" -o "%T" 2>/dev/null | grep -q RUNNING && { echo "[m] killing teardown $J"; scancel "$J"; }

for i in $(seq 1 40); do n=$(wc -l < "$RES" 2>/dev/null || echo 0); [ "$n" -ge 180 ] && break; sleep 60; done

python3 - <<'PY'
import json,statistics as s
def M(tag):
    p=[];t=[]
    for line in open(f'results/eval_polymarket/{tag}.jsonl'):
        r=json.loads(line);p.append(r['pred_delta']);t.append(r['true_delta'])
    n=len(p);mp=sum(p)/n;mt=sum(t)/n
    cov=sum((a-mp)*(b-mt) for a,b in zip(p,t))/n;sp=s.pstdev(p);st=s.pstdev(t)
    pear=cov/(sp*st) if sp and st else float('nan')
    def rk(x):
        o=sorted(range(len(x)),key=lambda i:x[i]);r=[0.0]*len(x);i=0
        while i<len(x):
            j=i
            while j+1<len(x) and x[o[j+1]]==x[o[i]]:j+=1
            for k in range(i,j+1):r[o[k]]=(i+j)/2+1
            i=j+1
        return r
    sp_=pear if False else None
    ss=sum((b-mt)**2 for b in t);sr=sum((a-b)**2 for a,b in zip(p,t));r2=1-sr/ss if ss else float('nan')
    mv=[(a,b) for a,b in zip(p,t) if abs(b)>0.01];cm=[(a,b) for a,b in mv if abs(a)>1e-6]
    cor=sum(1 for a,b in cm if (a>0)==(b>0));up=sum(1 for a,b in mv if b>0)
    import math
    # spearman
    rp=rk(p);rt=rk(t);mrp=sum(rp)/n;mrt=sum(rt)/n
    cv=sum((a-mrp)*(b-mrt) for a,b in zip(rp,rt))/n;sprp=s.pstdev(rp);sprt=s.pstdev(rt)
    spear=cv/(sprp*sprt) if sprp and sprt else float('nan')
    return n,pear,spear,r2,(cor/len(cm) if cm else float('nan')),(max(up,len(mv)-up)/len(mv) if mv else 0)
print("===== jin10d_bal (DAILY history, 7B, balanced test) =====")
for tag,lbl in [('jin10d_bal','daily 7B'),('jin10bal','hourly 3B (ref)')]:
    try:
        n,pe,sp,r2,da,ba=M(tag)
        print(f"  {lbl:16s} n={n} Pearson={pe:.3f} Spearman={sp:.3f} R2={r2:.3f} dirAcc={da:.1%} (base {ba:.0%})")
    except FileNotFoundError: print(f"  {lbl}: (no result)")
PY
