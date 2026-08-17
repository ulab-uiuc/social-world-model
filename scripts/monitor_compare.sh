#!/bin/bash
# Watch daily-7B (jin10d_bal) + pm-7B; kill each teardown after best-model save;
# wait for both evals; print the daily-vs-baseline comparison + history.
set -uo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model
DJ=$(cat logs/.jin10d_bal_train_jobid); PJ=$(cat logs/.pm_7b_train_jobid)

watch_kill(){ local j=$1 log="logs/slurm-worldmodel-$1.out";
  for i in $(seq 1 600); do
    grep -q "Best model saved" "$log" 2>/dev/null && break
    squeue -h -j "$j" >/dev/null 2>&1 || return 0
    sleep 60
  done
  squeue -h -j "$j" -o "%T" 2>/dev/null | grep -q RUNNING && { echo "[m] kill teardown $j"; scancel "$j"; }; }

watch_kill "$DJ" & watch_kill "$PJ" & wait
echo "[m] both trainings done"

for i in $(seq 1 480); do
  a=$(wc -l < results/eval_polymarket/jin10d_bal.jsonl 2>/dev/null||echo 0)
  b=$(wc -l < results/eval_polymarket/pm_7b.jsonl 2>/dev/null||echo 0)
  [ "$a" -ge 180 ] && [ "$b" -ge 3000 ] && break; sleep 60
done

python3 - <<'PY'
import json,statistics as s,os
def rk(x):
    o=sorted(range(len(x)),key=lambda i:x[i]);r=[0.0]*len(x);i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[o[j+1]]==x[o[i]]:j+=1
        for k in range(i,j+1):r[o[k]]=(i+j)/2+1
        i=j+1
    return r
def pear(a,b):
    n=len(a);ma=sum(a)/n;mb=sum(b)/n
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))/n;sa=s.pstdev(a);sb=s.pstdev(b)
    return cov/(sa*sb) if sa and sb else float('nan')
def M(tag):
    p=[];t=[]
    fn=f'results/eval_polymarket/{tag}.jsonl'
    if not os.path.exists(fn): return None
    for line in open(fn):
        r=json.loads(line);p.append(r['pred_delta']);t.append(r['true_delta'])
    n=len(p);mt=sum(t)/n
    ss=sum((y-mt)**2 for y in t);sr=sum((x-y)**2 for x,y in zip(p,t));r2=1-sr/ss if ss else float('nan')
    mv=[(x,y) for x,y in zip(p,t) if abs(y)>0.01];cm=[(x,y) for x,y in mv if abs(x)>1e-6]
    cor=sum(1 for x,y in cm if (x>0)==(y>0));up=sum(1 for x,y in mv if y>0)
    return n,pear(p,t),pear(rk(p),rk(t)),r2,(cor/len(cm) if cm else float('nan')),(max(up,len(mv)-up)/len(mv) if mv else 0)
rows=[
 ('jin10d_bal','NEW daily-7B / jin10 daily test(bal)'),
 ('pm_7b','pm-7B / swm-bench test_pm'),
 ('jin10bal','jin10 hourly-3B / hourly test(bal) [ref]'),
 ('pm','pm-3B ORIG / swm-bench test_pm [ref]'),
 ('e_cdpm','e_cdpm-3B / swm-bench test_pm [ref]'),
]
print("%-42s %5s %8s %8s %7s %8s %6s"%("model / test","n","Pear","Spear","R2","dirAcc","base"))
print("-"*92)
for tag,lbl in rows:
    m=M(tag)
    if not m: print("%-42s  (no result)"%lbl); continue
    n,pe,sp,r2,da,ba=m
    print("%-42s %5d %8.3f %8.3f %7.3f %7.1f%% %5.0f%%"%(lbl,n,pe,sp,r2,da*100,ba*100))
PY
