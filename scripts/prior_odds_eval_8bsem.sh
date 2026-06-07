#!/bin/bash
# Deployment口径 with CORRECT prior attributor (v6_397bsem_8b_odds_sem):
# run 3 odds forecasters + odds aggregation on the pre-attributed prior files.
# attr subset defined by GROUND-TRUTH attributions in the semdedup test files.
set -o pipefail
ENVDIR=/home/haofeiy2/.conda/envs/agentgym
REPO=/home/haofeiy2/social-world-model; cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
DISK3=/mnt/disk3_from_server2/haofeiy2/swm_saves
D="$REPO/data/social-world-model-v6-qwen3.5-397B-clean-semdedup"
PD=/mnt/disk2_from_server2/haofeiy2/swm_prior_attributed/v6_397bsem_8b_odds_sem
KSRC="$PD/test_kalshi_final_prior_v6_397bsem_8b_odds_sem.jsonl"
PSRC="$PD/test_polymarket_final_prior_v6_397bsem_8b_odds_sem.jsonl"
OUT="$REPO/results/v9odds_prior8bsem"; LG="$REPO/logs/prior_odds_8bsem"; mkdir -p "$OUT" "$LG"
GPUS="0 1 2 3 4 5 6 7 8 9"; NS=10

fore () {
  mp="$1"; mn="$2"; tag="$3"; mk="$4"; src="$5"; outf="$OUT/${tag}_${mk}.jsonl"
  [ -s "$outf" ] && { echo "skip $tag $mk"; return; }
  s=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g "$ENVDIR/bin/python" scripts/inference_multievent_forecaster.py \
      --test-data-path "$src" --model-path "$mp" --model-name "$mn" \
      --output-path "$OUT/${tag}_${mk}_s${s}.jsonl" --max-news 30 --batch-size 16 \
      --direct-soft-routing \
      --num-shards $NS --shard-idx $s > "$LG/${tag}_${mk}_s${s}.log" 2>&1 &
    s=$((s+1))
  done
  wait; cat "$OUT/${tag}_${mk}"_s*.jsonl > "$outf"; rm -f "$OUT/${tag}_${mk}"_s*.jsonl
  echo "fore $tag $mk -> $(wc -l <"$outf") $(date +%T)"
}
P06=$(ls -d "$DISK3"/fc06b_v9odds_semdedup/checkpoint-* 2>/dev/null|sort -t- -k2 -n|tail -1)
for spec in "saves_local/fc8b_v9odds_semdedup/final-model Qwen/Qwen3-8B fc8b" "$DISK3/fc4b_v9odds_semdedup/final-model Qwen/Qwen3-4B fc4b" "$P06 Qwen/Qwen3-0.6B fc06b"; do
  set -- $spec
  fore "$1" "$2" "$3" kalshi "$KSRC"
  fore "$1" "$2" "$3" poly   "$PSRC"
done

"$ENVDIR/bin/python" - <<PYEOF
import json, numpy as np, os
D="$D"; OUT="$OUT"
def zak(tf):
    ak=set()
    for l in open(f"{D}/{tf}"):
        d=json.loads(l);n=d.get("news") or []
        if any(0<=x.get("news_idx",-1)<len(n) and float(x.get("score") or 0)>0 for x in (d.get("attributions") or [])):
            ak.add((d.get("market_id"),(d.get("target") or {}).get("t")))
    return ak
def met(p,t):
    p=np.array(p,float);t=np.array(t,float);mae=np.mean(np.abs(p-t));b=np.mean(np.abs(t))
    mase=mae/b if b>0 else 9;mv=np.abs(t)>1e-6
    da=np.mean((p[mv]>0)==(t[mv]>0)) if mv.any() else 0;c=np.corrcoef(p,t)[0,1] if np.std(p)>0 else 0
    return mase,mae,da,c,len(p)
print("\n===== v9-odds-semdedup, PRIOR(8b odds sem) DIRECT soft routing =====")
print(f"{'model':>6}{'mkt':>7}{'subset':>6}{'MASE':>8}{'MAE':>9}{'DA':>7}{'Corr':>8}{'n':>6}")
for tag in ["fc8b","fc4b","fc06b"]:
    for mk,tf in [("kalshi","test_kalshi_final.jsonl"),("poly","test_polymarket_final.jsonl")]:
        pf=f"{OUT}/{tag}_{mk}.jsonl"
        if not os.path.exists(pf): print(f"{tag:>6}{mk:>7}  (missing)"); continue
        ak=zak(tf); rows=[json.loads(l) for l in open(pf)]
        for sub in ["all","attr"]:
            P=[];T=[]
            for r in rows:
                k=(r.get("market_id"),r.get("t"))
                if sub=="all" or k in ak: P.append(r["pred_delta"]);T.append(r["true_delta"])
            m=met(P,T); print(f"{tag:>6}{mk:>7}{sub:>6}{m[0]:>8.3f}{m[1]:>9.4f}{m[2]:>7.3f}{m[3]:>8.3f}{m[4]:>6}")
PYEOF
echo "=== PRIOR_ODDS_8BSEM_DONE ==="
