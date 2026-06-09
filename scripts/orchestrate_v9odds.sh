#!/bin/bash
# After 8B (port 29580) finishes: move its model (relative-path save bug), then
# train 4B (GPU5-8) + 0.6B (GPU0) concurrently, then posterior-eval all 3 with
# the odds+null categorical aggregation on the semdedup test set.
set -o pipefail
REPO=/home/haofeiy2/social-world-model; cd "$REPO"
ENVDIR=/home/haofeiy2/.conda/envs/agentgym
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
DISK3=/mnt/disk3_from_server2/haofeiy2/swm_saves
D="$REPO/data/social-world-model-v6-qwen3.5-397B-clean-semdedup"
OUT="$REPO/results/v9odds_semdedup_posterior"; LG="$REPO/logs/v9odds_eval"; mkdir -p "$OUT" "$LG"

# 1. wait for 8B
while pgrep -f "master_port=29580" >/dev/null; do sleep 60; done
sleep 20
# 2. move 8B (saved under scripts/saves_local due to relative path)
if [ -d scripts/saves_local/fc8b_v9odds_semdedup ]; then
  rm -rf saves_local/fc8b_v9odds_semdedup
  mv scripts/saves_local/fc8b_v9odds_semdedup saves_local/
fi
echo "8B moved: shards=$(ls saves_local/fc8b_v9odds_semdedup/final-model/*.safetensors 2>/dev/null|wc -l) $(date +%T)"
# 3. train 4B (GPU5-8) + 0.6B (GPU0) as children
pkill -9 -f train_multievent_world_model 2>/dev/null; sleep 3
bash scripts/train_fc_v9odds.sh fsdp   5,6,7,8 4 29581 Qwen/Qwen3-4B   fc4b_v9odds_semdedup  "$DISK3" 6 &
bash scripts/train_fc_v9odds.sh single 0       1 29582 Qwen/Qwen3-0.6B fc06b_v9odds_semdedup "$DISK3" 6 &
wait
echo "4B+0.6B done $(date +%T)"
sleep 20

# 4. resolve paths
P8="saves_local/fc8b_v9odds_semdedup/final-model"
P4="$DISK3/fc4b_v9odds_semdedup/final-model"
P06=$(ls -d "$DISK3"/fc06b_v9odds_semdedup/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "models: 8B=$P8 4B=$P4 0.6B=$P06"

# 5. posterior eval (odds aggregation), 8-GPU sharded
GPUS="0 1 2 3 4 5 6 7"; NS=8
runeval () {
  mp="$1"; mn="$2"; tag="$3"; mk="$4"; src="$5"; outf="$OUT/${tag}_${mk}.jsonl"
  [ -s "$outf" ] && { echo "skip $tag $mk"; return; }
  s=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g "$ENVDIR/bin/python" scripts/inference_multievent_world_model.py \
      --test-data-path "$src" --model-path "$mp" --model-name "$mn" \
      --output-path "$OUT/${tag}_${mk}_s${s}.jsonl" --max-news 30 --batch-size 16 \
      --null-rho0 1.0 --odds-eps 1e-3 --odds-temp 1.0 \
      --num-shards $NS --shard-idx $s > "$LG/${tag}_${mk}_s${s}.log" 2>&1 &
    s=$((s+1))
  done
  wait; cat "$OUT/${tag}_${mk}"_s*.jsonl > "$outf"; rm -f "$OUT/${tag}_${mk}"_s*.jsonl
  echo "done $tag $mk -> $(wc -l <"$outf")"
}
runeval "$P8" Qwen/Qwen3-8B  fc8b  kalshi "$D/test_kalshi.jsonl"
runeval "$P8" Qwen/Qwen3-8B  fc8b  poly   "$D/test_polymarket.jsonl"
runeval "$P4" Qwen/Qwen3-4B  fc4b  kalshi "$D/test_kalshi.jsonl"
runeval "$P4" Qwen/Qwen3-4B  fc4b  poly   "$D/test_polymarket.jsonl"
runeval "$P06" Qwen/Qwen3-0.6B fc06b kalshi "$D/test_kalshi.jsonl"
runeval "$P06" Qwen/Qwen3-0.6B fc06b poly   "$D/test_polymarket.jsonl"

# 6. score
"$ENVDIR/bin/python" - <<PYEOF
import json, numpy as np, os
D="$D"; OUT="$OUT"
def zak(tf):
    z={};ak=set()
    for l in open(f"{D}/{tf}"):
        d=json.loads(l);n=d.get("news") or [];k=(d.get("market_id"),(d.get("target") or {}).get("t"));z[k]=d.get("z_score")
        if any(0<=x.get("news_idx",-1)<len(n) and float(x.get("score") or 0)>0 for x in (d.get("attributions") or [])):ak.add(k)
    return z,ak
def met(p,t):
    p=np.array(p,float);t=np.array(t,float);mae=np.mean(np.abs(p-t));b=np.mean(np.abs(t))
    mase=mae/b if b>0 else 9;mv=np.abs(t)>1e-6
    da=np.mean((p[mv]>0)==(t[mv]>0)) if mv.any() else 0;c=np.corrcoef(p,t)[0,1] if np.std(p)>0 else 0
    return mase,da,c,len(p)
print("\n===== v9-odds-semdedup POSTERIOR (odds aggregation) =====")
print(f"{'model':>6}{'mkt':>7}{'subset':>6}{'MASE':>8}{'DA':>7}{'Corr':>8}{'n':>6}")
for tag in ["fc8b","fc4b","fc06b"]:
    for mk,tf in [("kalshi","test_kalshi.jsonl"),("poly","test_polymarket.jsonl")]:
        pf=f"{OUT}/{tag}_{mk}.jsonl"
        if not os.path.exists(pf): print(f"{tag:>6}{mk:>7}  (missing)"); continue
        z,ak=zak(tf); rows=[json.loads(l) for l in open(pf)]
        for sub in ["all","attr","z>4"]:
            P=[];T=[]
            for r in rows:
                k=(r.get("market_id"),r.get("t"))
                keep=True if sub=="all" else (k in ak) if sub=="attr" else (z.get(k) is not None and z.get(k)>4)
                if keep: P.append(r["pred_delta"]);T.append(r["true_delta"])
            m=met(P,T); print(f"{tag:>6}{mk:>7}{sub:>6}{m[0]:>8.3f}{m[1]:>7.3f}{m[2]:>8.3f}{m[3]:>6}")
PYEOF
echo "=== V9ODDS_SEMDEDUP_EVAL_DONE ==="
