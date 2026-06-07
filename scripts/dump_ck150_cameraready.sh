#!/bin/bash
set -o pipefail
ENVDIR=/home/haofeiy2/.conda/envs/agentgym
REPO=/home/haofeiy2/social-world-model; cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
MP=saves_local/fc8b_v9odds_semdedup_camera_ready_0603
PD=/mnt/disk2_from_server2/haofeiy2/swm_prior_attributed/v6_397bsem_8b_5ep_ckptscan
OUT="$REPO/dumps_ck150_cr"; LG="$REPO/logs/dump_ck150_cr"; mkdir -p "$OUT" "$LG"
GPUS="0 1 2 3 4 5 6 7 8 9"; NS=10
dump () {
  mk="$1"; src="$2"; outf="$OUT/fc8b_${mk}.jsonl"
  [ -s "$outf" ] && { echo "skip $mk"; return; }
  s=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g "$ENVDIR/bin/python" scripts/dump_per_news_predictions.py \
      --test-data-path "$src" --model-path "$MP" --model-name Qwen/Qwen3-8B \
      --output-path "$OUT/fc8b_${mk}_s${s}.jsonl" --max-news 0 \
      --num-shards $NS --shard-idx $s > "$LG/${mk}_s${s}.log" 2>&1 &
    s=$((s+1))
  done
  wait; cat "$OUT/fc8b_${mk}"_s*.jsonl > "$outf"; rm -f "$OUT/fc8b_${mk}"_s*.jsonl
  echo "dump $mk -> $(wc -l <"$outf") $(date +%T)"
}
dump kalshi "$PD/test_kalshi_final_ck150.jsonl"
dump poly   "$PD/test_polymarket_final_ck150.jsonl"
echo "=== DUMP_CK150_CR_DONE $(date +%T) ==="
