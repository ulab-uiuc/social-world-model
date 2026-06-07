#!/bin/bash
# Per-news mu dump for the 3 NEW 32B-trained forecasters, on the 32B-clean-semdedup
# test files (carry 32B oracle attributions). 8-GPU (0-7; 8/9 used by another user).
set -uo pipefail
ENVDIR=/home/haofeiy2/.conda/envs/agentgym
REPO=/home/haofeiy2/social-world-model; cd "$REPO"; export PYTHONPATH="$REPO"
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
DISK3=/mnt/disk3_from_server2/haofeiy2/swm_saves
D="$REPO/data/social-world-model-v6-qwen3-32B-clean-semdedup"
OUT="$REPO/dumps_32b"; LG="$REPO/logs/dump_32b"; mkdir -p "$OUT" "$LG"
GPUS="0 1 2 3 4 5 6 7"; NS=8
dump () {
  mp="$1"; mn="$2"; tag="$3"; mk="$4"; src="$5"; outf="$OUT/${tag}_${mk}.jsonl"
  [ -s "$outf" ] && { echo "skip $tag $mk"; return; }
  s=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g "$ENVDIR/bin/python" scripts/dump_per_news_predictions.py \
      --test-data-path "$src" --model-path "$mp" --model-name "$mn" \
      --output-path "$OUT/${tag}_${mk}_s${s}.jsonl" --max-news 0 \
      --num-shards $NS --shard-idx $s > "$LG/${tag}_${mk}_s${s}.log" 2>&1 &
    s=$((s+1))
  done
  wait; cat "$OUT/${tag}_${mk}"_s*.jsonl > "$outf"; rm -f "$OUT/${tag}_${mk}"_s*.jsonl
  echo "dump $tag $mk -> $(wc -l <"$outf") $(date +%T)"
}
for spec in "$DISK3/fc8b_v9odds_32Bposterior_semdedup/final-model Qwen/Qwen3-8B fc8b" "$DISK3/fc4b_v9odds_32Bposterior_semdedup/final-model Qwen/Qwen3-4B fc4b" "/mnt/disk3_from_server2/haofeiy2/swm_saves/fc06b_v9odds_32Bposterior_semdedup/checkpoint-3270 Qwen/Qwen3-0.6B fc06b"; do
  set -- $spec
  dump "$1" "$2" "$3" kalshi "$D/test_kalshi_final.jsonl"
  dump "$1" "$2" "$3" poly   "$D/test_polymarket_final.jsonl"
done
echo "=== DUMP_32B_DONE ==="
