#!/bin/bash
# Sharded prior-attribution inference across 8 GPUs for one checkpoint.
# Usage: shard_infer_8b.sh <checkpoint_dir> <tag> <model> [gpus]
#   tag -> output filenames; e.g. step300 / step600 / final
# Produces, per test split, one merged jsonl with per-news prior attribution scores:
#   $OUTROOT/<tag>/<split>_prior_<tag>.jsonl
CK=$1; TAG=$2; MODEL=${3:-Qwen/Qwen3-8B}; GPUS=${4:-1,2,3,4,5,7,8,9}
IFS=',' read -ra G <<< "$GPUS"; N=${#G[@]}

source ~/anaconda3/etc/profile.d/conda.sh; conda activate social-wm
cd /home/haofeiy2/social-world-model
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

OUTROOT=/mnt/disk2_from_server2/haofeiy2/swm_prior_attributed
OUT=$OUTROOT/$TAG; mkdir -p "$OUT"
LOGD=/home/haofeiy2/social-world-model/logs/shardinfer_$TAG; mkdir -p "$LOGD"

for SPLIT in test_kalshi_final test_polymarket_final; do
  echo "=== $TAG / $SPLIT : launching $N shards on GPUs $GPUS ==="
  pids=()
  for i in $(seq 0 $((N-1))); do
    g=${G[$i]}
    CUDA_VISIBLE_DEVICES=$g python scripts/inference_prior_attribution.py \
      --data-path ${EVALDATADIR:-data/v6_qwen3.5_397b_clean_semdedup}/$SPLIT.jsonl \
      --attributer-path "$CK" --model-name "$MODEL" \
      --output-path "$OUT/${SPLIT}_prior_${TAG}.shard${i}.jsonl" \
      --max-news 30 --num-shards $N --shard-idx $i \
      > "$LOGD/${SPLIT}.shard${i}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait $p || fail=1; done
  # merge shards (record order interleaved but each record intact)
  cat "$OUT/${SPLIT}_prior_${TAG}".shard*.jsonl > "$OUT/${SPLIT}_prior_${TAG}.jsonl"
  n=$(wc -l < "$OUT/${SPLIT}_prior_${TAG}.jsonl")
  echo "=== $TAG / $SPLIT : merged $n records -> $OUT/${SPLIT}_prior_${TAG}.jsonl (fail=$fail) ==="
  rm -f "$OUT/${SPLIT}_prior_${TAG}".shard*.jsonl
done
echo "SHARD-INFER DONE tag=$TAG -> $OUT"
ls -la "$OUT"
