#!/bin/bash
# Upload a data file to a HuggingFace DATASET repo (default: swmbench/swmbench).
# RUN ON A NETWORKED NODE. Auth via HF_TOKEN env (write token):
#   HF_TOKEN=hf_xxx bash scripts/upload_dataset.sh  [LOCAL_FILE] [HF_REPO] [PATH_IN_REPO]
# Defaults: data/swmbench_jin10_dailyhist_en.jsonl -> swmbench/swmbench (repo root)
set -euo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model

LOCAL="${1:-data/swmbench_jin10_dailyhist_en.jsonl}"
REPO="${2:-swmbench/swmbench}"
PATH_IN_REPO="${3:-$(basename "$LOCAL")}"
[ -f "$LOCAL" ] || { echo "!! file not found: $LOCAL"; exit 1; }
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
TOKARG=(); [ -n "${HF_TOKEN:-}" ] && TOKARG=(--token "$HF_TOKEN")

echo "uploading $LOCAL ($(du -h "$LOCAL"|cut -f1))  ->  hf dataset $REPO : $PATH_IN_REPO"
huggingface-cli upload "$REPO" "$LOCAL" "$PATH_IN_REPO" --repo-type dataset "${TOKARG[@]}"
echo "done -> https://huggingface.co/datasets/$REPO/blob/main/$PATH_IN_REPO"
