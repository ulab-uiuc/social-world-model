#!/bin/bash
# Upload a trained SWM world-model checkpoint to HuggingFace under the swmbench org.
# RUN ON A NETWORKED NODE (login node). Auth via HF_TOKEN env (write token for swmbench):
#   HF_TOKEN=hf_xxx bash scripts/upload_ckpt.sh  [CKPT_TAG] [HF_REPO]
# (or `huggingface-cli login` once instead of HF_TOKEN)
# Defaults: daily-7B (jin10d_bal) -> swmbench/swm-wm-jin10-daily-7b
set -euo pipefail
cd /storage/home/haofeiyu/social-world-model 2>/dev/null || cd /home/haofeiyu/social-world-model

TAG="${1:-jin10d_bal}"
REPO="${2:-swmbench/swm-wm-jin10-daily-7b}"
# huggingface_hub reads HF_TOKEN from the environment automatically.
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
TOKARG=(); [ -n "${HF_TOKEN:-}" ] && TOKARG=(--token "$HF_TOKEN")
CKPT="saves/world_model_qwen2p5_3b_${TAG}/best-model"
[ -d "$CKPT" ] || CKPT="saves/world_model_qwen2p5_3b_${TAG}/final-model"
[ -d "$CKPT" ] || { echo "!! checkpoint not found for TAG=$TAG"; exit 1; }
echo "uploading $CKPT  ->  hf: $REPO"
du -sh "$CKPT"

# drop a README so the repo documents base model + load recipe
cat > "$CKPT/README.md" <<'MD'
# SWM world model (MultiEventWorldModel)

Full fine-tuned **Qwen2.5-7B-Instruct** + regression head (`regression_head.bin`).
Predicts a prediction-market price **delta** from (question, daily price history, attributed news).

- base model: `Qwen/Qwen2.5-7B-Instruct` (NOT included — load separately)
- `predict_delta=True`, `pooling_method=last_token`, `max_news=8`, daily history
- load:
  ```python
  from swm.world_model import MultiEventWorldModel
  wm = MultiEventWorldModel(model_name="Qwen/Qwen2.5-7B-Instruct", max_news=8, predict_delta=True)
  wm.load("<this repo>")           # pred_price = before_price + pred_delta
  ```
Note: the local dir name contains "3b" as a leftover prefix; this checkpoint is 7B.
MD

# create repo (idempotent) + upload the whole folder to repo root
huggingface-cli repo create "$REPO" --type model -y "${TOKARG[@]}" 2>/dev/null || true
huggingface-cli upload "$REPO" "$CKPT" . --repo-type model "${TOKARG[@]}"
echo "done -> https://huggingface.co/$REPO"
