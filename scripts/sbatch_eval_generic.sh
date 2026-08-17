#!/bin/bash
#SBATCH --job-name=swm-eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-eval-%j.out
#SBATCH --error=logs/slurm-eval-%j.err

# Generic parameterized eval. Required env: TAG
set -euo pipefail
REPO=/storage/home/haofeiyu/social-world-model
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export HF_HOME=/storage/home/haofeiyu/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/home/haofeiyu/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/haofeiyu/.conda/envs/swm-train

: "${TAG:?}"
MODEL_BASE="${MODEL_PATH:-/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
TEST="${TEST_FILE:-$REPO/swm-bench/Qwen3.5-397B-attributed-data/test_polymarket.jsonl}"
# CKPT_TAG = which model to load (defaults to TAG); OUT_TAG = result filename
# (defaults to TAG). Decoupled so one model can be evaluated on several test
# sets into distinct result files (the 2x2 cross-domain matrix).
CKPT_TAG="${CKPT_TAG:-$TAG}"; OUT_TAG="${OUT_TAG:-$TAG}"
# Prefer the best-eval checkpoint; fall back to final-model.
CKPT="$REPO/saves/world_model_qwen2p5_3b_${CKPT_TAG}/best-model"
[ -d "$CKPT" ] || CKPT="$REPO/saves/world_model_qwen2p5_3b_${CKPT_TAG}/final-model"
OUT="$REPO/results/eval_polymarket/${OUT_TAG}.jsonl"
mkdir -p "$(dirname "$OUT")"
echo "[eval ${OUT_TAG}] ckpt_tag=$CKPT_TAG test=$TEST ckpt=$CKPT"

python scripts/inference_multievent_world_model.py \
    --test-data-path "$TEST" --model-path "$CKPT" --model-name "$MODEL_BASE" \
    --output-path "$OUT" --max-news "${MAXNEWS:-30}" --max-seq-length 1024 --batch-size "${EBATCH:-4}" \
    --null-rho0 1.0 --odds-eps 1e-3 --odds-temp 1.0
