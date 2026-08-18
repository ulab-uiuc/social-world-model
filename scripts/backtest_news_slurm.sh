#!/bin/bash
#SBATCH --job-name=bt-news
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-bt-news-%j.out
#SBATCH --error=logs/slurm-bt-news-%j.err
#
# News-driven backtest, end-to-end on one A100:
#   build (reverse retrieval) -> predict (7B world model) -> report (P&L)
# Offline: pypi is reachable but huggingface is not, so every model loads from a
# local path and HF is forced offline.

set -euo pipefail
REPO=/storage/home/haofeiyu/social-world-model
cd "$REPO"

ENV=/storage/home/haofeiyu/.conda/envs/swm-train
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/storage/home/haofeiyu/.cache/hf
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" logs results/backtest

DATA=data/swmbench_jin10_dailyhist_en.jsonl
CKPT="$REPO/saves/world_model_qwen2p5_3b_jin10d_bal/best-model"   # dir says 3b, weights are 7B
TOKENIZER=/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-7B-Instruct
EMBED=/storage/home/haofeiyu/Agentgym-RL/models/bge-small-en-v1.5

GRID=results/backtest/grid_news_retrieval.jsonl
PREDS=results/backtest/preds_news_retrieval.jsonl
REPORT=results/backtest/report_news.json

echo "== [1/3] build news-driven cells =="
python scripts/backtest_build_news_grid.py \
    --data "$DATA" --out "$GRID" \
    --embed-model "$EMBED" --embed-device cuda \
    --top-markets 5 --hold-hours 24 --max-news 8

echo "== [2/3] score with the 7B world model =="
python scripts/backtest_predict.py \
    --grid "$GRID" --model-path "$CKPT" --model-name "$TOKENIZER" \
    --out "$PREDS" --batch-size 8 --max-news 8 --dtype bfloat16

echo "== [3/3] P&L / baselines / equity curve =="
python scripts/backtest_report.py --preds news="$PREDS" --out "$REPORT"

echo "== DONE -> $REPORT =="
