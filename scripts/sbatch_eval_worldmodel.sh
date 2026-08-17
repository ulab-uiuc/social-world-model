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

set -euo pipefail

REPO=/storage/home/haofeiyu/social-world-model
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/storage/home/haofeiyu/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/home/haofeiyu/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/haofeiyu/.conda/envs/swm-train

MODEL_BASE=/storage/home/haofeiyu/Agentgym-RL/AgentGym-RL/models/Qwen2.5-3B-Instruct
TEST="$REPO/swm-bench/Qwen3.5-397B-attributed-data/test_polymarket.jsonl"
RES_DIR="$REPO/results/eval_polymarket"
mkdir -p "$RES_DIR"

# use final-model (only correctly-gathered save)
CKPT_PM="$REPO/saves/world_model_qwen2p5_3b_pm/final-model"
CKPT_CLEAN="$REPO/saves/world_model_qwen2p5_3b_clean/final-model"

echo "PM    ckpt : $CKPT_PM"
echo "CLEAN ckpt : $CKPT_CLEAN"

for pair in "pm:$CKPT_PM" "clean:$CKPT_CLEAN"; do
    tag=${pair%%:*}
    ckpt=${pair#*:}
    [ ! -d "$ckpt" ] && { echo "SKIP $tag (no ckpt at $ckpt)"; continue; }
    out="$RES_DIR/${tag}.jsonl"
    echo ""
    echo "=== $tag ($ckpt) -> $out ==="
    python scripts/inference_multievent_world_model.py \
        --test-data-path "$TEST" \
        --model-path "$ckpt" \
        --model-name "$MODEL_BASE" \
        --output-path "$out" \
        --max-news 30 \
        --max-seq-length 1024 \
        --batch-size 4 \
        --null-rho0 1.0 --odds-eps 1e-3 --odds-temp 1.0 \
        2>&1 | tail -80
done

echo ""
echo "=== summary ==="
ls -lh "$RES_DIR"
