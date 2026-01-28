#!/bin/bash
# Run inference with MultiEventForecaster
# Evaluates trained forecaster on test data and computes MAE/MSE

source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm

cd /home/haofeiy2/social-world-model/scripts

# Model paths from HuggingFace (skyyyyks/swm_models)
SWM_MODELS_DIR="/home/haofeiy2/swm_models"

# ============================================================
# Option A: Using precomputed attributions (test data already has attributions)
# Use this when test data has "attributed" suffix - no attributer model needed
# Using models from HuggingFace: skyyyyks/swm_models
# ============================================================

# Kalshi 0.6B with precomputed attributions
CUDA_VISIBLE_DEVICES=1 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_kalshi_0.6b_best \
    --model-name Qwen/Qwen3-0.6B \
    --output-path ../results/forecaster_kalshi_0.6b_precomputed.jsonl \
    --batch-size 4 &

# Kalshi 4B with precomputed attributions
CUDA_VISIBLE_DEVICES=3 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_kalshi_4b_best \
    --model-name Qwen/Qwen3-4B \
    --output-path ../results/forecaster_kalshi_4b_precomputed.jsonl \
    --batch-size 2 &

# Kalshi 8B with precomputed attributions
CUDA_VISIBLE_DEVICES=2 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_kalshi_8b_400 \
    --model-name Qwen/Qwen3-8B \
    --output-path ../results/forecaster_kalshi_8b_precomputed.jsonl \
    --batch-size 1 &

wait
echo "Kalshi precomputed evaluations complete!"

# ============================================================
# Option B: Full pipeline - Prior Attributer generates attributions, then Forecaster predicts
# Use this to test the complete inference pipeline
# Requires BOTH --model-path (forecaster) and --attributer-path (prior attributer)
# Note: No 0.6B prior_attributer available, using 4B for Kalshi
# ============================================================

# Kalshi 4B end-to-end
CUDA_VISIBLE_DEVICES=1 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_kalshi_4b_best \
    --attributer-path ${SWM_MODELS_DIR}/prior_attributer_kalshi_4b_1800 \
    --model-name Qwen/Qwen3-4B \
    --output-path ../results/forecaster_kalshi_4b_e2e.jsonl \
    --batch-size 2 &

# Kalshi 8B end-to-end
CUDA_VISIBLE_DEVICES=0 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_kalshi_8b_400 \
    --attributer-path ${SWM_MODELS_DIR}/prior_attributer_kalshi_8b_1400 \
    --model-name Qwen/Qwen3-8B \
    --output-path ../results/forecaster_kalshi_8b_e2e.jsonl \
    --batch-size 1 &

wait
echo "Kalshi end-to-end evaluations complete!"

# ============================================================
# Polymarket with precomputed attributions
# ============================================================

# Polymarket 0.6B precomputed
CUDA_VISIBLE_DEVICES=4 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_polymarket_0.6b_900 \
    --model-name Qwen/Qwen3-0.6B \
    --output-path ../results/forecaster_polymarket_0.6b_precomputed.jsonl \
    --batch-size 4 &

# Polymarket 4B precomputed
CUDA_VISIBLE_DEVICES=3 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_polymarket_4b_1400 \
    --model-name Qwen/Qwen3-4B \
    --output-path ../results/forecaster_polymarket_4b_precomputed.jsonl \
    --batch-size 2 &

# Polymarket 8B precomputed
CUDA_VISIBLE_DEVICES=2 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_polymarket_8b_1700 \
    --model-name Qwen/Qwen3-8B \
    --output-path ../results/forecaster_polymarket_8b_precomputed.jsonl \
    --batch-size 1 &

wait
echo "Polymarket precomputed evaluations complete!"

# ============================================================
# Polymarket end-to-end (Prior Attributer + Forecaster)
# Note: No 0.6B prior_attributer available
# ============================================================

# Polymarket 4B end-to-end (using prior_attributer_polymarket_4b_2900 - latest checkpoint)
CUDA_VISIBLE_DEVICES=3 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_polymarket_4b_1400 \
    --attributer-path ${SWM_MODELS_DIR}/prior_attributer_polymarket_4b_2900 \
    --model-name Qwen/Qwen3-4B \
    --output-path ../results/forecaster_polymarket_4b_e2e.jsonl \
    --batch-size 2 &

# Polymarket 8B end-to-end
CUDA_VISIBLE_DEVICES=2 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_test_2025-11-01.jsonl \
    --model-path ${SWM_MODELS_DIR}/multievent_forecaster_polymarket_8b_1700 \
    --attributer-path ${SWM_MODELS_DIR}/prior_attributer_polymarket_8b_1000 \
    --model-name Qwen/Qwen3-8B \
    --output-path ../results/forecaster_polymarket_8b_e2e.jsonl \
    --batch-size 1 &

wait
echo "All evaluations complete!"
