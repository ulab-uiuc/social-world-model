#!/bin/bash
# v44: v42b config + window_std filter + strictC attribution.
#
# Hypothesis: shrinkage in v42b comes from noisy attribution labels on
# illiquid markets. Filter out events where pre-BP price was too stable
# (window std < 0.02 = illiquid) AND use LLM-cleaned strictC scores.
#
# Same as v42b: predict p_after, news-only, MSE regressor.
GPU=${1:-5}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=$GPU python -u train_multievent_forecaster.py \
    --train-data-path /home/haofeiy2/social-world-model/data/vllm_attributed/combined_train_vllm_attributed_strictC.jsonl \
    --valid-data-path /home/haofeiy2/social-world-model/data/vllm_attributed/combined_valid_subset150.jsonl \
    --output-dir /mnt/data_from_server1/haofeiy2/swm_saves/forecaster_v44_predict_p_stdfilter \
    --model-name Qwen/Qwen3-0.6B \
    --cache-dir /mnt/data_from_server1/haofeiy2/social-world-model/cache \
    --train-batch-size 2 --gradient-accumulation-steps 8 --eval-batch-size 2 \
    --learning-rate 5e-5 --lora-r 16 --lora-alpha 32 --head-lr-multiplier 5 \
    --epochs 3 --max-news-per-bp 30 --max-seq-length 1024 \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    --predict-absolute-price \
    --null-subsample-ratio 0.0 \
    --window-std-threshold 0.02 \
    > /home/haofeiy2/social-world-model/logs/fc_v44_predict_p_stdfilter.log 2>&1
echo "v44 done"
