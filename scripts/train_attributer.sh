#!/bin/bash
# v2 of KL fix retrain: combines
#   (a) dataset.py negative-news fix (every has-event group has pos+neg mix)
#   (b) NEWS-FIRST prompt reordering (news content now at start of prompt)
#   (c) --max-seq-length 1024 (was 512)
# Reason: v1 trained on 100% truncated inputs -- median prompt was 632 tokens,
# right-truncation chopped off the news tail AND the rating question, so the
# model never saw the discriminative signal. AUC stayed at 0.5 regardless of
# loss type (mse_rank vs KL) or training duration.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts

export HF_HUB_CACHE=/mnt/data_from_server1/haofeiy2/.cache/huggingface/hub
export HF_TOKEN=
unset HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CACHE=/mnt/data_from_server1/haofeiy2/social-world-model/cache
TRAIN=/home/haofeiy2/social-world-model/data/vllm_attributed/combined_train_vllm_attributed.jsonl
VALID=/home/haofeiy2/social-world-model/data/vllm_attributed/combined_valid_subset150.jsonl
LOG=/home/haofeiy2/social-world-model/logs
MNB=8
MSL=1024   # was 512 -- this is the critical fix

# 0.6B KL FIX v2 -> GPU 4
CUDA_VISIBLE_DEVICES=4 python train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir ../saves/prior_attributer_combined_06b_klfix_v2 \
    --model-name Qwen/Qwen3-0.6B --cache-dir $CACHE --loss-type kl \
    --train-batch-size 2 --gradient-accumulation-steps 8 --eval-batch-size 2 \
    --learning-rate 5e-5 --lora-r 16 --lora-alpha 32 \
    --epochs 3 --max-news-per-bp $MNB --max-seq-length $MSL \
    --eval-steps 200 --save-steps 200 --logging-steps 10 --gradient-checkpointing \
    > $LOG/attr_06b_klfix_v2.log 2>&1 &
echo "0.6B KL FIX v2 PID=$!  GPU=4  log=$LOG/attr_06b_klfix_v2.log"

# 4B KL FIX v2 -> GPU 5
CUDA_VISIBLE_DEVICES=5 python train_attributer.py \
    --train-data-path $TRAIN --valid-data-path $VALID \
    --output-dir ../saves/prior_attributer_combined_4b_klfix_v2 \
    --model-name Qwen/Qwen3-4B --cache-dir $CACHE --loss-type kl \
    --train-batch-size 1 --gradient-accumulation-steps 16 --eval-batch-size 1 \
    --learning-rate 2e-5 --lora-r 32 --lora-alpha 64 \
    --epochs 3 --max-news-per-bp $MNB --max-seq-length $MSL \
    --eval-steps 250 --save-steps 250 --logging-steps 10 --gradient-checkpointing \
    > $LOG/attr_4b_klfix_v2.log 2>&1 &
echo "4B KL FIX v2 PID=$!  GPU=5  log=$LOG/attr_4b_klfix_v2.log"
