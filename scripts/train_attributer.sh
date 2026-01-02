#!/bin/bash
# Train PriorAttributer using KL divergence from precomputed posterior attributions

CUDA_VISIBLE_DEVICES=0 python train_attributer.py \
    --train-data-path ../data/attributed/polymarket_train.jsonl \
    --valid-data-path ../data/attributed/polymarket_dev.jsonl \
    --output-dir ../saves/prior_attributer \
    --cache-dir ../cache/prior_attributer \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --epochs 10 \
    --learning-rate 5e-5

# Sanity check
CUDA_VISIBLE_DEVICES=0 python train_attributer.py \
    --train-data-path ../data/attributed/polymarket_train.jsonl \
    --valid-data-path ../data/attributed/polymarket_dev.jsonl \
    --output-dir ../saves/prior_attributer_sanity \
    --cache-dir ../cache/prior_attributer_sanity \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --epochs 3 \
    --sanity-check

