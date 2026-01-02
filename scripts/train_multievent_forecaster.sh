#!/bin/bash
# Train MultiEventForecaster using precomputed attributions

CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path ../data/attributed/polymarket_train.jsonl \
    --valid-data-path ../data/attributed/polymarket_dev.jsonl \
    --output-dir ../saves/multievent_forecaster \
    --cache-dir ../cache/multievent_forecaster \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --epochs 10 \
    --learning-rate 5e-5

# Sanity check
CUDA_VISIBLE_DEVICES=0 python train_multievent_forecaster.py \
    --train-data-path ../data/attributed/polymarket_train.jsonl \
    --valid-data-path ../data/attributed/polymarket_dev.jsonl \
    --output-dir ../saves/multievent_forecaster_sanity \
    --cache-dir ../cache/multievent_forecaster_sanity \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --epochs 3 \
    --sanity-check

