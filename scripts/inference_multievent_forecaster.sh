#!/bin/bash
# Run inference with MultiEventForecaster

# Option A: Using precomputed attributions
CUDA_VISIBLE_DEVICES=0 python inference_multievent_forecaster.py \
    --test-data-path ../data/attributed/polymarket_test.jsonl \
    --model-path ../saves/multievent_forecaster/checkpoint-best \
    --output-path ../results/multievent_forecaster_predictions.jsonl \
    --batch-size 8

# Option B: Using PriorAttributer for on-the-fly attribution
# CUDA_VISIBLE_DEVICES=0 python inference_multievent_forecaster.py \
#     --test-data-path ../data/splitted_polymarket/polymarket_data_processed_test.jsonl \
#     --model-path ../saves/multievent_forecaster/checkpoint-best \
#     --attributer-path ../saves/prior_attributer/checkpoint-best \
#     --output-path ../results/multievent_forecaster_predictions.jsonl \
#     --batch-size 8

