#!/bin/bash
# Precompute attributions for market data using PosteriorAttributer

# Train data
CUDA_VISIBLE_DEVICES=0 python precompute_posterior_attributions.py \
    --input-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
    --output-data-path ../data/attributed/polymarket_train.jsonl \
    --corpus-news-path ../data/news/daily_news.jsonl \
    --attributer-model gpt-4o \
    --max-news-items 10 \
    --cache-dir ../cache/attributions

# Validation data
CUDA_VISIBLE_DEVICES=0 python precompute_posterior_attributions.py \
    --input-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
    --output-data-path ../data/attributed/polymarket_dev.jsonl \
    --corpus-news-path ../data/news/daily_news.jsonl \
    --attributer-model gpt-4o \
    --max-news-items 10 \
    --cache-dir ../cache/attributions

# Test data
CUDA_VISIBLE_DEVICES=0 python precompute_posterior_attributions.py \
    --input-data-path ../data/splitted_polymarket/polymarket_data_processed_test.jsonl \
    --output-data-path ../data/attributed/polymarket_test.jsonl \
    --corpus-news-path ../data/news/daily_news.jsonl \
    --attributer-model gpt-4o \
    --max-news-items 10 \
    --cache-dir ../cache/attributions

