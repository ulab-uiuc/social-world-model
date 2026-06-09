# Main WorldModel Pipeline

End-to-end production pipeline for the prediction-market world_model.
Side experiments + dead ends live in `backup_plan/` (gitignored).

## Current SOTA (full K + P test, ORACLE routing)

| dataset | model | dirAcc | MAE |
|---|---|---|---|
| **Kalshi** (1120 events) | SMART joint v44 + copy | **57.4%** | 0.0558 |
| **Polymarket** (3692 events) | SMART joint v42b + copy | **70.0%** | 0.0339 |

Beats copy-baseline (50%) by +7-20pp.

> ⚠️ The numbers above use **oracle attribution at routing** (Qwen3-8B sees
> after_price when scoring news). For a real production deployment you need
> the *prior attributer* (which does NOT see after_price). See pipeline below.

## SMART joint architecture

```
For each event:
    score = prior_attributer.score(market, news, before_price)   # prior (production)
                                                                 # OR oracle (eval)
    if any(score > 0):
        pred = v42b_or_v44.predict(event)
    else:
        pred = p_before
```

null events have true Δp ≈ 0 with no learnable signal — copy is optimal. Any LM
trained on null events makes things worse (we tried v43/v43b/v46; all lost to copy).

## Three-stage production pipeline

### Stage 1 — Posterior attribution (offline labelling)

"Posterior" because the scorer SEES after_price → uses outcome to judge causality.
Generates ground-truth attribution labels on training/test data.

- `inference_posterior_attribution.py` — VLLM-hosted Qwen3.5-9B (or Qwen3-8B), per-news
  /no_think prompt with explicit anti-hallucination guardrails. Empirically
  more accurate than the strictC variant (forced /think + BECAUSE→THEREFORE),
  which over-promotes tangentially related news.

### Stage 2 — Prior attribution (production scorer)

"Prior" because the scorer does NOT see after_price → usable at production
inference time. Trains a Qwen3-0.6B to predict the Stage-1 (posterior) scores
from (market, news, before_price) only.

- `train_attributer.py` — trainer (uses `swm/attributer.py`'s `MSERankTrainer`)
- `train_attributer.sh` — best launch config (klfix_v2: 0.6B LoRA + KL fix)
- `inference_prior_attribution.py` — apply trained prior attributer to new data

Existing ckpts: `/mnt/data_from_server1/haofeiy2/social-world-model/saves/prior_attributer_combined_06b_klfix_v2/`

### Stage 3 — WorldModel (has-news predictor)

MSE scalar regressor that predicts `p_after` directly from
(market, news, history, before_price).

- `train_multievent_world_model.py` — trainer
- `inference_multievent_world_model.py` — world_model-only inference
  (expects attribution already in test jsonl — used for our SMART joint eval)
- `train_fc_v42b_predict_p_newsonly.sh` — v42b (Polymarket SOTA backbone)
- `train_fc_v44_predict_p_stdfilter.sh` — v44 (Kalshi SOTA backbone)

### End-to-end inference (attributer + world_model in one pass)

For **production** — runs prior attributer FIRST to score news, then
world_model on top-K. No oracle attribution needed in test data.

- `inference_vllm.py` — **fastest** (~450 items/s). Uses vLLM with two LoRA
  adapters (attributer + world_model) sharing the same Qwen3-8B base.
  Recommended for production / large-scale eval.
- `inference_fast.py` — HF-based (~22 items/s, ~20x slower). No vLLM
  dependency. Useful for debugging or environments without vLLM.

### Analysis

- `analyze_smart_joint.py` — final eval (SMART joints on K+P)

## Data (in `data/vllm_attributed/`)

- `combined_train_vllm_attributed.jsonl` — train (12,582 events, ORIG oracle)
- `combined_valid_subset150.jsonl` — valid (150 events, small)
- `combined_test_kalshi.jsonl` — Kalshi test (1120)
- `combined_test_polymarket.jsonl` — Polymarket test (3692)
- `*_strictC.jsonl` — strictC re-rated variants (NOT recommended — see Stage 1)

## End-to-end reproduce

```bash
# Stage 1: oracle attribution (run once on each new dataset)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --port 8234 &
python scripts/inference_posterior_attribution.py \
    --input_file data/raw/combined_test_kalshi.jsonl \
    --output_file data/vllm_attributed/combined_test_kalshi.jsonl \
    --vllm_url http://localhost:8234/v1 --concurrency 64
# (repeat for train, valid, test_polymarket)

# Stage 2: train prior attributer (for production inference)
bash scripts/train_attributer.sh 0

# Stage 3: train world_models (~5-6h each on H100)
bash scripts/train_fc_v42b_predict_p_newsonly.sh 1   # P SOTA
bash scripts/train_fc_v44_predict_p_stdfilter.sh 2   # K SOTA

# Inference (oracle routing — what we report)
python scripts/inference_multievent_world_model.py \
    --test-data-path data/vllm_attributed/combined_test_kalshi.jsonl \
    --model-path /path/to/world_model_v44_predict_p_stdfilter/checkpoint-600 \
    --predict-absolute-price \
    --output-path results/v44_ck600_K.jsonl
# (repeat for v42b on P)

# Final eval
python scripts/analyze_smart_joint.py

# OPTIONAL — full production simulation (prior attributer routing)
# 1. Run prior attributer on test data to generate "production" attribution
python scripts/inference_prior_attribution.py \
    --data-path data/vllm_attributed/combined_test_kalshi.jsonl \
    --attributer-path /path/to/prior_attributer_06b_klfix_v2/checkpoint \
    --output-path data/vllm_attributed/combined_test_kalshi_PRIOR.jsonl
# 2. Then world_model inference + SMART joint on PRIOR-attributed test
```
