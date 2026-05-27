# E2E Pipeline Optimization (Attributer + Forecaster)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize end-to-end pipeline performance: PriorAttributer generates attributions on-the-fly, then MultiEventForecaster uses them to predict market prices. Both Kalshi and Polymarket datasets.

**Architecture:** Two-stage pipeline. Stage 1: PriorAttributer (news relevance scoring). Stage 2: MultiEventForecaster (price prediction using attributed news). Both are LoRA-finetuned Qwen3 models. We optimize the combined E2E performance (delta_mae, direction_accuracy) rather than attributer loss in isolation.

**Tech Stack:** PyTorch, HuggingFace Transformers, PEFT/LoRA, Qwen3 (0.6B/4B/8B)

---

## Current Baseline (E2E)

| Dataset    | Model | delta_mae | direction_acc | Attributer Used |
|------------|-------|-----------|---------------|-----------------|
| Kalshi     | 0.6B  | 0.0751    | 50.5%         | prior_attributer_kalshi_06b/cp-3350 |
| Kalshi     | 4B    | 0.0704    | 52.1%         | prior_attributer_kalshi_4b/cp-1800 (HF) |
| Kalshi     | 8B    | 0.0840    | 55.0%         | prior_attributer_kalshi_8b/cp-1400 (HF) |
| Polymarket | 4B    | 0.0397    | 33.3%         | prior_attributer_polymarket_4b/cp-2900 (HF) |
| Polymarket | 8B    | 0.0489    | 31.0%         | prior_attributer_polymarket_8b/cp-1000 (HF) |

Best attributer eval_losses:
- Kalshi 0.6B: 0.1611 (cp-2500), 4B: 0.1521 (cp-1000), 8B: 0.1550 (cp-2500)
- Polymarket 0.6B: 0.0914 (cp-1500), 4B: 0.0875 (cp-2000)

## Available Resources

- **GPUs free:** 2 (96GB free), 8 (~63GB free), 9 (~62GB free)
- **Local models:** All attributer + forecaster checkpoints for Kalshi/Polymarket 0.6B/4B/8B
- **HF models missing locally:** `swm_models/` directory doesn't exist; need to use local `saves/` checkpoints
- **Data:** `splitted_kalshi_v2_0102/` and `splitted_polymarket_v2_0102/` (attributed + non-attributed)

## Strategy

The pipeline bottleneck is the attributer — it determines which news the forecaster sees. Key insight: the **forecaster was trained with GPT-4o attributions** (precomputed), but at inference uses **PriorAttributer attributions** (on-the-fly). The distribution gap between these hurts E2E performance.

Three optimization axes:
1. **Retrain forecaster using PriorAttributer-generated attributions** (close the train/inference gap)
2. **Improve attributer training** (reduce epochs, better regularization to avoid overfitting)
3. **E2E evaluation sweep** to find best attributer checkpoint × forecaster checkpoint combo

---

## Chunk 1: Evaluate Current Models E2E

### Task 1: Run E2E inference for all existing model combinations

We need baselines on all dataset × model-size combos using locally available checkpoints.

**Files:**
- Read: `scripts/inference_multievent_forecaster.py`
- Read: `saves/prior_attributer_*/` (best checkpoints)
- Read: `saves/multievent_forecaster_*/` (best checkpoints)

- [ ] **Step 1: Run Kalshi 0.6B E2E with best attributer**

```bash
# GPU 2 - Kalshi 0.6B
screen -dmS kalshi_06b_e2e bash -c '
source ~/anaconda3/etc/profile.d/conda.sh && conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
CUDA_VISIBLE_DEVICES=2 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_kalshi_v2_0102/kalshi_data_processed_with_news_test_2025-11-01.jsonl \
    --model-path ../saves/multievent_forecaster_kalshi/checkpoint-8000 \
    --attributer-path ../saves/prior_attributer_kalshi_06b/checkpoint-2500 \
    --model-name Qwen/Qwen3-0.6B \
    --output-path ../results/kalshi_06b_e2e_best_attr.jsonl \
    --batch-size 4 2>&1 | tee ../logs/kalshi_06b_e2e.log
'
```

- [ ] **Step 2: Run Polymarket 0.6B E2E**

```bash
# GPU 8 - Polymarket 0.6B
screen -dmS poly_06b_e2e bash -c '
source ~/anaconda3/etc/profile.d/conda.sh && conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
CUDA_VISIBLE_DEVICES=8 python inference_multievent_forecaster.py \
    --test-data-path /mnt/data_from_server1/haofeiy2/social-world-model/data/splitted_polymarket_v2_0102/polymarket_data_processed_with_news_test_2025-11-01.jsonl \
    --model-path ../saves/multievent_forecaster_polymarket/checkpoint-1000 \
    --attributer-path ../saves/prior_attributer_polymarket_06b/checkpoint-1500 \
    --model-name Qwen/Qwen3-0.6B \
    --output-path ../results/poly_06b_e2e_best_attr.jsonl \
    --batch-size 4 2>&1 | tee ../logs/poly_06b_e2e.log
'
```

- [ ] **Step 3: Collect and compare metrics**

Check `results/*_e2e_best_attr.metrics.json` and compare against baselines.

---

## Chunk 2: Generate PriorAttributer Attributions for Training Data

Key optimization: retrain the forecaster on PriorAttributer-generated attributions instead of GPT attributions, so training and inference distributions match.

### Task 2: Create script to generate attributions on training data

**Files:**
- Create: `scripts/generate_prior_attributions.py`

- [ ] **Step 1: Write attribution generation script**

Script takes train data (non-attributed), runs PriorAttributer on it, and outputs attributed training data.

```python
"""Generate PriorAttributer attributions for training data."""
import argparse
from swm.attributer import BasicPriorAttributer
from swm.utils.utils import load_flat_samples_as_markets, set_seed
import jsonlines
# ... loads attributer, runs predict(), saves attributed data
```

- [ ] **Step 2: Generate Kalshi 0.6B attributions on train data**

```bash
CUDA_VISIBLE_DEVICES=2 python generate_prior_attributions.py \
    --data-path /mnt/.../kalshi_data_processed_with_news_train_2025-11-01.jsonl \
    --attributer-path ../saves/prior_attributer_kalshi_06b/checkpoint-2500 \
    --model-name Qwen/Qwen3-0.6B \
    --output-path /mnt/.../kalshi_prior_attributed_train_2025-11-01.jsonl
```

- [ ] **Step 3: Generate Polymarket 0.6B attributions on train data**

Same as above but for Polymarket.

---

## Chunk 3: Retrain Forecaster on PriorAttributer Attributions

### Task 3: Train forecaster using prior-generated attributions

The hypothesis: training the forecaster on the same attribution distribution it sees at inference should improve E2E performance.

**Files:**
- Use: `scripts/train_multievent_forecaster.py`

- [ ] **Step 1: Train Kalshi 0.6B forecaster on prior attributions**

```bash
screen -dmS kalshi_forecaster_prior bash -c '
source ~/anaconda3/etc/profile.d/conda.sh && conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
CUDA_VISIBLE_DEVICES=2 python train_multievent_forecaster.py \
    --train-data-path /mnt/.../kalshi_prior_attributed_train_2025-11-01.jsonl \
    --valid-data-path /mnt/.../kalshi_data_processed_with_news_attributed_test_2025-11-01.jsonl \
    --output-dir ../saves/multievent_forecaster_kalshi_prior \
    --model-name Qwen/Qwen3-0.6B \
    --train-batch-size 8 --gradient-accumulation-steps 2 --eval-batch-size 4 \
    --epochs 5 --logging-steps 10 --save-steps 50 --eval-steps 100 \
    --gradient-checkpointing --learning-rate 5e-5 --lora-r 16
'
```

- [ ] **Step 2: Train Polymarket 0.6B forecaster on prior attributions**

Same pattern for Polymarket data.

- [ ] **Step 3: Evaluate retrained forecasters E2E**

Run inference with the retrained forecaster + same attributer and compare metrics.

---

## Chunk 4: Attributer Checkpoint Sweep

### Task 4: Find optimal attributer checkpoint for E2E performance

The best attributer eval_loss may not correspond to best E2E performance. Test multiple attributer checkpoints with the same forecaster.

- [ ] **Step 1: List available attributer checkpoints**

For Kalshi 0.6B: checkpoints 500, 1000, 1500, 2000, 2500, 3000, ...

- [ ] **Step 2: Run E2E inference with each attributer checkpoint**

```bash
for ckpt in 500 1000 1500 2000 2500 3000 3500; do
    CUDA_VISIBLE_DEVICES=2 python inference_multievent_forecaster.py \
        --test-data-path .../kalshi_test.jsonl \
        --model-path ../saves/multievent_forecaster_kalshi/checkpoint-8000 \
        --attributer-path ../saves/prior_attributer_kalshi_06b/checkpoint-$ckpt \
        --model-name Qwen/Qwen3-0.6B \
        --output-path ../results/kalshi_06b_e2e_attr_cp${ckpt}.jsonl \
        --batch-size 4
done
```

- [ ] **Step 3: Compare E2E metrics across checkpoints**

Find which attributer checkpoint gives best delta_mae and direction_accuracy.

---

## Chunk 5: Improve Attributer Training

### Task 5: Train attributer with better hyperparameters

Based on our experiments: both linear and cosine schedules overfit after ~1.5 epochs with eval_loss ~0.30 on v2 data. The original model achieved 0.16 on v1 data. Key differences to investigate:
- Fewer epochs (2-3 instead of 10)
- Stronger regularization (higher dropout, lower LoRA rank)
- Early stopping patience

- [ ] **Step 1: Train with 3 epochs, higher dropout**

```bash
screen -dmS attr_improved bash -c '
source ~/anaconda3/etc/profile.d/conda.sh && conda activate social-wm
cd /home/haofeiy2/social-world-model/scripts
CUDA_VISIBLE_DEVICES=9 python train_attributer.py \
    --train-data-path .../attributed_train.jsonl \
    --valid-data-path .../attributed_test.jsonl \
    --output-dir ../saves/prior_attributer_kalshi_06b_improved \
    --model-name Qwen/Qwen3-0.6B \
    --eval-steps 100 --save-steps 100 \
    --train-batch-size 1 --gradient-accumulation-steps 8 --eval-batch-size 4 \
    --epochs 3 --learning-rate 3e-5 --lora-r 8 --max-news-per-bp 30 \
    --gradient-checkpointing --seed 42
'
```

- [ ] **Step 2: Evaluate improved attributer E2E**

---

## Execution Order

1. **Task 1** (Chunk 1): Run E2E baselines — ~30 min per run, parallel on GPUs 2/8/9
2. **Task 4** (Chunk 4): Attributer checkpoint sweep — can run parallel with Task 1
3. **Task 2** (Chunk 2): Generate prior attributions for train data — ~1 hour
4. **Task 3** (Chunk 3): Retrain forecaster — ~2-4 hours
5. **Task 5** (Chunk 5): Improved attributer training — ~2-3 hours, parallel with Task 3
6. Repeat E2E eval with best models from Tasks 3+5
