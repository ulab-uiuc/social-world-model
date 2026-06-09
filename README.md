![Social World Model](assets/swm.png)

<h1 align="center">
  Building Social World Models with Large Language Models
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/ulab-uiuc/social-world-model/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://huggingface.co/datasets/lwaekfjlk/social-world-model-v6-qwen3.5-397B-clean-semdedup"><img src="https://img.shields.io/badge/🤗%20Dataset-SWM--Bench-yellow" alt="HF Dataset"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" alt="PyTorch"></a>
</p>

**Social World Model (SWM)** forecasts how a prediction market's belief (price)
moves in response to news. It has two trained components:

1. **Social attributor** — given a market question, its price history, and a set
   of candidate news, scores *which* news drives the belief shift. The
   **posterior** attributor is a large prompted LLM (Qwen3.5-397B) that scores
   news with knowledge of the realized move (the oracle/training signal); the
   **prior** attributor is a small LLM fine-tuned to *imitate* that signal at
   inference (no peek at the future).
2. **WorldModel** — given the attributed news + history, predicts the price
   change. Per-news predictions are aggregated into the final forecast through
   the attributor's weights.

At inference we run the two **jointly**: the attributor selects/weights news,
the world model turns that into a price move. We report two settings — **prior**
(deployable, prior attributor) and **posterior** (oracle-attribution ceiling).

> **Recipes in one line.** The **attributor** is trained with a standard
> **forward-KL** objective for **1 epoch**; the **world model** is trained with a
> standard **weighted MSE**. Both operate over the same **odds distribution**:
> per-news Bernoulli responsibilities are mapped to a categorical over
> *(news ∪ no-news)* via odds with a null prior mass (see
> [`swm/dataset.py`](swm/dataset.py)), so weak/irrelevant events shrink toward
> "no change" instead of being force-normalized to a confident pick.

## Installation

```bash
git clone https://github.com/ulab-uiuc/social-world-model.git
cd social-world-model
conda create -n social-world-model python=3.10 -y && conda activate social-world-model
pip install -e .

export HF_TOKEN="..."   # dataset download / model push
```

## Quick Start

**Download the benchmark** (history + candidate news + posterior attributions,
chronologically split; test sets for Polymarket and Kalshi):

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('lwaekfjlk/social-world-model-v6-qwen3.5-397B-clean-semdedup', \
  repo_type='dataset', local_dir='data/social-world-model-v6-qwen3.5-397B-clean-semdedup')"
```

Each record is one `(history, candidate_news, target, attributions)` group.
The `attributions` field holds the **posterior** (oracle) scores; the
*attributed subset* (`attr`) is the records with ≥1 non-zero-score news.

## Stage 1 — Train the attributor (forward-KL, 1 epoch, odds)

The attributor learns to reproduce the posterior odds distribution over
candidate news with a forward-KL loss. One epoch is enough.

```bash
DATA=data/social-world-model-v6-qwen3.5-397B-clean-semdedup
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_attributer.py \
    --train-data-path $DATA/train_clean.jsonl \
    --valid-data-path $DATA/valid_subset150.jsonl \
    --output-dir saves/prior_attributer_8b \
    --model-name Qwen/Qwen3-8B \
    --target-mode odds --null-odds 1.0 --odds-eps 1e-3 --odds-temp 1.0 \
    --epochs 1 --max-news 30 --max-seq-length 1024 \
    --train-batch-size 2 --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 --gradient-checkpointing
```

`--target-mode odds` sets the odds-distribution target; forward-KL is the
default (pass `--reverse-kl` to switch). See [`scripts/train_attributer.sh`](scripts/train_attributer.sh)
for the 0.6B / 4B / 8B sweep.

## Stage 2 — Train the world model (weighted MSE, odds)

The world model is trained on the posterior-attributed data with a per-news,
responsibility-weighted MSE under the same odds routing. Full fine-tuning under
FSDP (`MODE GPUS NPROC PORT MODEL TAG SAVE EP`):

```bash
# 8B, 8-GPU FSDP, 6 epochs
bash scripts/train_fc_v9odds.sh fsdp 0,1,2,3,4,5,6,7 8 29500 Qwen/Qwen3-8B fc8b saves_local 6

# 0.6B, single GPU
bash scripts/train_fc_v9odds.sh single 0 1 29501 Qwen/Qwen3-0.6B fc06b saves_local 6
```

This wraps [`scripts/train_multievent_world_model.py`](scripts/train_multievent_world_model.py)
with `--per-news-loss --odds-null-categorical` (the deployed recipe).

## Stage 3 — Joint inference (prior & posterior)

**Posterior** (oracle attribution, already in the test file) → odds routing:

```bash
DATA=data/social-world-model-v6-qwen3.5-397B-clean-semdedup
python scripts/inference_multievent_world_model.py \
    --test-data-path $DATA/test_kalshi_final.jsonl \
    --model-path saves_local/fc8b/final-model --model-name Qwen/Qwen3-8B \
    --output-path results/posterior_kalshi.jsonl --max-news 30
```

**Prior** (deployable) — first attribute with the trained attributor, then
forecast with **direct soft routing** (use the prior weights as-is, no oracle):

```bash
# (a) prior attribution: replaces each record's `attributions` with the model's
python scripts/inference_prior_attribution.py \
    --data-path $DATA/test_kalshi_final.jsonl \
    --attributer-path saves/prior_attributer_8b --model-name Qwen/Qwen3-8B \
    --output-path results/test_kalshi_prior.jsonl --max-news 30

# (b) forecast on the prior-attributed file
python scripts/inference_multievent_world_model.py \
    --test-data-path results/test_kalshi_prior.jsonl \
    --model-path saves_local/fc8b/final-model --model-name Qwen/Qwen3-8B \
    --direct-soft-routing \
    --output-path results/prior_kalshi.jsonl --max-news 30
```

Both inference scripts shard across GPUs with `--num-shards N --shard-idx i`
(launch one process per GPU, then concatenate the shards).

Each output row carries `pred_delta` / `true_delta`; aggregate metrics
(MASE, MAE, 3-way directional accuracy, Pearson correlation) over the full set
and the attributed subset with [`scripts/eval_all_vs_attr.py`](scripts/eval_all_vs_attr.py).

> **Building the data yourself (optional).** The posterior attributions ship
> with the dataset. To regenerate them, run the prompted posterior attributor
> ([`scripts/inference_posterior_attribution.py`](scripts/inference_posterior_attribution.py),
> Qwen3.5-397B via vLLM) over raw market+news records, then
> [`scripts/semantic_dedup.py`](scripts/semantic_dedup.py) to dedupe news.

## License

[Apache 2.0](https://github.com/ulab-uiuc/social-world-model/blob/main/LICENSE)
