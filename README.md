![Social World Model](assets/swm.png)

<h1 align="center">
  Building Social World Models with Large Language Models
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/ulab-uiuc/social-world-model/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://huggingface.co/datasets/ulab-ai/swm-bench"><img src="https://img.shields.io/badge/🤗%20Dataset-SWM--Bench-yellow" alt="HF Dataset"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" alt="PyTorch"></a>
</p>

A **Social World Model (SWM)** predicts how a market's collective belief (a
prediction-market price) shifts in response to news. Given a question, its price
history, and recent news, it forecasts the next price move — and, crucially,
*which* news drove it.

This repo provides a **training method** and a benchmark (**SWM-Bench**) for
that task. An SWM is built from two components that share the same forecasting
objective:

- a **social attributor**, which scores how responsible each news item is for a
  belief shift, and
- a **world model**, which predicts the price move from the attributed news and
  history.

We train both and run them jointly at inference. The attributor comes in two
forms: a large prompted LLM **posterior** attributor that scores news with
knowledge of the realized move (the training signal and an oracle ceiling), and
a small fine-tuned **prior** attributor that imitates it without seeing the
future (the deployable system).

## Installation

```bash
git clone https://github.com/ulab-uiuc/social-world-model.git
cd social-world-model
conda create -n social-world-model python=3.10 -y && conda activate social-world-model
pip install -e .

export HF_TOKEN="..."   # dataset download / model push
```

## Quick Start

Download **SWM-Bench** (price history + candidate news + posterior attributions,
chronologically split into train / Polymarket-test / Kalshi-test):

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ulab-ai/swm-bench', repo_type='dataset', local_dir='data/swm-bench')"
```

SWM-Bench has three parts:
- `raw/` — the original Polymarket / Kalshi price series + crawled news.
- `Qwen3.5-397B-attributed-data/` — the records labeled by the Qwen3.5-397B
  posterior attributor (our main dataset).
- `Qwen3-32B-attributed-data/` — the same records labeled by Qwen3-32B.

Each record is one `(history, candidate_news, target, attributions)` example.
`attributions` holds the posterior (oracle) scores; `*_with_nonzero_attribution.jsonl`
are the splits restricted to records with at least one non-zero-score news (used
for training).

## Stage 1 — Train the attributor

The attributor is trained to reproduce the posterior's responsibility
distribution over candidate news (forward-KL; one epoch is enough):

```bash
DATA=data/swm-bench/Qwen3.5-397B-attributed-data
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_attributer.py \
    --train-data-path $DATA/train_with_nonzero_attribution.jsonl \
    --valid-data-path $DATA/valid_subset150.jsonl \
    --output-dir saves/attributer_8b --model-name Qwen/Qwen3-8B \
    --epochs 1 --max-news 30 --max-seq-length 1024 \
    --train-batch-size 2 --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 --gradient-checkpointing
```

## Stage 2 — Train the world model

The world model is trained on the posterior-attributed data with a per-news,
responsibility-weighted regression loss, full fine-tuning under FSDP
(`MODE GPUS NPROC PORT MODEL TAG SAVE EP`):

```bash
# 8B, 8-GPU FSDP, 6 epochs
bash scripts/train_fc_v9odds.sh fsdp 0,1,2,3,4,5,6,7 8 29500 Qwen/Qwen3-8B wm8b saves_local 6

# 0.6B, single GPU
bash scripts/train_fc_v9odds.sh single 0 1 29501 Qwen/Qwen3-0.6B wm06b saves_local 6
```

## Stage 3 — Joint inference (prior & posterior)

**Posterior** (oracle attribution, already in the test file):

```bash
DATA=data/swm-bench/Qwen3.5-397B-attributed-data
python scripts/inference_multievent_world_model.py \
    --test-data-path $DATA/test_kalshi.jsonl \
    --model-path saves_local/wm8b/final-model --model-name Qwen/Qwen3-8B \
    --output-path results/posterior_kalshi.jsonl --max-news 30
```

**Prior** (deployable) — attribute with the trained attributor, then forecast:

```bash
# (a) prior attribution: replace each record's attributions with the model's
python scripts/inference_prior_attribution.py \
    --data-path $DATA/test_kalshi.jsonl \
    --attributer-path saves/attributer_8b --model-name Qwen/Qwen3-8B \
    --output-path results/test_kalshi_prior.jsonl --max-news 30

# (b) forecast on the prior-attributed file
python scripts/inference_multievent_world_model.py \
    --test-data-path results/test_kalshi_prior.jsonl \
    --model-path saves_local/wm8b/final-model --model-name Qwen/Qwen3-8B \
    --direct-soft-routing \
    --output-path results/prior_kalshi.jsonl --max-news 30
```

Both inference scripts shard across GPUs with `--num-shards N --shard-idx i`.
Each output row has `pred_delta` / `true_delta`; score MASE, MAE, directional
accuracy, and correlation over the full set and the attributed subset with
[`scripts/eval_all_vs_attr.py`](scripts/eval_all_vs_attr.py).

## License

[Apache 2.0](https://github.com/ulab-uiuc/social-world-model/blob/main/LICENSE)
