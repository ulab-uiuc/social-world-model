"""Upload active data files to HuggingFace Hub.

Repo: ulab-uiuc/social-world-model (PRIVATE dataset)

Requires HF_TOKEN env or `hf auth login` first:
    hf auth login   # paste token from https://huggingface.co/settings/tokens

Usage:
    python upload_data_to_hf.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "lwaekfjlk/social-world-model"
REPO_TYPE = "dataset"
PRIVATE = True

DATA_DIR = Path("/home/haofeiy2/social-world-model/data/vllm_attributed")

# Active files only (~2.2 GB total)
FILES = [
    "combined_train_vllm_attributed.jsonl",          # 306 MB  ORIG train
    "combined_train_vllm_attributed_strictC.jsonl",  # 1.3 GB  strictC train
    "combined_valid_subset150.jsonl",                # 5.4 MB  valid
    "combined_test_kalshi.jsonl",                    # 37 MB   ORIG K test
    "combined_test_kalshi_strictC.jsonl",            # 80 MB   strictC K test
    "combined_test_polymarket.jsonl",                # 134 MB  ORIG P test
    "combined_test_polymarket_strictC.jsonl",        # 308 MB  strictC P test
]

README = """---
license: cc-by-nc-4.0
language:
- en
tags:
- prediction-markets
- time-series
- forecasting
- causal-attribution
size_categories:
- 10K<n<100K
---

# Social World Model — Prediction Market Forecasting Dataset

Posterior-attributed news/price data from Kalshi and Polymarket prediction
markets, used to train and evaluate the SMART joint forecaster.

## Files

| File | Rows | Description |
|---|---|---|
| `combined_train_vllm_attributed.jsonl` | 12,582 | Train set. ORIG oracle attribution from Qwen3-8B (per-news /no_think scoring). |
| `combined_train_vllm_attributed_strictC.jsonl` | 12,582 | Same train events, but news re-scored by Qwen3-8B /think + forced BECAUSE→THEREFORE causal chain. Experimentally LESS accurate than ORIG. |
| `combined_valid_subset150.jsonl` | 150 | Held-out valid for ckpt selection. |
| `combined_test_kalshi.jsonl` | 1,120 | Kalshi test. ORIG attribution. |
| `combined_test_kalshi_strictC.jsonl` | 1,120 | Kalshi test, strictC attribution. |
| `combined_test_polymarket.jsonl` | 3,692 | Polymarket test. ORIG attribution. |
| `combined_test_polymarket_strictC.jsonl` | 3,692 | Polymarket test, strictC attribution. |

## Per-row schema

```json
{
  "market_id": "KXHONDURASPRES-25-RMON",
  "event_id": "KXHONDURASPRES-25",
  "question": "Who will be elected President of Honduras in 2025?",
  "categories": ["Politics", "Election"],
  "sample_type": "breakpoint",       // or "normal_point"
  "before": {"t": <unix>, "p": 0.32},
  "after":  {"t": <unix>, "p": 0.73},
  "change": 0.41,
  "z_score": 26.98,
  "window_start": <unix>,
  "window_end": <unix>,
  "window_history": [{"t": <unix>, "p": 0.47}, ...],   // 17 days: 15 history + before + after
  "news": [{"title": "...", "description": "...", "published_at": "..."}, ...],
  "attributions": [{"news_idx": 0, "score": 0.85}, ...]   // ORIG: just news_idx + score
  // strictC files add: llm_orig_score, llm_strength, llm_verdict, llm_justification, llm_full_response
}
```

## Attribution generation

ORIG attribution = Qwen3-8B/3.5-9B (VLLM) prompted with market + news + before/after prices
in `/no_think` mode, scoring causality 0-100 (then divided by 100 for `score`).

strictC = same model but `/think` mode + max_tokens=4096, forced to construct
BECAUSE→THEREFORE→STRENGTH causal chain. Empirically over-promotes
tangentially-related news (false positives), so ORIG is recommended.

## Breakpoint definition

`sample_type=breakpoint` if the day's |price change| exceeds the rolling
robust z-score threshold (z > 2.0, computed with 30-day MAD-based detector).
`normal_point` events are non-anomalous days sampled for training balance.

## Splits

train ~71.7% / valid ~0.9% / test 28% (Kalshi 6.4% + Polymarket 21%).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ulab-uiuc/social-world-model", data_files="combined_test_kalshi.jsonl")
```
"""


def main():
    api = HfApi()
    print(f"Creating/ensuring repo: {REPO_ID} (private={PRIVATE})")
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)

    # README
    print("uploading README.md")
    api.upload_file(
        path_or_fileobj=README.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    # Data files
    for fname in FILES:
        local = DATA_DIR / fname
        if not local.exists():
            print(f"  SKIP (missing): {fname}")
            continue
        size_mb = local.stat().st_size / 1024 / 1024
        print(f"  uploading {fname} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"    done")

    print(f"\nAll uploaded → https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
