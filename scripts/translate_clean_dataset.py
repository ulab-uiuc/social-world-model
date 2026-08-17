#!/usr/bin/env python3
"""Translate the Chinese jin10 news in polymarket_cat5_clean.jsonl to English.

Runs a local Qwen2.5-7B-Instruct on a single GPU (transformers, batched).
Only the news `content` field is Chinese; we translate the unique contents
(dedup -> ~1.9k items) and write them back, producing an English clean set
that matches the English swm-bench distribution.

Output: data/polymarket_cat5_clean_en.jsonl  (same schema; `content` is now
English, original kept as `content_zh`). A content->translation cache is
persisted to data/.translation_cache.json so re-runs are ~free.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = Path("data/polymarket_cat5_clean.jsonl")
DST = Path("data/polymarket_cat5_clean_en.jsonl")
CACHE = Path("data/.translation_cache.json")

SYSTEM = (
    "You are a professional financial news translator. Translate the user's "
    "Chinese text into fluent, accurate English. Preserve numbers, names, "
    "dates, tickers and units exactly. Output ONLY the English translation "
    "with no preamble, quotes, or notes."
)


def collect_unique(src: Path):
    seen = {}
    for line in src.open():
        r = json.loads(line)
        for bp in r["breakpoints"]:
            for n in bp["news"]:
                c = n["content"]
                if c not in seen:
                    seen[c] = None
    return list(seen.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
        print(f"loaded {len(cache)} cached translations")

    contents = collect_unique(SRC)
    todo = [c for c in contents if c not in cache]
    print(f"{len(contents)} unique news; {len(todo)} to translate")

    if todo:
        tok = AutoTokenizer.from_pretrained(args.model_path)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()

        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": c},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for c in batch
            ]
            enc = tok(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=1024,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1] :]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for c, t in zip(batch, texts):
                cache[c] = t.strip()
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
            if (i // args.batch_size) % 10 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))

        CACHE.write_text(json.dumps(cache, ensure_ascii=False))

    # Write translated dataset.
    n = 0
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            for bp in r["breakpoints"]:
                for nw in bp["news"]:
                    zh = nw["content"]
                    nw["content_zh"] = zh
                    nw["content"] = cache.get(zh, zh)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} markets -> {DST}")


if __name__ == "__main__":
    main()
