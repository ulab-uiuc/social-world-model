#!/usr/bin/env python3
"""Materialise streaming contexts, so SFT and the ICL baseline read the same text.

Emits train/valid/test splits (chronological 80/10/10 by `target.t`, matching
scripts/split_temporal.py) where every row carries the fully-assembled prompt,
the regression label, and a `_ctx` block recording what went into the context.

Building the prompts once and sharing the file is the point: an ICL baseline
that reads a different prompt from the one the fine-tune trained on measures the
prompt, not the training.

    python scripts/build_streaming_data.py --data swmbench_jin10_daily_en.jsonl \\
        --out-dir data/streaming --max-tokens 16384
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm.backtest.retrieval import EmbeddingRetriever
from swm.streaming import (
    StreamingContextBuilder,
    anchor_price,
    index_history,
    past_of,
    realised_change,
    record_time,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='the DAILY-history jsonl')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--max-news-per-block', type=int, default=6)
    p.add_argument('--max-current-news', type=int, default=60)
    p.add_argument('--max-blocks', type=int, default=40)
    p.add_argument('--no-siblings', action='store_true',
                   help='own-market history only; 31%% of records then have an '
                        'empty context, which is the ablation')
    p.add_argument('--train-frac', type=float, default=0.80)
    p.add_argument('--valid-frac', type=float, default=0.10)
    p.add_argument('--train-cutoff', default='2026-05-24',
                   help='assert the test block starts after this; "" to skip')
    p.add_argument('--no-rank-news', action='store_true',
                   help="keep each record's news in its original list order. "
                        'Off by default: that order is arbitrary, so filling a '
                        '30k window from the top of it packs the context with '
                        'headlines that have nothing to do with the market.')
    p.add_argument('--embed-model', default='BAAI/bge-small-en-v1.5')
    p.add_argument('--embed-device', default='cuda')
    return p.parse_args()


def rank_news_by_relevance(records, model_name, device):
    """Sort every record's news by cosine relevance to its own market.

    The builder fills a block from the top of `record['news']`, and that list
    arrives in wire order, so a market about shipping through the Strait of
    Hormuz was being handed A-share sector moves and Singapore gold clearing.
    Ordering by relevance keeps the context just as full while making
    truncation drop the least relevant headline rather than an arbitrary one.

    Ranking is per-record against that record's OWN market, so a sibling block
    shown inside another market's context stays ordered by what explains the
    sibling's move.
    """
    import numpy as np

    retriever = EmbeddingRetriever(model_name, device=device)
    corpus, index_of = [], {}
    for record in records:
        for item in record.get('news') or []:
            key = ((item.get('title') or '').strip(), item.get('published_at'))
            if key in index_of:
                continue
            title = (item.get('title') or '').strip()
            body = (item.get('description') or '').strip()
            index_of[key] = len(corpus)
            corpus.append(f'{title}\n{body}' if body and body != title else title)
    retriever.fit_news(corpus)

    market_ids, texts, seen = [], [], set()
    for record in records:
        mid = str(record['market_id'])
        if mid in seen:
            continue
        seen.add(mid)
        market_ids.append(mid)
        # question + tags, not question + description. The description is
        # resolution boilerplate that drags the embedding toward generic
        # finance; the tags are curated topic labels ("Hormuz", "Iran", "Oil").
        # Measured on 500 train records against the oracle attributions:
        # AUC 0.786 for question+tags, 0.764 question alone, 0.753 with the
        # description, 0.769 with both.
        question = record.get('question', '')
        tags = ' '.join(dict.fromkeys(record.get('tags') or []))
        texts.append(f'{question}\n{tags}'.strip())
    retriever.fit_markets(market_ids, texts)
    print(f'[rank] {len(corpus)} unique headlines, {len(market_ids)} markets')

    gains = []
    for record in records:
        news = record.get('news') or []
        if len(news) < 2:
            continue
        idxs, keep = [], []
        for i, item in enumerate(news):
            key = ((item.get('title') or '').strip(), item.get('published_at'))
            if key in index_of:
                idxs.append(index_of[key])
                keep.append(i)
        if not idxs:
            continue
        sims = retriever.similarities(str(record['market_id']), idxs)
        order = np.argsort(-sims)
        record['news'] = [news[keep[j]] for j in order]
        gains.append(float(sims[order[0]]) - float(np.mean(sims)))
    if gains:
        import statistics

        print(f'[rank] top-1 similarity beats the record mean by '
              f'{statistics.fmean(gains):.4f} on average')
    return records


def main():
    args = parse_args()
    records = [json.loads(line) for line in open(args.data) if line.strip()]
    records.sort(key=record_time)
    print(f'{len(records)} records')

    if not args.no_rank_news:
        records = rank_news_by_relevance(records, args.embed_model, args.embed_device)

    # Index over EVERY record: a test row may legitimately show train-period
    # history for its own market, which is what a live system would have.
    by_market, by_event = index_history(records)
    builder = StreamingContextBuilder(
        max_tokens=args.max_tokens,
        max_news_per_block=args.max_news_per_block,
        max_current_news=args.max_current_news,
        max_blocks=args.max_blocks,
        include_siblings=not args.no_siblings,
    )

    n = len(records)
    bounds = (int(n * args.train_frac), int(n * (args.train_frac + args.valid_frac)))
    splits = {
        'train': records[: bounds[0]],
        'valid': records[bounds[0] : bounds[1]],
        'test': records[bounds[1] :],
    }

    if args.train_cutoff:
        import datetime as dt

        cutoff = int(
            dt.datetime.strptime(args.train_cutoff, '%Y-%m-%d')
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
        first = record_time(splits['test'][0])
        if first < cutoff:
            raise SystemExit(
                f'test starts {dt.datetime.fromtimestamp(first, dt.timezone.utc):%Y-%m-%d}, '
                f'before the {args.train_cutoff} cutoff'
            )
        print(
            f'out-of-sample check: test starts '
            f'{dt.datetime.fromtimestamp(first, dt.timezone.utc):%Y-%m-%d}, '
            f'{(first - cutoff) / 86400:.0f} days after {args.train_cutoff}'
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        toks, own, sib, empty = [], [], [], 0
        path = out_dir / f'{name}.jsonl'
        with path.open('w') as fout:
            for record in rows:
                label = realised_change(record)
                before = anchor_price(record)
                if label is None or before is None:
                    continue
                own_hist, sib_hist = past_of(record, by_market, by_event)
                built = builder.build(record, own_hist, sib_hist)
                stats = built['stats']
                toks.append(stats['approx_tokens'])
                own.append(stats['own_blocks'])
                sib.append(stats['sibling_blocks'])
                empty += (stats['own_blocks'] + stats['sibling_blocks']) == 0
                fout.write(
                    json.dumps(
                        {
                            'market_id': str(record['market_id']),
                            'event_id': str(record.get('event_id')),
                            't': record_time(record),
                            'question': record.get('question', ''),
                            'prompt': built['prompt'],
                            'label_delta': label,
                            'before_price': before,
                            'target_price': record['target']['p'],
                            '_ctx': stats,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
        if not toks:
            continue
        toks_sorted = sorted(toks)
        print(
            f'  {name:5s} {len(toks):5d} rows -> {path}\n'
            f'        approx tokens: median {statistics.median(toks):.0f} '
            f'p90 {toks_sorted[int(0.9 * len(toks))]:.0f} max {max(toks)}\n'
            f'        blocks/row: own median {statistics.median(own):.0f} '
            f'sibling median {statistics.median(sib):.0f} | '
            f'empty context {empty} ({100 * empty / len(toks):.0f}%)'
        )

    sample = json.loads((out_dir / 'test.jsonl').read_text().split('\n')[0])
    print(f'\n--- sample prompt ({sample["_ctx"]["approx_tokens"]} approx tokens) ---')
    print(sample['prompt'][:1500])
    print('...')
    print(sample['prompt'][-400:])
    print(f'label_delta = {sample["label_delta"]:+.4f}')


if __name__ == '__main__':
    main()
