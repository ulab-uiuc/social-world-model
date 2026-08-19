#!/usr/bin/env python3
"""Build the (market x decision-time) grid the world model is asked to score.

Two universes, both restricted to the final 10% of `swmbench_jin10_dailyhist_en`
by `target.t` -- the slice the checkpoint never trained on:

  --mode breakpoint   one cell per test record: re-runs the jin10 test set, but
                      priced as a trade rather than as a regression metric.
  --mode grid         every market that is live at each decision time, whether
                      or not it moved. This is the honest universe: a strategy
                      has to decide *not* to trade the quiet markets too, and
                      only here can false positives cost anything.

Two news sources:

  --news oracle       the record's own headlines plus the dataset's
                      `attributions`. Those were scored with the realised move
                      in hand, so this is an upper bound, not a live system.
  --news retrieval    the global jin10 wire sliced to the decision window, with
                      relevance from a bi-encoder calibrated on TRAIN records.

Every cell carries a `_bt` block with the tradeable quotes: the entry price a
trader could actually see at decision time, the settlement quote, and the
holding period. Those are labels for the P&L only -- the prompt is built from
`history`, `question` and `news` alone.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from swm.backtest.newsstream import NewsStream
from swm.backtest.retrieval import EmbeddingRetriever, fit_calibration
from swm.backtest.universe import (
    DAY,
    MIN_HISTORY_POINTS,
    PriceSeries,
    load_records,
    record_time,
    temporal_split,
)

HOUR = 3600


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='swmbench_jin10_dailyhist_en.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['breakpoint', 'grid'], default='breakpoint')
    p.add_argument('--news', choices=['oracle', 'retrieval', 'none'], default='retrieval')
    p.add_argument('--max-news', type=int, default=8, help='must match the checkpoint')
    p.add_argument('--lookback-hours', type=float, default=2.5)
    p.add_argument('--lead-hours', type=float, default=1.5,
                   help='decision lands this long before the target timestamp')
    p.add_argument('--train-frac', type=float, default=0.80)
    p.add_argument('--valid-frac', type=float, default=0.10)
    p.add_argument('--train-cutoff', type=str, default='2026-05-24',
                   help='Date the checkpoint stopped training (UTC). The split '
                        'derived from --train-frac is asserted to start after '
                        'this; set to "" to skip the check.')
    p.add_argument('--embed-model', type=str, default='BAAI/bge-small-en-v1.5')
    p.add_argument('--embed-device', type=str, default='cuda')
    p.add_argument('--max-staleness-hours', type=float, default=72.0,
                   help='grid mode: how old the last quote may be for a market '
                        'to count as live')
    p.add_argument('--max-settle-gap-hours', type=float, default=30.0,
                   help='grid mode: how far ahead a settlement quote may sit')
    p.add_argument('--calib-records', type=int, default=400,
                   help='train records used to fit the similarity calibration')
    p.add_argument('--limit-times', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def oracle_news(record, max_news):
    """The record's own headlines, ranked by the dataset's posterior scores."""
    news = record.get('news') or []
    scored = [
        (a['news_idx'], float(a['score']))
        for a in record.get('attributions') or []
        if 0 <= a.get('news_idx', -1) < len(news) and float(a.get('score') or 0) > 0
    ]
    scored.sort(key=lambda x: -x[1])
    scored = scored[:max_news]
    return (
        [news[i] for i, _ in scored],
        [{'news_idx': j, 'score': s} for j, (_, s) in enumerate(scored)],
    )


def main():
    args = parse_args()
    lookback = int(args.lookback_hours * HOUR)
    lead = int(args.lead_hours * HOUR)

    records = load_records(args.data)
    train, valid, test = temporal_split(records, args.train_frac, args.valid_frac)
    print(f'records={len(records)} train={len(train)} valid={len(valid)} test={len(test)}')

    series = PriceSeries(records)
    stream = NewsStream(records)
    print(f'markets={len(series.market_ids)} news_stream={len(stream)}')

    # The chronological fraction is re-derived here, but the checkpoint was
    # trained off a *different* file (swmbench_jin10_attributed_filtered_en via
    # scripts/debias_split.py), so the fraction alone does not prove the window
    # is out of sample. Assert it against the cutoff the training scripts pin.
    if args.train_cutoff:
        cutoff = int(
            dt.datetime.strptime(args.train_cutoff, '%Y-%m-%d')
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
        first_test = min(record_time(r) for r in test)
        if first_test < cutoff:
            raise SystemExit(
                f'test window starts {dt.datetime.fromtimestamp(first_test, dt.timezone.utc):%Y-%m-%d}, '
                f'before the {args.train_cutoff} train cutoff -- would be in sample'
            )
        print(
            f'out-of-sample check: test starts '
            f'{dt.datetime.fromtimestamp(first_test, dt.timezone.utc):%Y-%m-%d}, '
            f'{(first_test - cutoff) / DAY:.0f} days after the {args.train_cutoff} '
            f'train cutoff'
        )

    decision_times = sorted({record_time(r) for r in test})
    if args.limit_times:
        decision_times = decision_times[: args.limit_times]
    lo, hi = decision_times[0], decision_times[-1]
    print(f'test window: {lo} .. {hi}  ({len(decision_times)} decision times)')

    retriever = calibration = None
    news_index_of = {}
    if args.news == 'retrieval':
        retriever = EmbeddingRetriever(args.embed_model, device=args.embed_device)
        calibration = fit_calibration(
            train, retriever, args.calib_records, args.seed
        )
        # Re-fit both corpora on everything the test window needs.
        uniq = {}
        for t in decision_times:
            for item in NewsStream.dedupe(stream.window(t - lookback, t - lead)):
                uniq[(item.title, item.t)] = item
        corpus = list(uniq.values())
        news_index_of = {(i.title, i.t): n for n, i in enumerate(corpus)}
        retriever.fit_news([i.text for i in corpus])
        market_ids = series.market_ids
        retriever.fit_markets(
            market_ids, [series.meta[m].text for m in market_ids]
        )
        print(f'[retrieval] news corpus={len(corpus)} markets={len(market_ids)}')

    def news_for(market_id, decision_t, target_t, record):
        if args.news == 'none':
            return [], []
        if args.news == 'oracle':
            if record is None:
                return [], []
            return oracle_news(record, args.max_news)
        items = NewsStream.dedupe(stream.window(target_t - lookback, target_t - lead))
        if not items:
            return [], []
        idxs = [news_index_of[(i.title, i.t)] for i in items]
        hits = retriever.top_k(
            market_id, idxs, k=args.max_news, calibration=calibration
        )
        back = {news_index_of[(i.title, i.t)]: i for i in items}
        picked = [back[h.news_index] for h in hits]
        return (
            [p.as_prompt_dict() for p in picked],
            [
                {'news_idx': j, 'score': h.score, 'similarity': h.similarity}
                for j, h in enumerate(hits)
            ],
        )

    settle_gap = int(args.max_settle_gap_hours * HOUR)
    stale = int(args.max_staleness_hours * HOUR)
    by_time_records = {}
    for r in test:
        by_time_records.setdefault(record_time(r), {})[str(r['market_id'])] = r

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cells = n_skipped = 0
    holds = []
    with out_path.open('w') as fout:
        for target_t in tqdm(decision_times, desc=f'building {args.mode}'):
            decision_t = target_t - lead
            if args.mode == 'breakpoint':
                market_ids = list(by_time_records.get(target_t, {}))
            else:
                market_ids = [
                    m
                    for m in series.market_ids
                    if series.is_live(m, decision_t, max_staleness=stale)
                ]
            for market_id in market_ids:
                record = by_time_records.get(target_t, {}).get(market_id)
                history = series.daily_history(market_id, target_t, as_of=decision_t)
                if len(history) < MIN_HISTORY_POINTS:
                    n_skipped += 1
                    continue
                entry_quote = series.quote(market_id, decision_t)
                if entry_quote is None:
                    n_skipped += 1
                    continue
                settle = series.forward_quote(market_id, target_t, max_gap=settle_gap)
                if settle is None:
                    n_skipped += 1
                    continue
                news, attributions = news_for(
                    market_id, decision_t, target_t, record
                )
                meta = series.meta[market_id]
                holds.append(settle[0] - entry_quote[0])
                cell = {
                    'market_id': market_id,
                    'event_id': meta.event_id,
                    'question': meta.question,
                    'description': meta.description,
                    'categories': meta.categories,
                    'history': history,
                    'news': news,
                    'attributions': attributions,
                    'target': {'t': target_t, 'p': settle[1]},
                    '_bt': {
                        'mode': args.mode,
                        'news_source': args.news,
                        'is_breakpoint': record is not None,
                        'decision_t': decision_t,
                        'entry_t': entry_quote[0],
                        'entry_price': entry_quote[1],
                        'entry_staleness': decision_t - entry_quote[0],
                        'settle_t': settle[0],
                        'settle_price': settle[1],
                        'hold_seconds': settle[0] - entry_quote[0],
                        'prev_price': history[-1]['p'],
                        'resolution': meta.outcome,
                        'n_news': len(news),
                    },
                }
                fout.write(json.dumps(cell, ensure_ascii=False) + '\n')
                n_cells += 1

    print(f'cells={n_cells} skipped={n_skipped} -> {out_path}')
    if holds:
        holds.sort()
        print(
            f'hold seconds: p10={holds[len(holds) // 10] / HOUR:.1f}h '
            f'median={holds[len(holds) // 2] / HOUR:.1f}h '
            f'p90={holds[9 * len(holds) // 10] / HOUR:.1f}h'
        )


if __name__ == '__main__':
    main()
