#!/usr/bin/env python3
"""Build a *news-driven* backtest universe: walk the jin10 wire in time order.

The grid builder (scripts/backtest_build_grid.py) iterates `decision_time x live
market` and retrieves news *for each market*. This one inverts the axis, which
is what a headline-reacting trader actually does:

    for each important headline, in publication order:
        retrieve the top-K markets the headline is about   (reverse retrieval)
        for each, open a position now and hold ~24h

The two retrieval directions share one bi-encoder and one TRAIN-fit calibration
(swm.backtest.retrieval): reverse retrieval (`top_markets`) picks *which* markets
to trade; forward retrieval (`top_k`, via `news_for`) fills each cell's prompt
with that market's recent news so the input matches the checkpoint's training
distribution rather than a single-headline out-of-distribution prompt.

Every cell carries the exact same schema and `_bt` block as the grid builder, so
scripts/backtest_predict.py and scripts/backtest_report.py consume it unchanged.
`_bt.is_breakpoint` is always False here; the P&L labels (`entry_price`,
`settle_price`) never reach a prompt.

No-lookahead guarantees, all anchored at the headline's publication time
`decision_t`:
  * prompt news  -> stream.window(decision_t - lookback, decision_t + 1)  (<= t)
  * entry quote  -> series.quote(market, decision_t)                      (<= t)
  * history      -> daily_history(target_t, as_of=decision_t), ends at t  (<= t)
  * settle quote -> series.forward_quote(market, target_t)  (>= t + hold), P&L only
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


class HourlyPrices:
    """Real hourly polymarket price book, loaded from *_series.jsonl.

    Each line is {market_id, ..., series:[{t,p}, ...]} sampled ~hourly. Used for
    the *tradeable* quotes (entry / settle) so a news trade fills at the next
    hourly tick after the headline instead of snapping to the daily reconstructed
    series. The model prompt still uses the daily history it was trained on.
    """

    def __init__(self, path, keep=None):
        import bisect
        self._bisect = bisect
        self._ts, self._ps = {}, {}
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            mid = str(r.get('market_id'))
            if keep is not None and mid not in keep:
                continue
            s = r.get('series') or []
            if not s:
                continue
            s.sort(key=lambda x: x['t'])
            self._ts[mid] = [int(x['t']) for x in s]
            self._ps[mid] = [float(x['p']) for x in s]

    def __contains__(self, mid):
        return mid in self._ts

    def forward_quote(self, mid, t, max_gap=None):
        """First tick at or after t -- the price a trader gets by acting at t."""
        ts = self._ts.get(mid)
        if not ts:
            return None
        i = self._bisect.bisect_left(ts, t)
        if i >= len(ts):
            return None
        if max_gap is not None and ts[i] - t > max_gap:
            return None
        return ts[i], self._ps[mid][i]

    def has_near(self, mid, t, max_staleness):
        """A tick within max_staleness before t -- market is live/quoted."""
        ts = self._ts.get(mid)
        if not ts:
            return False
        i = self._bisect.bisect_right(ts, t) - 1
        return i >= 0 and 0 <= t - ts[i] <= max_staleness


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='swmbench_jin10_dailyhist_en.jsonl')
    p.add_argument('--price-series', type=str, default=None,
                   help='hourly price book (e.g. data/polymarket_2026_series.jsonl). '
                        'When given, entry/settle fill at the next hourly tick after '
                        'the headline; the daily history in the prompt is unchanged.')
    p.add_argument('--out', required=True)
    p.add_argument('--news', choices=['retrieval'], default='retrieval',
                   help='only bi-encoder retrieval is meaningful here')
    p.add_argument('--max-news', type=int, default=8, help='must match the checkpoint')
    p.add_argument('--top-markets', type=int, default=5,
                   help='K markets traded per headline (before position dedup)')
    p.add_argument('--news-lookback-hours', type=float, default=2.5,
                   help='prompt news window, ending at the headline time')
    p.add_argument('--hold-hours', type=float, default=24.0,
                   help='settle at the first quote this long after entry')
    p.add_argument('--entry-timing', choices=['pre', 'post'], default='post',
                   help="'post' (realistic): fill at the first quote AT/AFTER the "
                        "headline, which has already absorbed the news. 'pre' "
                        "(optimistic): fill at the last quote BEFORE the headline "
                        "-- flatters returns because you capture the news move for "
                        "free; kept only for comparison.")
    p.add_argument('--entry-max-gap-hours', type=float, default=24.0,
                   help="post mode: how long after the headline a fill may sit; "
                        "if the next quote is further out, the market is not "
                        "tradeable around the news and the cell is dropped")
    p.add_argument('--important-only', action='store_true', default=True,
                   help='only trade on jin10 important==1 headlines')
    p.add_argument('--all-news', dest='important_only', action='store_false',
                   help='trade on every headline, not just important ones')
    p.add_argument('--train-frac', type=float, default=0.80)
    p.add_argument('--valid-frac', type=float, default=0.10)
    p.add_argument('--train-cutoff', type=str, default='2026-05-24',
                   help='Date the checkpoint stopped training (UTC). Headlines '
                        'before this are dropped so the run stays out of sample; '
                        'set to "" to skip the check.')
    p.add_argument('--embed-model', type=str, default='BAAI/bge-small-en-v1.5')
    p.add_argument('--embed-device', type=str, default='cuda')
    p.add_argument('--max-staleness-hours', type=float, default=72.0,
                   help='how old the last quote may be for a market to count as live')
    p.add_argument('--max-settle-gap-hours', type=float, default=48.0,
                   help='how far past the hold horizon a settlement quote may sit')
    p.add_argument('--calib-records', type=int, default=400,
                   help='train records used to fit the similarity calibration')
    p.add_argument('--limit-news', type=int, default=None,
                   help='cap the number of headlines processed (smoke tests)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    lookback = int(args.news_lookback_hours * HOUR)
    hold = int(args.hold_hours * HOUR)
    settle_gap = int(args.max_settle_gap_hours * HOUR)
    entry_gap = int(args.entry_max_gap_hours * HOUR)
    stale = int(args.max_staleness_hours * HOUR)

    records = load_records(args.data)
    train, valid, test = temporal_split(records, args.train_frac, args.valid_frac)
    print(f'records={len(records)} train={len(train)} valid={len(valid)} test={len(test)}')

    series = PriceSeries(records)
    stream = NewsStream(records)
    print(f'markets={len(series.market_ids)} news_stream={len(stream)}')

    book = None
    if args.price_series:
        book = HourlyPrices(args.price_series, keep=set(series.market_ids))
        covered = sum(1 for m in series.market_ids if m in book)
        print(f'[hourly] loaded {args.price_series}: {covered}/{len(series.market_ids)} '
              f'markets have an hourly book')

    # Out-of-sample window: mirror the grid builder's cutoff assertion, then trade
    # only headlines published from the start of the test window onward.
    news_start = min(record_time(r) for r in test)
    if args.train_cutoff:
        cutoff = int(
            dt.datetime.strptime(args.train_cutoff, '%Y-%m-%d')
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
        if news_start < cutoff:
            raise SystemExit(
                f'test window starts {dt.datetime.fromtimestamp(news_start, dt.timezone.utc):%Y-%m-%d}, '
                f'before the {args.train_cutoff} train cutoff -- would be in sample'
            )
        news_start = max(news_start, cutoff)
    print(
        f'trading headlines from '
        f'{dt.datetime.fromtimestamp(news_start, dt.timezone.utc):%Y-%m-%d} onward'
    )

    triggers = [
        n for n in stream.items
        if n.t >= news_start and (not args.important_only or n.important == 1)
    ]
    if args.limit_news:
        triggers = triggers[: args.limit_news]
    print(f'trigger headlines: {len(triggers)} '
          f'(important_only={args.important_only})')

    # One bi-encoder, two directions. Fit calibration on TRAIN records first
    # (this transiently fits the retriever on the train corpora), then re-fit
    # both corpora on everything the test window needs: all markets, and the
    # union of every prompt window plus the triggers themselves.
    retriever = EmbeddingRetriever(args.embed_model, device=args.embed_device)
    calibration = fit_calibration(train, retriever, args.calib_records, args.seed)

    uniq = {}
    for n in triggers:
        for item in NewsStream.dedupe(stream.window(n.t - lookback, n.t + 1)):
            uniq[(item.title, item.t)] = item
    corpus = list(uniq.values())
    news_index_of = {(i.title, i.t): j for j, i in enumerate(corpus)}
    retriever.fit_news([i.text for i in corpus])
    market_ids = series.market_ids
    retriever.fit_markets(market_ids, [series.meta[m].text for m in market_ids])
    print(f'[retrieval] news corpus={len(corpus)} markets={len(market_ids)}')

    def prompt_news_for(market_id, decision_t):
        """Forward retrieval: the market's most relevant news up to the headline.

        Anchored at `decision_t` (the publication time), never at the settlement
        time, so nothing published after the trade is opened can enter the prompt.
        """
        items = NewsStream.dedupe(stream.window(decision_t - lookback, decision_t + 1))
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

    # Dedupe cells on (market_id, target_t): backtest_predict.py keys on that and
    # would otherwise silently overwrite. Two headlines at the same second landing
    # on the same market collapse to the one with the stronger relevance.
    cells: dict[tuple, dict] = {}
    n_pairs = n_skipped = 0
    live_cache: dict[int, list[str]] = {}
    for n in tqdm(triggers, desc='news-driven'):
        decision_t = n.t
        target_t = decision_t + hold
        trigger_idx = news_index_of.get((n.title, n.t))
        if trigger_idx is None:
            continue
        live = live_cache.get(decision_t)
        if live is None:
            if book is not None:
                live = [
                    m for m in market_ids
                    if book.has_near(m, decision_t, stale)
                ]
            else:
                live = [
                    m for m in market_ids
                    if series.is_live(m, decision_t, max_staleness=stale)
                ]
            live_cache[decision_t] = live
        hits = retriever.top_markets(
            trigger_idx, live, k=args.top_markets, calibration=calibration
        )
        for hit in hits:
            if hit.score <= 0.0:  # headline is not about this market -- decline
                continue
            market_id = hit.market_id
            # Realistic fill: the first quote AT/AFTER the headline, which has
            # already absorbed the news. With an hourly book that is the next
            # hourly tick (~<=1h lag); on the daily series it snaps to the next
            # daily quote. 'pre' fills at the last quote BEFORE the headline --
            # optimistic, since it hands you the news move for free.
            if book is not None:
                entry_quote = book.forward_quote(
                    market_id, decision_t, max_gap=entry_gap
                )
            elif args.entry_timing == 'post':
                entry_quote = series.forward_quote(
                    market_id, decision_t, max_gap=entry_gap
                )
            else:
                entry_quote = series.quote(market_id, decision_t)
            if entry_quote is None:
                n_skipped += 1
                continue
            history = series.daily_history(market_id, target_t, as_of=decision_t)
            if len(history) < MIN_HISTORY_POINTS:
                n_skipped += 1
                continue
            # Hold from the actual fill, not from the headline, so the horizon is
            # honest even when the fill lands after the news.
            settle_src = book if book is not None else series
            settle = settle_src.forward_quote(
                market_id, entry_quote[0] + hold, max_gap=settle_gap
            )
            if settle is None or settle[0] <= entry_quote[0]:
                n_skipped += 1
                continue
            n_pairs += 1
            key = (market_id, target_t)
            prev = cells.get(key)
            if prev is not None and prev['_bt']['trigger_sim'] >= hit.similarity:
                continue
            news, attributions = prompt_news_for(market_id, decision_t)
            meta = series.meta[market_id]
            cells[key] = {
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
                    'mode': 'news',
                    'news_source': args.news,
                    'entry_timing': args.entry_timing,
                    'is_breakpoint': False,
                    'decision_t': decision_t,
                    'entry_t': entry_quote[0],
                    'entry_price': entry_quote[1],
                    'entry_lag': entry_quote[0] - decision_t,
                    'entry_staleness': decision_t - entry_quote[0],
                    'settle_t': settle[0],
                    'settle_price': settle[1],
                    'hold_seconds': settle[0] - entry_quote[0],
                    'prev_price': history[-1]['p'],
                    'resolution': meta.outcome,
                    'n_news': len(news),
                    'trigger_title': n.title,
                    'trigger_t': n.t,
                    'trigger_sim': float(hit.similarity),
                    'trigger_score': float(hit.score),
                },
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cells.values(), key=lambda c: (c['_bt']['decision_t'], c['market_id']))
    holds = []
    with out_path.open('w') as fout:
        for cell in ordered:
            holds.append(cell['_bt']['hold_seconds'])
            fout.write(json.dumps(cell, ensure_ascii=False) + '\n')

    print(f'headline->market pairs={n_pairs} skipped={n_skipped} '
          f'distinct cells={len(cells)} -> {out_path}')
    if holds:
        holds.sort()
        print(
            f'hold seconds: p10={holds[len(holds) // 10] / HOUR:.1f}h '
            f'median={holds[len(holds) // 2] / HOUR:.1f}h '
            f'p90={holds[9 * len(holds) // 10] / HOUR:.1f}h'
        )


if __name__ == '__main__':
    main()
