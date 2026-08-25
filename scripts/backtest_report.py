#!/usr/bin/env python3
"""Turn scored backtest cells into P&L, baselines and sensitivity tables.

Emits a JSON blob (consumed by the HTML report) holding, for each prediction
file: the usual forecasting metrics, the headline strategy comparison, a
threshold x cost x entry-drift sweep, and the equity curve.

Two settlement conventions are reported because they answer different
questions. `move` closes the position at the next observed quote -- for a
breakpoint cell that is exactly the 2h move the dataset was built around.
`resolution` holds to the market's final outcome (1 or 0), which is what a
buy-and-hold trader would actually realise but folds in weeks of unrelated
drift.
"""

import argparse
import json
import math
import statistics as stats
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm.backtest.engine import (
    BacktestConfig,
    Candidate,
    always_no_signal,
    always_yes_signal,
    equity_curve,
    evaluate_strategies,
    run_strategy,
    swm_signal,
    trade_metrics,
)

THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
COSTS = [0.0, 0.005, 0.01, 0.02]
DRIFTS = [0.0, 0.25, 0.50]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preds', nargs='+', required=True,
                   help='name=path.jsonl pairs, e.g. retrieval=results/.../preds.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--cost', type=float, default=0.01)
    p.add_argument('--entry-drift', type=float, default=0.0)
    p.add_argument('--sizing', default='fixed_notional',
                   choices=['fixed_notional', 'fixed_shares'])
    p.add_argument('--breakpoints-only', action='store_true',
                   help='restrict a grid run to its breakpoint cells')
    return p.parse_args()


def load_candidates(path: str, exit_mode: str, breakpoints_only: bool):
    out: list[Candidate] = []
    for line in open(path):
        row = json.loads(line)
        bt = row['_bt']
        if breakpoints_only and not bt.get('is_breakpoint'):
            continue
        if exit_mode == 'resolution':
            settle = bt.get('resolution')
            if settle is None:
                continue
        else:
            settle = bt['settle_price']
        out.append(
            Candidate(
                t=int(row['t']),
                market_id=str(row['market_id']),
                event_id=str(row.get('event_id') or row['market_id']),
                entry=float(bt['entry_price']),
                exit=float(settle),
                pred_price=float(row['pred_price']),
                prev_price=float(bt['prev_price']),
                extra={
                    'entry_t': bt.get('entry_t'),
                    'settle_t': bt.get('settle_t'),
                    'is_breakpoint': bt.get('is_breakpoint', True),
                    'scored': row.get('scored', True),
                    'n_news': row.get('n_news', 0),
                    'hold_seconds': bt.get('hold_seconds'),
                    'question': row.get('question', ''),
                    'true_price': row.get('true_price'),
                },
            )
        )
    return out


def information_metrics(candidates: Sequence[Candidate]) -> dict:
    """Does the forecast add anything over what the book already says?

    The model is trained to predict the price 24h ahead of a 24h-old anchor. By
    the time a trade can be placed, the book has moved most of that distance
    already, so a large share of the training target is public information
    rather than alpha. These numbers separate the two:

      * `model_rmse` vs `quote_rmse` -- is the forecast a better estimate of the
        settlement price than simply reading the current quote;
      * `target_overlap` -- how much of the trained target is already visible;
      * `residual_corr` -- correlation of the model's own delta with the part of
        the move that is still on the table at entry, which is the only part a
        trade can capture.
    """
    if len(candidates) < 3:
        return {}
    prev = [c.prev_price for c in candidates]
    entry = [c.entry for c in candidates]
    true = [
        c.extra.get('true_price') if c.extra.get('true_price') is not None else c.exit
        for c in candidates
    ]
    pred = [c.pred_price for c in candidates]
    delta = [p - q for p, q in zip(pred, prev)]
    trained_target = [t - q for t, q in zip(true, prev)]
    already = [e - q for e, q in zip(entry, prev)]
    residual = [t - e for t, e in zip(true, entry)]

    def _corr(a, b):
        n = len(a)
        ma, mb = stats.fmean(a), stats.fmean(b)
        sa, sb = stats.pstdev(a), stats.pstdev(b)
        if not sa or not sb:
            return float('nan')
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n / (sa * sb)

    def _rmse(a, b):
        return math.sqrt(stats.fmean([(x - y) ** 2 for x, y in zip(a, b)]))

    n = len(candidates)
    residual_corr = _corr(delta, residual)
    return {
        'n': n,
        'model_rmse': _rmse(pred, true),
        'quote_rmse': _rmse(entry, true),
        'anchor_rmse': _rmse(prev, true),
        'target_overlap': _corr(trained_target, already),
        'trained_corr': _corr(delta, trained_target),
        'residual_corr': residual_corr,
        'residual_t': (
            residual_corr * math.sqrt((n - 2) / max(1e-12, 1 - residual_corr ** 2))
        ),
        'mean_abs_trained_move': stats.fmean([abs(x) for x in trained_target]),
        'mean_abs_already': stats.fmean([abs(x) for x in already]),
        'mean_abs_residual': stats.fmean([abs(x) for x in residual]),
    }


def forecast_metrics(candidates: Sequence[Candidate]) -> dict:
    """Regression-style metrics, on the same cells the trades come from."""
    pred = [c.pred_price - c.prev_price for c in candidates]
    true = [
        (c.extra.get('true_price') if c.extra.get('true_price') is not None else c.exit)
        - c.prev_price
        for c in candidates
    ]
    n = len(pred)
    if n < 2:
        return {'n': n}
    mp, mt = stats.fmean(pred), stats.fmean(true)
    sp, st = stats.pstdev(pred), stats.pstdev(true)
    cov = sum((a - mp) * (b - mt) for a, b in zip(pred, true)) / n
    moved = [(a, b) for a, b in zip(pred, true) if abs(b) > 1e-6 and abs(a) > 1e-9]
    baseline = sum(b * b for b in true) / n
    mse = sum((a - b) ** 2 for a, b in zip(pred, true)) / n
    return {
        'n': n,
        'pearson': cov / (sp * st) if sp and st else float('nan'),
        'mae': sum(abs(a - b) for a, b in zip(pred, true)) / n,
        'rmse': math.sqrt(mse),
        'no_change_rmse': math.sqrt(baseline),
        'skill_vs_no_change': 1 - mse / baseline if baseline else float('nan'),
        'direction_accuracy': (
            sum(1 for a, b in moved if (a > 0) == (b > 0)) / len(moved) if moved else float('nan')
        ),
        'pred_std': sp,
        'true_std': st,
    }


def sweep(candidates: Sequence[Candidate], base: BacktestConfig) -> list[dict]:
    rows = []
    for cost in COSTS:
        for drift in DRIFTS:
            for thr in THRESHOLDS:
                cfg = BacktestConfig(
                    threshold=thr, cost=cost, entry_drift=drift,
                    sizing=base.sizing, seed=base.seed,
                )
                trades = run_strategy(candidates, swm_signal, cfg)
                m = trade_metrics(trades, n_candidates=len(candidates), bootstrap=False)
                rows.append({'cost': cost, 'entry_drift': drift, 'threshold': thr, **m})
    return rows


def unique_positions(candidates: Sequence[Candidate]) -> list:
    """First cell per distinct position.

    Forecast and information metrics have to be computed on these, not on raw
    cells: in grid mode one daily-quoted position is re-offered ~4x, and
    averaging over cells would silently weight those positions four times.
    """
    seen, out = set(), []
    for c in sorted(candidates, key=lambda x: x.t):
        if c.position_key in seen:
            continue
        seen.add(c.position_key)
        out.append(c)
    return out


def describe(candidates: Sequence[Candidate]) -> dict:
    holds = [c.extra['hold_seconds'] for c in candidates if c.extra.get('hold_seconds')]
    moves = [c.exit - c.entry for c in candidates]
    return {
        'n_cells': len(candidates),
        'n_distinct_positions': len({c.position_key for c in candidates}),
        'n_breakpoint_cells': sum(1 for c in candidates if c.extra.get('is_breakpoint')),
        'n_markets': len({c.market_id for c in candidates}),
        'n_events': len({c.event_id for c in candidates}),
        'n_times': len({c.t for c in candidates}),
        't_start': min(c.t for c in candidates) if candidates else None,
        't_end': max(c.t for c in candidates) if candidates else None,
        'median_hold_hours': (stats.median(holds) / 3600) if holds else None,
        'mean_move': stats.fmean(moves) if moves else None,
        'mean_abs_move': stats.fmean([abs(m) for m in moves]) if moves else None,
        'frac_move_up': (sum(1 for m in moves if m > 0) / len(moves)) if moves else None,
        'mean_entry_price': stats.fmean([c.entry for c in candidates]) if candidates else None,
        'frac_routed_null': (
            sum(1 for c in candidates if abs(c.pred_price - c.prev_price) < 1e-9)
            / len(candidates)
        ) if candidates else None,
    }


def stratify(candidates: Sequence[Candidate], cfg: BacktestConfig) -> dict:
    """Where the P&L comes from: cell type, entry price, holding period, month.

    A single blended ROI hides the two things most likely to be driving it --
    cheap longshots under fixed-notional sizing, and the long holds that quiet
    markets are forced into because they are only quoted daily.
    """
    import datetime as dt

    def slice_metrics(subset):
        if len(subset) < 5:
            return None
        trades = run_strategy(subset, swm_signal, cfg)
        base = run_strategy(subset, lambda c: 1.0, cfg, threshold=0.0)
        return {
            'n_cells': len(subset),
            'swm': trade_metrics(trades, n_candidates=len(subset), bootstrap=False),
            'always_yes': trade_metrics(base, n_candidates=len(subset), bootstrap=False),
        }

    def bucket(name, keyfn):
        groups: dict[str, list[Candidate]] = {}
        for c in candidates:
            groups.setdefault(keyfn(c), []).append(c)
        return {
            name: {k: m for k, v in sorted(groups.items()) if (m := slice_metrics(v))}
        }

    def price_bucket(c):
        for hi, label in ((0.10, '0.02-0.10'), (0.25, '0.10-0.25'),
                          (0.50, '0.25-0.50'), (0.75, '0.50-0.75')):
            if c.entry < hi:
                return label
        return '0.75-0.98'

    def hold_bucket(c):
        hours = (c.extra.get('hold_seconds') or 0) / 3600
        for hi, label in ((3, '<=3h'), (8, '3-8h'), (16, '8-16h'), (26, '16-26h')):
            if hours <= hi:
                return label
        return '>26h'

    out = {}
    out.update(bucket('by_entry_price', price_bucket))
    out.update(bucket('by_hold', hold_bucket))
    out.update(bucket('by_cell_type',
                      lambda c: 'breakpoint' if c.extra.get('is_breakpoint') else 'quiet'))
    out.update(bucket('by_month',
                      lambda c: dt.datetime.fromtimestamp(c.t, dt.timezone.utc).strftime('%Y-%m')))
    return out


def main():
    args = parse_args()
    report = {
        'config': {
            'threshold': args.threshold, 'cost': args.cost,
            'entry_drift': args.entry_drift, 'sizing': args.sizing,
            'breakpoints_only': args.breakpoints_only,
        },
        'runs': {},
    }
    for spec in args.preds:
        name, path = spec.split('=', 1)
        entry = {}
        for exit_mode in ('move', 'resolution'):
            candidates = load_candidates(path, exit_mode, args.breakpoints_only)
            if not candidates:
                continue
            cfg = BacktestConfig(
                threshold=args.threshold, cost=args.cost,
                entry_drift=args.entry_drift, sizing=args.sizing,
            )
            unique = unique_positions(candidates) if cfg.dedupe_positions else list(
                candidates
            )
            block = {
                'universe': describe(candidates),
                'forecast': forecast_metrics(unique),
                'information': information_metrics(unique),
                'strategies': evaluate_strategies(candidates, cfg),
            }
            trades = run_strategy(candidates, swm_signal, cfg)
            block['equity_curve'] = equity_curve(trades)
            # The baselines get curves too: a single line is unreadable without
            # the thing it is supposed to be beating drawn next to it.
            block['equity_curves'] = {
                'SWM': block['equity_curve'],
                'Always YES': equity_curve(
                    run_strategy(candidates, always_yes_signal, cfg, threshold=0.0)
                ),
                'Always NO': equity_curve(
                    run_strategy(candidates, always_no_signal, cfg, threshold=0.0)
                ),
            }
            block['n_trades'] = len(trades)
            if exit_mode == 'move':
                block['sweep'] = sweep(candidates, cfg)
                block['strata'] = stratify(candidates, cfg)
                block['trades'] = [
                    {
                        't': t.t, 'market_id': t.market_id, 'direction': t.direction,
                        'entry': t.entry, 'exit': t.exit, 'signal': t.signal,
                        'pnl': t.pnl, 'ret': t.ret,
                    }
                    for t in trades
                ]
            entry[exit_mode] = block
        report['runs'][name] = entry
        print(f'== {name} ==')
        for exit_mode, block in entry.items():
            swm = block['strategies'].get('swm', {})
            yes = block['strategies'].get('always_yes', {})
            matched = (swm.get('matched') or {}).get('always_yes', {})
            print(
                f'  [{exit_mode:10s}] cells={block["universe"]["n_cells"]:6d} '
                f'trades={swm.get("n_trades", 0):5d} '
                f'ROI={swm.get("roi", 0):+.2%} '
                f'| always-YES all {yes.get("roi", 0):+.2%} '
                f'matched {matched.get("roi", float("nan")):+.2%} '
                f'| hit={swm.get("hit_rate", float("nan")):.1%} '
                f'corr={block["forecast"].get("pearson", float("nan")):.3f}'
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
