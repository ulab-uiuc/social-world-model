"""Trading rules, cost model, baselines and P&L statistics.

Conventions for a binary market quoted at price p in (0, 1):

  * a YES share costs p and pays 1 if the market resolves YES;
  * a NO share costs 1 - p and pays 1 if it resolves NO;
  * exiting early at quote q pays q per YES share (1 - q per NO share).

So per share held, P&L is (q - p) long and (p - q) short, while the capital
actually tied up is p long and 1 - p short. Both numbers are reported: per
share is the honest "cents of edge" measure, while return on capital is what
"收益率" usually means -- but it is dominated by cheap longshots, where one
dollar buys 50 shares, so neither is meaningful alone.
"""

import math
import random as _random
import statistics as _stats
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

PRICE_FLOOR = 0.02  # clamp quotes away from 0/1 so returns stay finite


@dataclass
class BacktestConfig:
    """Knobs the headline number is most sensitive to."""

    threshold: float = 0.05  # minimum |edge| in price units to take a position
    cost: float = 0.01  # half-spread + fees, in price units, charged on entry
    exit_cost: float = 0.0  # additional cost charged on exit
    entry_drift: float = 0.0  # fraction of the move already in the book at entry
    sizing: str = 'fixed_notional'  # 'fixed_notional' | 'fixed_shares'
    stake: float = 1.0  # dollars of capital (notional) or shares per trade
    price_floor: float = PRICE_FLOOR
    max_positions_per_time: int | None = None  # cap simultaneous trades
    dedupe_positions: bool = True  # one fill per distinct (market, entry, exit)
    seed: int = 42


@dataclass
class Trade:
    t: int
    market_id: str
    event_id: str
    direction: int  # +1 YES, -1 NO
    entry: float
    exit: float
    signal: float
    capital: float
    pnl: float
    shares: float = 1.0

    @property
    def ret(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


@dataclass
class Candidate:
    """One scored (market, decision-time) cell, ready to be traded or skipped."""

    t: int
    market_id: str
    event_id: str
    entry: float  # quote observable at decision time
    exit: float  # settlement quote
    pred_price: float  # world-model price forecast
    prev_price: float  # last daily history point (24h-ago quote)
    extra: dict = field(default_factory=dict)

    @property
    def edge(self) -> float:
        return self.pred_price - self.entry

    @property
    def position_key(self) -> tuple:
        """Identity of the underlying position, independent of decision time.

        A quiet market is quoted once a day, so every decision time inside one
        inter-quote gap resolves to the same entry quote and the same settlement
        quote. Those cells are not separate opportunities -- they are the same
        trade offered repeatedly, and only the news window differs.
        """
        return (
            self.market_id,
            self.extra.get('entry_t', self.t),
            self.extra.get('settle_t', self.t),
        )


def _clamp(p: float, floor: float) -> float:
    return min(max(p, floor), 1.0 - floor)


def make_trade(
    candidate: Candidate, direction: int, config: BacktestConfig
) -> Trade | None:
    """Turn a signed decision into a filled trade under the cost model."""
    if direction == 0:
        return None
    entry = _clamp(candidate.entry, config.price_floor)
    exit_price = _clamp(candidate.exit, config.price_floor)

    # We can only quote the last observed price, which predates the fill.
    # entry_drift is the share of the eventual move already in the book when the
    # order lands, so the fill sits that much closer to the exit. It shrinks the
    # capturable move symmetrically -- taxing a correct call and refunding an
    # incorrect one -- which is the real physics of arriving late, not a
    # one-sided penalty. Net effect is adverse for any signal better than a coin
    # flip; entry_drift=1 leaves no move to capture at all.
    if config.entry_drift:
        entry = _clamp(
            entry + config.entry_drift * (exit_price - entry), config.price_floor
        )

    # Cost is paid in price units on the side actually bought. Clamping the
    # result would silently refund the spread whenever side_price + cost lands
    # above the ceiling, so an unaffordable fill is dropped instead.
    side_price = entry if direction > 0 else 1.0 - entry
    cost_price = side_price + config.cost
    if cost_price >= 1.0:
        return None
    cost_price = max(cost_price, config.price_floor)
    payout = exit_price if direction > 0 else 1.0 - exit_price
    payout = max(payout - config.exit_cost, 0.0)

    per_share = payout - cost_price
    if config.sizing == 'fixed_shares':
        shares = config.stake
        capital = cost_price * shares
    else:  # fixed_notional: spend `stake` dollars of capital
        capital = config.stake
        shares = capital / cost_price

    return Trade(
        t=candidate.t,
        market_id=candidate.market_id,
        event_id=candidate.event_id,
        direction=direction,
        entry=entry,
        exit=exit_price,
        signal=candidate.edge,
        capital=capital,
        pnl=per_share * shares,
        shares=shares,
    )


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

SignalFn = Callable[[Candidate], float]


def swm_signal(c: Candidate) -> float:
    return c.edge


def swm_delta_signal(c: Candidate) -> float:
    """The model's own forecast delta, ignoring how far the book already moved.

    `swm_signal` compares the forecast against the live quote, so part of its
    edge is really mean reversion: the model anchors on the 24h-ago price while
    the entry quote is 2h old. This variant isolates the model's directional
    call from that anchoring difference.
    """
    return c.pred_price - c.prev_price


def momentum_signal(c: Candidate) -> float:
    """Yesterday-to-now drift, the cheapest non-model directional signal."""
    return c.entry - c.prev_price


def contrarian_signal(c: Candidate) -> float:
    return -momentum_signal(c)


def always_yes_signal(c: Candidate) -> float:
    return 1.0


def always_no_signal(c: Candidate) -> float:
    return -1.0


def perfect_direction_signal(c: Candidate) -> float:
    """Perfect foresight on direction only.

    Note this is a ceiling *per cell traded*, not a ceiling on ROI: under
    fixed-notional sizing a strategy that trades only cheap markets can post a
    higher return on capital than perfect foresight applied to every cell, since
    a dollar buys fifty shares at 2c and one share at 95c. That is exactly why
    the baselines below are also run on the model's own traded subset.
    """
    return c.exit - c.entry


# Back-compat alias.
oracle_signal = perfect_direction_signal


def random_signal_factory(seed: int) -> SignalFn:
    def signal(c: Candidate) -> float:
        rng = _random.Random(f'{seed}:{c.market_id}:{c.t}')
        return 1.0 if rng.random() < 0.5 else -1.0

    return signal


def run_strategy(
    candidates: Sequence[Candidate],
    signal_fn: SignalFn,
    config: BacktestConfig,
    threshold: float | None = None,
) -> list[Trade]:
    """Apply a signal to every candidate in time order and collect the fills."""
    thr = config.threshold if threshold is None else threshold
    trades: list[Trade] = []
    by_time: dict[int, list[tuple]] = {}
    # One fill per distinct position, taken at the first decision time that
    # fires on it. Without this the grid counts a single daily-quoted position
    # ~4x over -- inflating n_trades, shrinking the naive t-stat by roughly the
    # duplication factor, and re-weighting ROI toward the quiet long-hold
    # markets that happen to be duplicated most. It sits inside run_strategy so
    # the model and every baseline are deduplicated on identical terms.
    filled: set = set()
    for c in sorted(candidates, key=lambda x: (x.t, x.market_id)):
        if config.dedupe_positions and c.position_key in filled:
            continue
        signal = signal_fn(c)
        # A zero signal is *no opinion*, not a short. It has to be excluded
        # explicitly: at threshold 0 the `abs(signal) < thr` test never fires,
        # and the cells the model routed to its null option (36% of the grid,
        # emitted as pred_delta = 0) would otherwise all be sold short at full
        # size -- turning "the model declined to forecast" into a position.
        if signal == 0.0 or abs(signal) < thr:
            continue
        if config.dedupe_positions:
            filled.add(c.position_key)
        by_time.setdefault(c.t, []).append((abs(signal), c, 1 if signal > 0 else -1))

    for t in sorted(by_time):
        picks = sorted(by_time[t], key=lambda x: -x[0])
        if config.max_positions_per_time:
            picks = picks[: config.max_positions_per_time]
        for _, candidate, direction in picks:
            trade = make_trade(candidate, direction, config)
            if trade is not None:
                trades.append(trade)
    return trades


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _max_drawdown(trades: Sequence['Trade']) -> float:
    """Deepest peak-to-trough fall of cumulative P&L, as a share of capital.

    Measured on the P&L path in dollars and divided by total capital deployed.
    Reading it off the normalised equity curve instead would divide early
    points by capital that has not been committed yet, which flattens the
    early swings and understates the drawdown.
    """
    total_capital = sum(t.capital for t in trades)
    if not total_capital:
        return 0.0
    cumulative, peak, worst = 0.0, 0.0, 0.0
    for trade in sorted(trades, key=lambda x: x.t):
        cumulative += trade.pnl
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return worst / total_capital


def equity_curve(trades: Sequence[Trade], initial: float = 1.0) -> list[dict]:
    """Cumulative P&L over time, normalised by total capital deployed.

    Every trade is treated as an independent unit-capital bet (no compounding),
    so the curve reads as cumulative return on the money put to work.
    """
    ordered = sorted(trades, key=lambda x: x.t)
    total_capital = sum(t.capital for t in ordered) or 1.0
    cumulative, points = 0.0, []
    for trade in ordered:
        cumulative += trade.pnl
        points.append(
            {
                't': trade.t,
                'pnl': cumulative,
                'equity': initial + cumulative / total_capital,
                'market_id': trade.market_id,
            }
        )
    return points


def _bootstrap_roi_ci(
    trades: Sequence[Trade],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple | None:
    """Cluster bootstrap on event_id.

    Several markets belong to one event and several decision times hit the same
    market, so trade outcomes are far from independent; resampling whole event
    blocks keeps that correlation in the interval.
    """
    if len(trades) < 5:
        return None
    blocks: dict[str, list[Trade]] = {}
    for trade in trades:
        blocks.setdefault(trade.event_id or trade.market_id, []).append(trade)
    keys = list(blocks)
    rng = _random.Random(seed)
    rois = []
    for _ in range(n_resamples):
        pnl = capital = 0.0
        for _ in keys:
            for trade in blocks[keys[rng.randrange(len(keys))]]:
                pnl += trade.pnl
                capital += trade.capital
        if capital:
            rois.append(pnl / capital)
    if not rois:
        return None
    rois.sort()
    lo = rois[int(alpha / 2 * len(rois))]
    hi = rois[min(len(rois) - 1, int((1 - alpha / 2) * len(rois)))]
    return lo, hi


def trade_metrics(
    trades: Sequence[Trade],
    n_candidates: int | None = None,
    bootstrap: bool = True,
    seed: int = 0,
) -> dict:
    if not trades:
        return {'n_trades': 0, 'roi': 0.0, 'pnl': 0.0}
    pnl = sum(t.pnl for t in trades)
    capital = sum(t.capital for t in trades)
    rets = [t.ret for t in trades]
    per_share = [
        (t.exit - t.entry) if t.direction > 0 else (t.entry - t.exit) for t in trades
    ]
    mean_ret = pnl / capital if capital else 0.0
    sd = _stats.pstdev(rets) if len(rets) > 1 else 0.0
    curve = equity_curve(trades)
    out = {
        'n_trades': len(trades),
        'coverage': (len(trades) / n_candidates) if n_candidates else None,
        'pnl': pnl,
        'capital_deployed': capital,
        'roi': mean_ret,
        'mean_trade_return': _stats.fmean(rets),
        'median_trade_return': _stats.median(rets),
        # Gross is the raw price move captured before the spread; net is the
        # same quantity after it. Both are averaged over TRADES, not over
        # shares -- a share-weighted net would be dominated by the cheap
        # markets where one dollar buys fifty shares, and would not be
        # comparable to the gross figure sitting next to it. On a thin book the
        # two differ by roughly the spread, which is often the whole result.
        'edge_per_share': _stats.fmean(per_share),
        'net_edge_per_share': _stats.fmean(
            [t.pnl / t.shares for t in trades if t.shares]
        ),
        'hit_rate': sum(1 for t in trades if t.pnl > 0) / len(trades),
        'long_share': sum(1 for t in trades if t.direction > 0) / len(trades),
        'sharpe_per_trade': (_stats.fmean(rets) / sd) if sd else float('nan'),
        'return_std': sd,
        'max_drawdown': _max_drawdown(trades),
        'final_equity': curve[-1]['equity'] if curve else 1.0,
    }
    if len(rets) > 1 and sd:
        out['t_stat'] = _stats.fmean(rets) / (sd / math.sqrt(len(rets)))
    if bootstrap:
        ci = _bootstrap_roi_ci(trades, seed=seed)
        if ci:
            out['roi_ci_low'], out['roi_ci_high'] = ci
    return out


MATCHED_BASELINES: dict[str, SignalFn] = {
    'always_yes': always_yes_signal,
    'always_no': always_no_signal,
    'perfect_direction': perfect_direction_signal,
}


def matched_baselines(
    candidates: Sequence[Candidate],
    model_trades: Sequence[Trade],
    config: BacktestConfig,
) -> dict[str, dict]:
    """Baselines restricted to the cells the model itself chose to trade.

    Comparing a selective model against a baseline that trades everything
    confounds two things: whether the model picks good cells, and whether it
    calls direction better than a coin. Re-running the trivial rules on exactly
    the model's traded subset isolates the second, which is the question
    "does the forecast add anything" actually turns on.
    """
    keys = {(t.market_id, t.t) for t in model_trades}
    subset = [c for c in candidates if (c.market_id, c.t) in keys]
    if not subset:
        return {}
    out: dict[str, dict] = {}
    for name, fn in MATCHED_BASELINES.items():
        trades = run_strategy(subset, fn, config, threshold=0.0)
        out[name] = trade_metrics(trades, n_candidates=len(subset), seed=config.seed)
    rnd = run_strategy(subset, random_signal_factory(config.seed), config, threshold=0.0)
    out['random'] = trade_metrics(rnd, n_candidates=len(subset), seed=config.seed)
    return out


DEFAULT_STRATEGIES: dict[str, SignalFn] = {
    'swm': swm_signal,
    'swm_delta': swm_delta_signal,
    'momentum': momentum_signal,
    'contrarian': contrarian_signal,
    'always_yes': always_yes_signal,
    'always_no': always_no_signal,
}


def evaluate_strategies(
    candidates: Sequence[Candidate],
    config: BacktestConfig,
    strategies: dict[str, SignalFn] | None = None,
) -> dict[str, dict]:
    """Run every strategy over the same candidate set under identical costs.

    Baselines that ignore the signal (always_yes / always_no / random) are run
    at threshold 0 so they trade the full candidate set; comparing them against
    the model at its own threshold is the point -- a model that only fires on
    5% of cells has to earn more per trade to be worth anything.
    """
    strategies = dict(strategies or DEFAULT_STRATEGIES)
    strategies['random'] = random_signal_factory(config.seed)
    strategies['perfect_direction'] = perfect_direction_signal

    results: dict[str, dict] = {}
    for name, fn in strategies.items():
        unconditional = name in ('always_yes', 'always_no', 'random')
        thr = 0.0 if unconditional else config.threshold
        trades = run_strategy(candidates, fn, config, threshold=thr)
        results[name] = trade_metrics(
            trades, n_candidates=len(candidates), seed=config.seed
        )
        results[name]['threshold'] = thr
        if name.startswith('swm'):
            results[name]['matched'] = matched_baselines(candidates, trades, config)
    return results
