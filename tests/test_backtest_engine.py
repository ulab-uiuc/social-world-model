"""Hand-checked arithmetic for the backtest P&L, and the no-lookahead guarantees."""


import pytest

from swm.backtest.engine import (
    BacktestConfig,
    Candidate,
    equity_curve,
    make_trade,
    matched_baselines,
    run_strategy,
    swm_signal,
    trade_metrics,
)
from swm.backtest.universe import PriceSeries

DAY = 86400


def cell(entry, exit_, pred=None, prev=None, t=1000, mid='m1', eid='e1'):
    return Candidate(
        t=t, market_id=mid, event_id=eid, entry=entry, exit=exit_,
        pred_price=entry if pred is None else pred,
        prev_price=entry if prev is None else prev,
    )


# --------------------------------------------------------------- P&L math

def test_long_pnl_matches_hand_calculation():
    # Buy YES at 0.40 with a 0.01 spread => 0.41 a share. $1 of capital buys
    # 1/0.41 = 2.4390 shares. Exit at 0.60 => 0.19 a share => +0.4634 on $1.
    cfg = BacktestConfig(cost=0.01, sizing='fixed_notional', stake=1.0)
    trade = make_trade(cell(0.40, 0.60), +1, cfg)
    assert trade.pnl == pytest.approx((0.60 - 0.41) / 0.41, rel=1e-9)
    assert trade.capital == 1.0
    assert trade.ret == pytest.approx(0.46341463, rel=1e-6)


def test_short_pnl_uses_the_no_side_price():
    # Buy NO at 1-0.40 = 0.60, +0.01 spread => 0.61. Exit quote 0.25 pays
    # 1-0.25 = 0.75 a share => +0.14 a share on 0.61 of cost.
    cfg = BacktestConfig(cost=0.01, sizing='fixed_notional', stake=1.0)
    trade = make_trade(cell(0.40, 0.25), -1, cfg)
    assert trade.pnl == pytest.approx((0.75 - 0.61) / 0.61, rel=1e-9)


def test_fixed_shares_capital_is_the_cost_of_one_share():
    cfg = BacktestConfig(cost=0.01, sizing='fixed_shares', stake=1.0)
    long_trade = make_trade(cell(0.40, 0.60), +1, cfg)
    assert long_trade.capital == pytest.approx(0.41)
    assert long_trade.pnl == pytest.approx(0.60 - 0.41)
    short_trade = make_trade(cell(0.40, 0.25), -1, cfg)
    assert short_trade.capital == pytest.approx(0.61)
    assert short_trade.pnl == pytest.approx(0.75 - 0.61)


def test_cost_always_reduces_pnl_on_both_sides():
    free = BacktestConfig(cost=0.0)
    charged = BacktestConfig(cost=0.02)
    for direction, exit_price in ((+1, 0.60), (-1, 0.25)):
        assert (
            make_trade(cell(0.40, exit_price), direction, charged).pnl
            < make_trade(cell(0.40, exit_price), direction, free).pnl
        )


def test_entry_drift_shrinks_the_capturable_move_both_ways():
    """Arriving late moves the fill toward the exit: winners pay, losers are refunded."""
    none_ = BacktestConfig(cost=0.0, entry_drift=0.0)
    half = BacktestConfig(cost=0.0, entry_drift=0.5)
    # Correct long into a rise: less edge left to capture.
    assert make_trade(cell(0.40, 0.60), +1, half).pnl < make_trade(cell(0.40, 0.60), +1, none_).pnl
    # Correct short into a fall: same.
    assert make_trade(cell(0.40, 0.25), -1, half).pnl < make_trade(cell(0.40, 0.25), -1, none_).pnl
    # Wrong long into a fall: the loss shrinks too. Drift is not a one-sided fee,
    # it is the move having already happened.
    assert make_trade(cell(0.40, 0.25), +1, half).pnl > make_trade(cell(0.40, 0.25), +1, none_).pnl


def test_full_drift_leaves_only_the_spread():
    cfg = BacktestConfig(cost=0.0, entry_drift=1.0)
    assert make_trade(cell(0.40, 0.60), +1, cfg).pnl == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- selection

def test_zero_signal_is_no_trade_even_at_threshold_zero():
    """A null-routed cell must not become a full-size short."""
    cfg = BacktestConfig(threshold=0.0, cost=0.0)
    cells = [cell(0.50, 0.90, pred=0.50, mid='null'), cell(0.50, 0.90, pred=0.55, mid='live')]
    assert [t.market_id for t in run_strategy(cells, swm_signal, cfg)] == ['live']


def test_unaffordable_fill_is_dropped_not_silently_discounted():
    """side price + spread past 1.0 has no fill; clamping would refund the spread."""
    cfg = BacktestConfig(cost=0.05)
    assert make_trade(cell(0.97, 0.99), +1, cfg) is None       # 0.97 + 0.05 > 1
    assert make_trade(cell(0.02, 0.01), -1, cfg) is None       # NO side is 0.98
    assert make_trade(cell(0.90, 0.99), +1, cfg) is not None   # 0.90 + 0.05 fits


def test_max_drawdown_uses_the_pnl_path_not_the_normalised_curve():
    cfg = BacktestConfig(cost=0.0, sizing='fixed_shares')
    # +0.10, then -0.30, then +0.10 -> trough is -0.20 against 1.50 of capital.
    trades = [
        make_trade(cell(0.50, 0.60, t=1, mid='a'), +1, cfg),
        make_trade(cell(0.50, 0.20, t=2, mid='b'), +1, cfg),
        make_trade(cell(0.50, 0.60, t=3, mid='c'), +1, cfg),
    ]
    m = trade_metrics(trades, bootstrap=False)
    assert m['max_drawdown'] == pytest.approx(-0.30 / 1.50, rel=1e-9)


def test_threshold_gates_on_the_signal_not_the_outcome():
    cfg = BacktestConfig(threshold=0.05, cost=0.0)
    cells = [cell(0.50, 0.90, pred=0.52), cell(0.50, 0.90, pred=0.58, mid='m2')]
    trades = run_strategy(cells, swm_signal, cfg)
    assert [t.market_id for t in trades] == ['m2']


def test_max_positions_per_time_keeps_the_strongest_signals():
    cfg = BacktestConfig(threshold=0.05, cost=0.0, max_positions_per_time=1)
    cells = [
        cell(0.50, 0.90, pred=0.58, mid='weak'),
        cell(0.50, 0.90, pred=0.80, mid='strong'),
    ]
    assert [t.market_id for t in run_strategy(cells, swm_signal, cfg)] == ['strong']


def test_repeated_offers_of_one_position_fill_once():
    """A daily-quoted market re-offered at many decision times is one trade."""
    def offer(t):
        c = cell(0.50, 0.90, pred=0.80, mid='m1', t=t)
        c.extra = {'entry_t': 100, 'settle_t': 900}   # same quote pair each time
        return c

    cfg = BacktestConfig(threshold=0.05, cost=0.0)
    offers = [offer(t) for t in (10, 20, 30, 40)]
    trades = run_strategy(offers, swm_signal, cfg)
    assert len(trades) == 1
    assert trades[0].t == 10  # the first decision time that fired

    # and the raw behaviour is still available for comparison
    loose = BacktestConfig(threshold=0.05, cost=0.0, dedupe_positions=False)
    assert len(run_strategy(offers, swm_signal, loose)) == 4


def test_dedupe_does_not_merge_genuinely_distinct_positions():
    cfg = BacktestConfig(threshold=0.05, cost=0.0)
    a = cell(0.50, 0.90, pred=0.80, mid='m1', t=10)
    a.extra = {'entry_t': 100, 'settle_t': 900}
    b = cell(0.50, 0.90, pred=0.80, mid='m1', t=20)
    b.extra = {'entry_t': 500, 'settle_t': 1300}   # the quote moved on
    assert len(run_strategy([a, b], swm_signal, cfg)) == 2


def test_matched_baselines_run_on_exactly_the_model_cells():
    cfg = BacktestConfig(threshold=0.05, cost=0.0)
    cells = [
        cell(0.50, 0.90, pred=0.80, mid='traded'),
        cell(0.50, 0.10, pred=0.50, mid='skipped'),
    ]
    trades = run_strategy(cells, swm_signal, cfg)
    matched = matched_baselines(cells, trades, cfg)
    assert matched['always_yes']['n_trades'] == len(trades) == 1
    # always-YES on the traded cell only: 0.50 -> 0.90 is +0.40 on 0.50 capital.
    assert matched['always_yes']['roi'] == pytest.approx(0.8)


# ---------------------------------------------------------------- metrics

def test_roi_is_capital_weighted_not_a_mean_of_returns():
    cfg = BacktestConfig(cost=0.0, sizing='fixed_shares')
    trades = [
        make_trade(cell(0.10, 0.20, mid='cheap'), +1, cfg),
        make_trade(cell(0.90, 0.80, mid='dear', eid='e2'), +1, cfg),
    ]
    m = trade_metrics(trades, bootstrap=False)
    assert m['roi'] == pytest.approx((0.10 - 0.10) / (0.10 + 0.90), abs=1e-12)
    assert m['mean_trade_return'] == pytest.approx((1.0 + (-0.1111111)) / 2, rel=1e-4)


def test_edge_per_share_is_signed_by_direction():
    cfg = BacktestConfig(cost=0.0, sizing='fixed_shares')
    winner_short = make_trade(cell(0.60, 0.40), -1, cfg)
    assert trade_metrics([winner_short], bootstrap=False)['edge_per_share'] == pytest.approx(0.20)


def test_net_edge_per_share_is_gross_minus_the_spread():
    """Both are per-trade averages, so they must differ by exactly the cost."""
    cfg = BacktestConfig(cost=0.01, sizing='fixed_notional')
    trades = [
        make_trade(cell(0.10, 0.20, mid='cheap'), +1, cfg),
        make_trade(cell(0.80, 0.70, mid='dear', eid='e2'), -1, cfg),
    ]
    m = trade_metrics(trades, bootstrap=False)
    assert m['net_edge_per_share'] == pytest.approx(m['edge_per_share'] - 0.01, abs=1e-9)


def test_max_drawdown_is_zero_for_a_monotone_curve():
    cfg = BacktestConfig(cost=0.0)
    trades = [
        make_trade(cell(0.50, 0.60, t=1, mid='a'), +1, cfg),
        make_trade(cell(0.50, 0.60, t=2, mid='b'), +1, cfg),
    ]
    assert trade_metrics(trades, bootstrap=False)['max_drawdown'] == pytest.approx(0.0)


def test_equity_curve_is_ordered_and_ends_at_total_return():
    cfg = BacktestConfig(cost=0.0)
    trades = [
        make_trade(cell(0.50, 0.40, t=20, mid='b'), +1, cfg),
        make_trade(cell(0.50, 0.60, t=10, mid='a'), +1, cfg),
    ]
    curve = equity_curve(trades)
    assert [p['t'] for p in curve] == [10, 20]
    total = sum(t.pnl for t in trades) / sum(t.capital for t in trades)
    assert curve[-1]['equity'] == pytest.approx(1 + total)


# ------------------------------------------------------------- no lookahead

def _records():
    """Two records on one market, 10 days apart, each carrying its own quotes."""
    def rec(target_t, price):
        return {
            'market_id': 'm1', 'event_id': 'e1', 'question': 'Q?', 'description': '',
            'outcome': 'Yes',
            'history': [{'t': target_t - (16 - i) * DAY, 'p': 0.3} for i in range(16)],
            'before': {'t': target_t - 7200, 'p': price - 0.05},
            'target': {'t': target_t, 'p': price},
            'future': [{'t': target_t + 3600, 'p': price + 0.01}],
        }
    return [rec(100 * DAY, 0.5), rec(110 * DAY, 0.9)]


def test_quote_never_returns_a_future_price():
    series = PriceSeries(_records())
    t = 100 * DAY
    observed_at, price = series.quote('m1', t)
    assert observed_at <= t
    # a later record's daily resample also lands on this timestamp; the exact
    # `target` quote is the one that must win.
    assert price == 0.5
    # one second before the target quote, the best available is the 2h-prior one
    assert series.quote('m1', t - 1)[1] == pytest.approx(0.45)


def test_daily_history_is_clipped_by_as_of():
    series = PriceSeries(_records())
    target_t = 110 * DAY
    unclipped = series.daily_history('m1', target_t)
    clipped = series.daily_history('m1', target_t, as_of=100 * DAY)
    assert len(clipped) <= len(unclipped)
    assert all(p['t'] <= target_t - DAY for p in unclipped)
    # nothing observed after the as_of cut may set a history point's price
    assert all(p['p'] != 0.9 for p in clipped)


def test_is_live_rejects_a_resolved_or_stale_market():
    series = PriceSeries(_records())
    assert series.is_live('m1', 100 * DAY - 1)
    assert not series.is_live('m1', 200 * DAY)  # nothing left to settle against
    assert not series.is_live('unknown', 100 * DAY)
