"""Market universe: price series, no-lookahead lookups, and the decision grid.

`swmbench_jin10_dailyhist_en.jsonl` stores one record per detected 2h breakpoint,
but every record also carries a slice of its market's price path (16 daily
history points, a `before` quote 2h pre-move, the `target` quote at the move,
and up to 16 `future` points). Unioning those slices across every record of a
market reconstructs a usable price series for that market -- which is what lets
us quote an entry price at an arbitrary decision time and settle a position
later, including for markets that had no breakpoint at that time.
"""

import bisect
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DAY = 86400
HISTORY_POINTS = 16  # must match training (scripts/build_jin10_dailyhist.py)
HISTORY_STEP = DAY
HISTORY_END_OFFSET = DAY  # last history point is one day before the target
MIN_HISTORY_POINTS = 8


def load_records(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def record_time(record: dict) -> int:
    """The prediction timestamp of a record (`target.t`)."""
    return int((record.get('target') or {}).get('t') or record.get('move_hour_t') or 0)


def temporal_split(
    records: Sequence[dict],
    train_frac: float = 0.80,
    valid_frac: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Chronological 80/10/10 split, mirroring scripts/split_temporal.py.

    The checkpoint under test was trained on the earliest 80% by `target.t`, so
    the backtest must run on the final 10% to stay out of sample.
    """
    ordered = sorted(records, key=record_time)
    n = len(ordered)
    i_train = int(n * train_frac)
    i_valid = int(n * (train_frac + valid_frac))
    return ordered[:i_train], ordered[i_train:i_valid], ordered[i_valid:]


@dataclass
class MarketMeta:
    market_id: str
    event_id: str
    question: str
    description: str
    categories: list[str]
    outcome: float | None  # 1.0 / 0.0 once resolved, else None

    @property
    def text(self) -> str:
        desc = (self.description or '')[:400]
        return f'{self.question}\n{desc}'.strip()


def _outcome_to_price(raw) -> float | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token == 'yes':
        return 1.0
    if token == 'no':
        return 0.0
    return None


class PriceSeries:
    """Per-market observed price path, assembled from every record's quotes."""

    def __init__(self, records: Iterable[dict]):
        # Two tiers, because they are not equally accurate at the same
        # timestamp. `before` / `target` / `future` are quotes read straight off
        # the book at that instant; a `history` point is the daily resample, and
        # carries the last price observed at or before its stamp, which can be
        # older. Where both exist for one timestamp the exact quote wins.
        exact: dict[str, dict[int, float]] = {}
        sampled: dict[str, dict[int, float]] = {}
        meta: dict[str, MarketMeta] = {}
        for record in records:
            mid = str(record.get('market_id'))
            coarse = sampled.setdefault(mid, {})
            fine = exact.setdefault(mid, {})
            for point in record.get('history') or []:
                coarse[int(point['t'])] = float(point['p'])
            for key in ('before', 'target'):
                quote = record.get(key)
                if quote and quote.get('t') is not None:
                    fine[int(quote['t'])] = float(quote['p'])
            for point in record.get('future') or []:
                fine[int(point['t'])] = float(point['p'])
            if mid not in meta:
                meta[mid] = MarketMeta(
                    market_id=mid,
                    event_id=str(record.get('event_id') or mid),
                    question=record.get('question') or '',
                    description=record.get('description') or '',
                    categories=list(record.get('categories') or []),
                    outcome=_outcome_to_price(record.get('outcome')),
                )
        merged = {
            mid: {**sampled.get(mid, {}), **exact.get(mid, {})}
            for mid in set(sampled) | set(exact)
        }
        self.meta = meta
        self._ts: dict[str, list[int]] = {}
        self._ps: dict[str, list[float]] = {}
        for mid, bucket in merged.items():
            times = sorted(bucket)
            self._ts[mid] = times
            self._ps[mid] = [bucket[t] for t in times]

    def __contains__(self, market_id: str) -> bool:
        return market_id in self._ts

    @property
    def market_ids(self) -> list[str]:
        return list(self._ts)

    def span(self, market_id: str) -> tuple[int, int] | None:
        times = self._ts.get(market_id)
        return (times[0], times[-1]) if times else None

    def quote(self, market_id: str, t: int) -> tuple[int, float] | None:
        """Last price observed at or before `t` -- (observed_at, price).

        Strictly backward-looking: this is the quote a trader could actually see
        at time `t`, and how stale it is.
        """
        times = self._ts.get(market_id)
        if not times:
            return None
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return None
        return times[i], self._ps[market_id][i]

    def price_at(
        self, market_id: str, t: int, max_staleness: int | None = None
    ) -> float | None:
        found = self.quote(market_id, t)
        if found is None:
            return None
        observed_at, price = found
        if max_staleness is not None and t - observed_at > max_staleness:
            return None
        return price

    def forward_quote(
        self, market_id: str, t: int, max_gap: int | None = None
    ) -> tuple[int, float] | None:
        """First price observed at or after `t` -- used only to settle a trade."""
        times = self._ts.get(market_id)
        if not times:
            return None
        i = bisect.bisect_left(times, t)
        if i >= len(times):
            return None
        if max_gap is not None and times[i] - t > max_gap:
            return None
        return times[i], self._ps[market_id][i]

    def daily_history(
        self,
        market_id: str,
        target_t: int,
        as_of: int | None = None,
    ) -> list[dict[str, float]]:
        """16 daily points ending one day before `target_t`.

        Reproduces scripts/build_jin10_dailyhist.py exactly so the prompt the
        model sees at backtest time matches the one it was trained on. `as_of`
        clips the underlying series to what was observable at decision time, so
        a market's later quotes can never influence which points get emitted.
        """
        times = self._ts.get(market_id)
        if not times:
            return []
        prices = self._ps[market_id]
        if as_of is not None:
            cut = bisect.bisect_right(times, as_of)
            times, prices = times[:cut], prices[:cut]
            if not times:
                return []
        end = target_t - HISTORY_END_OFFSET
        out: list[dict[str, float]] = []
        for i in range(HISTORY_POINTS - 1, -1, -1):
            t = end - i * HISTORY_STEP
            if t < times[0] or t > times[-1] + HISTORY_STEP:
                continue
            j = bisect.bisect_right(times, t) - 1
            if j >= 0:
                out.append({'t': t, 'p': prices[j]})
        return out

    def is_live(
        self,
        market_id: str,
        t: int,
        max_staleness: int = 3 * DAY,
        require_future: bool = True,
    ) -> bool:
        """Tradeable at `t`: a recent enough quote, and (for settlement) a later one.

        `require_future` only gates whether we can *score* the trade; it uses no
        information the model sees, just whether the data can price an exit.
        """
        found = self.quote(market_id, t)
        if found is None or t - found[0] > max_staleness:
            return False
        if found[1] <= 0.0 or found[1] >= 1.0:
            return False  # already effectively resolved -- nothing to trade
        return not (require_future and self.forward_quote(market_id, t) is None)


def build_daily_record(
    series: PriceSeries,
    market_id: str,
    target_t: int,
    news: list[dict],
    attributions: list[dict],
    decision_t: int,
) -> dict | None:
    """A world-model-ready record for one (market, decision-time) grid cell."""
    history = series.daily_history(market_id, target_t, as_of=decision_t)
    if len(history) < MIN_HISTORY_POINTS:
        return None
    meta = series.meta.get(market_id)
    if meta is None:
        return None
    settle = series.forward_quote(market_id, target_t)
    if settle is None:
        return None
    return {
        'market_id': market_id,
        'event_id': meta.event_id,
        'question': meta.question,
        'description': meta.description,
        'categories': meta.categories,
        'history': history,
        'news': news,
        'attributions': attributions,
        # `target.p` is the settlement quote. It is a label: it reaches the
        # P&L, never the prompt (the prompt only renders `target.t` as a date).
        'target': {'t': target_t, 'p': settle[1]},
    }
