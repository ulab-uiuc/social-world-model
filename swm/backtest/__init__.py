"""Walk-forward backtest of the SWM world model on real prediction-market prices.

The pieces:
  universe    -- price-series reconstruction, no-lookahead price lookup, the
                 (market x decision-time) grid the model is asked to score
  newsstream  -- the global jin10 news stream, sliced by publication time
  retrieval   -- news <-> market relevance (replaces the oracle `attributions`)
  engine      -- trading rules, costs, baselines, P&L / equity metrics
"""

from .engine import (
    BacktestConfig,
    evaluate_strategies,
    matched_baselines,
    trade_metrics,
)
from .newsstream import NewsStream
from .retrieval import Calibration, EmbeddingRetriever, calibrate
from .universe import PriceSeries, temporal_split

__all__ = [
    'BacktestConfig',
    'Calibration',
    'EmbeddingRetriever',
    'NewsStream',
    'PriceSeries',
    'calibrate',
    'evaluate_strategies',
    'matched_baselines',
    'temporal_split',
    'trade_metrics',
]
