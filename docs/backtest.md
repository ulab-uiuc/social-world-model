# SWM trading backtest

Can `swmbench/swm-wm-jin10-daily-7b` make money if you point it at a real
prediction market?

**Short answer: it has learned something real, and that something is currently
worth less than the bid-ask spread.**

On a direction-neutral set of 2,512 held-out positions the model earns
**+0.53c per share gross**, where buying YES blind and buying NO blind both earn
exactly 0.00c. That signal is not noise — the correlation between the model's
forecast and the part of the move still tradeable at order time is +0.081 with
t = +5.6 over 4,819 positions. But a 1c spread costs more than 0.53c, so the
strategy returns **-0.5% on capital (95% CI -3.3% to +2.8%)**, and perfect
direction on the same positions would have earned +6.02c per share — the
forecast captures about **9%** of the edge that was there.

The headline **+31%** you get by backtesting on the jin10 test set alone is an
artifact: that set contains only detected breakpoints, 70% of which are up
moves, so buying YES blind on the very same positions returns **+43%**.

---

## 1. What this measures, and why it needs care

`swmbench_jin10_dailyhist_en.jsonl` is not a market history. Every one of its
3,134 records exists **because a large 2h price move was detected there**
(`sample_type: breakpoint`). Three consequences drive the whole design:

| Property of the data | Consequence for a backtest |
|---|---|
| Records are selected on large moves | You only ever trade when you already know something big is about to happen. Baselines must be run on the identical cells. |
| 70% of those moves are up | "Always buy YES" is a strong strategy on this set. It is the number to beat, not `0%`. |
| `attributions` were scored with the realised move in hand | Feeding them to the model at trade time leaks the answer. They are reported only as an explicit upper bound. |

## 2. Exact settings

| Setting | Value | Why |
|---|---|---|
| Model | `swmbench/swm-wm-jin10-daily-7b` | full-FT Qwen2.5-7B + regression head |
| Tokenizer | `Qwen/Qwen2.5-7B-Instruct` | checkpoint ships weights only |
| `predict_delta` | `True` | restored from `llm_regressor_config.json` |
| `pooling_method` | `last_token` | restored from checkpoint |
| `max_news` | `8` | restored from checkpoint; grid uses the same |
| `max_seq_length` | `1024` | matches training |
| dtype | bf16 backbone, fp32 head | training ran FSDP bf16 compute over fp32 masters |
| `null_rho0` / `odds_eps` / `odds_temp` | `1.0` / `1e-3` / `1.0` | training defaults |
| Retrieval encoder | `BAAI/bge-small-en-v1.5` | bi-encoder, cosine |
| News window | `[target.t - 2.5h, target.t - 1.5h)` | where 100% of the dataset's headlines fall |
| Decision time | `target.t - 1.5h` | after the last headline is public |
| Entry price | last quote at or before decision time | usually the `target.t - 2h` quote |
| Exit (`move`) | first quote at or after `target.t` | exactly the 2h move on breakpoint cells |
| Exit (`resolution`) | market outcome, 1 or 0 | buy and hold |
| Spread / fee | `0.01` headline; swept `0 / 0.005 / 0.01 / 0.02` | charged in price units on the side bought |
| Entry slippage | `0.0` headline; swept `0 / 0.25 / 0.5` | fraction of the move already in the book |
| Entry threshold | `|edge| >= 0.05`; swept over 8 values | |
| Sizing | `fixed_notional` and `fixed_shares`, both reported | they disagree by more than the result |
| Price floor | `0.02` | quotes clamped to [0.02, 0.98] |
| Position dedup | on | one fill per `(market, entry_t, settle_t)` |
| Confidence intervals | cluster bootstrap on `event_id`, 2,000 resamples | trades are not independent |

## 3. Data, split, and the out-of-sample proof

3,134 records, 920 Polymarket markets, 2025-07-16 to 2026-07-18. Split
chronologically 80/10/10 by `target.t`, matching `scripts/split_temporal.py`.

```
train  2,507   2025-07-16 .. 2026-06-07
valid    313   2026-06-07 .. 2026-06-17
test     314   2026-06-17 .. 2026-07-18   <- the backtest
               139 markets, 84 events, 231 distinct decision times
```

The checkpoint was trained on a split of a **different file**
(`swmbench_jin10_attributed_filtered_en.jsonl`, via `scripts/debias_split.py`),
so an 80% index fraction on this file does not by itself prove the window is
held out. The branch's own `scripts/backtest_build.py` pins the training cutoff
at **2026-05-24**. `backtest_build_grid.py` asserts the test window starts after
it rather than assuming so:

```
out-of-sample check: test starts 2026-06-17, 25 days after the 2026-05-24 train cutoff
```

## 4. Timing: what is knowable when

Every headline attached to a record is published in a one-hour window ending 90
minutes before the move — verified across the whole file, no exceptions.

```
 target.t-24h        target.t-2.5h ── news window ── -1.5h        target.t
      │                      │                    │  │                │
 history[-1]            first headline    last headline │          settlement
 = model's anchor                                  entry quote
                                              (target.t-2h) and
                                               decision_t
```

* **Prompt inputs** — `question`, 16 daily history points, retrieved headlines.
  `PriceSeries.daily_history(..., as_of=decision_t)` clips the series *before*
  sampling, so a later quote cannot change which points are emitted.
* **Entry** — the last quote at or before `decision_t`. This is the weakest
  assumption in the harness: the data has no finer granularity, so we cannot see
  whether the book moved between `-2h` and `-1.5h`. `--entry-drift` prices that
  in.
* **Exit** — a label. It reaches the P&L and never the prompt.

**Prompt parity was verified, not assumed**: all 314 test cells produce a
byte-identical prompt string to what `MultiEventWorldModelDataset` emits from the
original record, along with identical `before_price`, history length and target.

## 5. Replacing the oracle attributions with retrieval

The `attributions` field is a posterior label. `--news retrieval` replaces it
with something a live system could actually compute:

1. Union every record's headline list into a **reconstructed news stream** —
   114,316 de-duplicated English items with timestamps. See the caveat in §6:
   this is not the jin10 wire, it is the union of the one-hour windows around
   detected breakpoints.
2. At each decision, slice the wire to the news window and collapse
   re-publications of the same headline.
3. Score each headline against the market text with the bi-encoder, keep top 8.
4. Map cosine similarity onto the `[0, 0.95]` range the model's odds transform
   expects, using a mapping fit on **train-split records only**:
   `lo` = 95th percentile of unattributed news, `hi` = 95th percentile of oracle
   positives.

Two details matter.

The window is a **superset** of the record's own list (median 73 headlines vs
51), so retrieval must discriminate rather than pick from a pre-narrowed
shortlist. And step 4 calibrates on *absolute* similarity rather than softmaxing
the retrieved set: a softmax would hand the top hit ~0.95 on every cell,
including the thousands of grid cells no headline is about, and the model could
never route to its null option. With calibration it routes **35.8%** of grid
cells to null.

On the train split the encoder separates attributed from unattributed headlines
at **AUC 0.736** — real, but far from oracle. That gap is the honest price of not
knowing which headline mattered, and it costs roughly half the forecast
correlation (0.34 oracle → 0.18 retrieval).

## 6. Two universes

| mode | cells | distinct positions | what it asks |
|---|---|---|---|
| `breakpoint` | 314 | 314 | given a big move is coming, can it call the direction? |
| `grid` | 20,856 | 4,819 | at each of 231 decision times, score **every** live market — including deciding *not* to trade |

The grid is 231 decision times x 52–134 live markets each (median 71). Only
**314 cells (1.5%)** carry a breakpoint; the other 20,542 are quiet markets that
a live system must decline. Example: at 2026-06-20 19:00 UTC the same Iran
headlines are scored against 134 markets simultaneously, one of which is
"Will Solana dip to $60 in June?".

### The timeline is breakpoint-driven, not news-driven

The 231 decision times are the `target.t` of the test records — the moments a
breakpoint was detected — not the moments news arrived. Over the 30-day test
window that is 7.7 decisions per day, with a median gap of 1h between
consecutive decisions but a p90 of 9h and a maximum of **73h**.

This is a property of the source file, not a choice. The reconstructed news
stream is assembled from per-record news lists, so it only exists near
breakpoints:

| Test-window hours | Count | Headlines in the stream |
|---|---|---|
| Covered by some decision window | 261 | 11,094 |
| Not covered | 467 | **0** |

**467 of 728 hours contain no news at all.** Iterating the stream
chronologically instead of iterating decision times would therefore produce the
identical 231 decision points — there is nothing to iterate over in between.

So the backtest answers "when news lands, does the model pick the right market
and the right direction?" It cannot answer "run this 24/7 for a year and does
it profit", because for 64% of the wall-clock there is no input. Doing that
needs the raw jin10 archive over the window plus a price series that can settle
at arbitrary times — `swmbench_2h_1year_with_jin10_news.jsonl` and
`polymarket_*_series.jsonl` in the same HF repo have both, but that file's news
is untranslated Chinese while this checkpoint was trained on English.

## 7. One position, filled once

A quiet market is quoted once a day, so every decision time inside one
inter-quote gap resolves to the **same** entry quote and the **same** settlement
quote. Those cells are one position offered repeatedly, not separate
opportunities.

```
market 678748 "Will Israel strike 5 countries in 2026?"
  231 grid cells  ->  55 distinct positions
  06-17 20:00  entry 0.301 (quoted 06-16 23:00)  settle 0.272   new
  06-17 21:00  entry 0.301 (quoted 06-16 23:00)  settle 0.272   same trade
  06-17 22:00  entry 0.301 (quoted 06-16 23:00)  settle 0.272   same trade
```

`run_strategy` therefore fills each `(market_id, entry_t, settle_t)` once, at the
first decision time whose signal fires, for the model and every baseline alike
(`BacktestConfig.dedupe_positions`, on by default).

This is not a detail. Counting the repeats inflates the trade count fourfold,
shrinks the naive t-statistic by about its square root, and re-weights ROI toward
the quiet long-hold markets that repeat most — enough to **flip the sign** of the
grid result (-3.08% duplicated vs -0.53% deduplicated).

## 8. Accounting and baselines

A YES share costs `p` and pays 1 on YES; a NO share costs `1-p`. Exiting at quote
`q` pays `q` per YES share. So per share, P&L is `q - p` long and `p - q` short,
against capital of `p` and `1 - p`.

Both sizings are reported because on this data they disagree by more than the
entire result:

* `fixed_notional` — $1 of capital per trade. What "收益率" usually means, and
  where a dollar buys 50 shares at 2c and 1 share at 95c.
* `fixed_shares` — equal share count. Strips the leverage out.

Every trivial rule is run **twice**: over the full candidate set, and matched to
the exact cells the model chose. Comparing a selective model against a baseline
that trades everything confounds cell selection with directional skill.

| baseline | signal |
|---|---|
| `always_yes` / `always_no` | constant |
| `random` | seeded coin flip |
| `momentum` / `contrarian` | `entry - prev_price` and its negation |
| `perfect_direction` | `exit - entry` — a per-cell ceiling, not an ROI ceiling |
| `swm` | `pred_price - entry` (forecast vs the book) |
| `swm_delta` | `pred_price - prev_price` (the model's own delta) |

---

# Results

All numbers: retrieval news, `move` exit, 1c spread, no slippage,
`|edge| >= 0.05`, fixed-notional sizing unless stated.

## 9. Breakpoint universe — 314 held-out cells

### Forecasting

| News source | Pearson | Direction | RMSE | No-change RMSE | Skill | Forecast σ |
|---|---|---|---|---|---|---|
| Retrieval (live) | 0.180 | 65.2% | 0.1981 | 0.2038 | +0.056 | 0.054 |
| Oracle attributions (leaky) | 0.340 | 71.1% | 0.1872 | 0.2038 | +0.156 | 0.070 |

Both beat the no-change baseline, and the model is properly shrunk — it forecasts
a 0.054 standard deviation against a realised 0.191, which is what a calibrated
regressor on a noisy target should do.

### Trading

| Strategy | Trades | Return on capital | 95% CI | Gross/share | Net/share | Hit rate |
|---|---|---|---|---|---|---|
| **SWM** | 196 | **+31.2%** | +9.0 to +63.7% | +0.90c | -0.10c | 50.5% |
| SWM delta | 115 | +60.6% | +24.6 to +111.0% | +5.97c | +4.97c | 73.0% |
| Momentum | 155 | +5.7% | -1.5 to +12.6% | +1.69c | +0.69c | 60.6% |
| Contrarian | 155 | +2.7% | -9.3 to +22.9% | -1.69c | -2.69c | 39.4% |
| **Always YES** | 314 | **+36.8%** | +23.0 to +56.3% | +4.91c | +3.91c | 70.1% |
| Always NO | 314 | -7.1% | -11.2 to -2.3% | -4.91c | -5.91c | 29.9% |
| Coin flip | 314 | +20.8% | +8.7 to +38.2% | +0.67c | -0.33c | 52.9% |
| Perfect direction | 314 | +53.4% | +39.8 to +73.3% | +11.16c | +10.16c | 100% |

### The matched comparison

On **exactly the 196 cells the model chose to trade**:

| On SWM's own cells | Return | Gross/share | Hit rate |
|---|---|---|---|
| **SWM** | **+31.2%** | +0.90c | 50.5% |
| Always YES | **+43.0%** | +6.02c | 71.4% |
| Always NO | -10.4% | -6.02c | 28.6% |
| Coin flip | +24.8% | +1.02c | 54.6% |
| Perfect direction | +59.1% | +12.22c | 100% |

**The model loses to buying YES blind on its own picks**, and its hit rate
(50.5%) is worse than the majority class (70%). Holding to resolution is worse
still: +18.5% against +51.3% for matched always-YES.

### Where the +31% comes from

| Entry price | Cells | SWM trades | SWM return | Always-YES return | SWM gross/share |
|---|---|---|---|---|---|
| 0.02–0.10 | 43 | 21 | **+300.0%** | +198.4% | +16.89c |
| 0.10–0.25 | 87 | 47 | +9.7% | +23.8% | +0.95c |
| 0.25–0.50 | 105 | 71 | -2.6% | +7.8% | -1.25c |
| 0.50–0.75 | 59 | 44 | -10.0% | +5.0% | -1.78c |
| 0.75–0.98 | 20 | 13 | -2.1% | -7.6% | -4.25c |

Every cent of positive return sits in **21 trades at 2–10c**. Re-run the
identical trades with equal share counts and the result inverts:

| Equal-share sizing | SWM | SWM delta | Always YES | Always NO |
|---|---|---|---|---|
| Return on capital | **-0.2%** | +13.0% | +10.8% | -9.0% |

## 10. Full grid — 20,856 cells, 4,819 distinct positions

The universe is now balanced: mean move **-0.39c**, **38.4% up**, mean absolute
move 3.95c, mean entry price 0.292. The free money is gone.

| Strategy | Trades | Return on capital | 95% CI | Gross/share | Net/share | Hit rate | Long |
|---|---|---|---|---|---|---|---|
| **SWM** | 2,512 | **-0.5%** | -3.3 to +2.8% | +0.53c | -0.47c | 34.4% | 57% |
| SWM delta | 1,405 | -0.6% | -5.2 to +4.7% | +0.59c | -0.41c | 31.8% | 70% |
| Momentum | 1,570 | -1.6% | -3.8 to +0.4% | -0.20c | -1.20c | 40.8% | 54% |
| Contrarian | 1,570 | -1.6% | -3.8 to +1.1% | +0.20c | -0.80c | 39.2% | 46% |
| Always YES | 4,819 | -3.1% | -5.3 to -0.8% | +0.09c | -0.91c | 29.0% | 100% |
| Always NO | 4,819 | -1.0% | -1.8 to -0.1% | -0.09c | -1.09c | 31.1% | 0% |
| Coin flip | 4,819 | -1.6% | -3.0 to +0.1% | +0.08c | -0.92c | 30.4% | 50% |
| Perfect direction | 1,471 | +41.6% | +36.7 to +48.1% | +11.74c | +10.74c | 100% | 51% |

### The matched comparison, and the cleanest evidence of learning

On the **2,512 positions the model chose**:

| On SWM's own positions | Return | Gross/share |
|---|---|---|
| Always YES | -2.7% | **-0.00c** |
| Always NO | -1.1% | **+0.00c** |
| Coin flip | -0.7% | +0.24c |
| **SWM** | **-0.5%** | **+0.53c** |
| Perfect direction | +17.4% | +6.02c |

The model's selection is **directionally neutral** — buying YES blind and buying
NO blind on those positions both earn exactly 0.00c per share, so there is no
drift to ride. The model earns +0.53c on the same positions. That is genuine
directional information, and it is **9% of the 6.02c that was available**.

### Where it comes from

| Cell type | Cells | SWM trades | SWM return | Always-YES return | SWM gross/share |
|---|---|---|---|---|---|
| Breakpoint | 314 | 196 | +31.2% | +36.8% | +0.90c |
| Quiet | 20,542 | 2,378 | -3.0% | -5.2% | +0.46c |

Equal-share sizing: SWM -1.1%, always-YES -2.8%, always-NO -1.6%.
Resolution exit: SWM -5.1% against +9.8% for matched always-YES.

## 11. What the model actually learned

| Universe | Model RMSE | **Live quote RMSE** | 24h anchor RMSE | Target already public | Corr w/ trained target | Corr w/ tradeable part |
|---|---|---|---|---|---|---|
| Breakpoint (314) | 0.1981 | **0.1432** | 0.2038 | 71% | 0.180 | **0.242** (t=+4.4) |
| Full grid (4,819) | 0.1238 | **0.0826** | 0.1221 | 74% | 0.091 | **0.081** (t=+5.6) |

Read the first two columns together. **As an estimate of where the price settles,
the model is worse than simply reading the current quote** — 0.124 against 0.083
on the grid — and no better than the 24-hour-old price it anchors on.

That is less a failure of training than a statement about the target. The model
is graded on `target.p - history[-1].p`, a 24h move, but by decision time the
book has already walked most of it:

```
grid:  24h move |0.0728|  =  already in the book |0.0512|  +  still tradeable |0.0456|
       corr(24h move, already-in-book) = +0.74
```

**Three quarters of what the objective rewards is re-deriving public
information.** The last column is the part that can become a position, and it is
positive and significant — but small.

## 12. Sensitivity

Cost sweep at `|edge| >= 0.05`, no slippage:

| Spread | Breakpoint return | Grid return | Grid net/share |
|---|---|---|---|
| 0.0c | +45.7% | **+6.9%** | +0.53c |
| 0.5c | +37.5% | **+2.8%** | +0.03c |
| 1.0c | +31.2% | **-0.5%** | -0.47c |
| 2.0c | +21.5% | -5.9% | -1.46c |

Break-even sits at a spread of roughly **0.7c**. Real Polymarket books on these
markets are wider.

Entry slippage at 1c spread:

| Move already in the book | Breakpoint | Grid |
|---|---|---|
| 0% | +31.2% | -0.5% |
| 25% | +5.1% | -5.0% |
| 50% | -1.4% | -6.6% |

Since the signal is public wire copy that every participant reads at the same
moment, some leakage is the realistic case, not the pessimistic one.

The threshold sweep spans 8 values x 4 costs x 3 slippages = 96 configurations on
one test set. Read it as a robustness check, not a menu.

---

## 13. How to run

```bash
DATA=swmbench_jin10_dailyhist_en.jsonl          # swmbench/swmbench on HF
CKPT=swm-wm-jin10-daily-7b                      # weights only

# 1. build the cells the model is asked to score
python scripts/backtest_build_grid.py --data $DATA --mode breakpoint --news retrieval \
    --out results/backtest/grid_breakpoint_retrieval.jsonl
python scripts/backtest_build_grid.py --data $DATA --mode grid --news retrieval \
    --out results/backtest/grid_full_retrieval.jsonl

# 2. score them; shard across GPUs with --num-shards / --shard-idx
python scripts/backtest_predict.py --grid <grid>.jsonl --model-path $CKPT \
    --model-name Qwen/Qwen2.5-7B-Instruct --out <preds>.jsonl --batch-size 8

# 3. P&L, baselines, sweeps  (add --sizing fixed_shares for the second view)
python scripts/backtest_report.py --preds retrieval=<preds>.jsonl \
    --out results/backtest/report.json --threshold 0.05 --cost 0.01

# 4. HTML report
python scripts/backtest_html.py --notional results/backtest/report_breakpoint.json \
    --shares results/backtest/report_breakpoint_shares.json \
    --grid results/backtest/report_grid.json \
    --grid-shares results/backtest/report_grid_shares.json \
    --out results/backtest/report.html
```

Layout:

```
swm/backtest/
  universe.py    price-series reconstruction, no-lookahead quotes, the decision grid
  newsstream.py  the global jin10 wire, sliced by publication time
  retrieval.py   news <-> market relevance, train-calibrated
  engine.py      trading rules, cost model, baselines, P&L statistics
scripts/
  backtest_build_grid.py  backtest_predict.py  backtest_report.py  backtest_html.py
tests/test_backtest_engine.py
```

## 14. Verification

* **23 unit tests** covering hand-computed long/short P&L, that costs and
  slippage move in the right direction on both sides, that a zero signal is not
  a short, that unaffordable fills are dropped rather than silently discounted,
  the position-dedup rule, and the no-lookahead guarantees of `PriceSeries`.
* **Prompt parity**: 314/314 cells reproduce the training pipeline's prompt
  byte-for-byte.
* **Out-of-sample assertion** against the pinned 2026-05-24 training cutoff.
* **Adversarial audit** across four lenses (lookahead, accounting, statistics,
  train/inference parity) with refute-by-default verification. 25 findings
  raised, 16 refuted, 2 confirmed and fixed:
  * position duplication in grid mode (flipped the sign of the grid result);
  * `_clamp` deleting the spread on any side priced above 0.97.

## 15. Limitations

* **314 breakpoint cells over one month** is a small sample; intervals are wide.
* **Entry uses a quote up to 30 minutes stale.** The slippage sweep stands in for
  measuring it properly.
* **No order book.** Fills are assumed at the quote for the full stake, with cost
  as a flat spread. Depth at 2–10c is thin, and that is exactly where the
  fixed-notional P&L comes from.
* **Grid membership is itself selected on future information.** A market appears
  at a decision time only because it was quoted near it, and it was quoted
  because a breakpoint happened somewhere in its record. The grid over-samples
  markets that turned out to be volatile. It does not bias *direction*, and every
  baseline is scored on the same cells, but this is not the market as a whole.
* **`resolution` exit only sees markets that have resolved**, a survivorship
  filter on the long-hold numbers. The `move` exit has none.
* **Heterogeneous holding periods in grid mode**: quiet markets are quoted daily,
  so they hold ~24h against the breakpoint cells' 2h. `strata.by_hold` exposes
  this.
* **The news stream is not a wire.** It exists only in the 261 of 728
  test-window hours that sit near a breakpoint (§6). Every number here is
  conditional on news having landed; none of it measures idle time.

## 16. What would make it tradeable

1. **Train on the residual, not the 24h move.** Target should be settlement price
   minus the quote at decision time. As set up, three quarters of the gradient
   goes into predicting what the book already shows.
2. **Show the model the current quote.** Its history stops 24 hours before the
   prediction, so it forecasts without the single most informative number
   available — the live-quote column in §11 is the baseline it is being denied.
3. **Score against the residual too.** Pearson against the 24h delta reads 0.091
   on the grid while the tradeable correlation is 0.081; only the second can be
   turned into a position.
4. **Evaluate trading on a direction-balanced universe.** On the breakpoint set
   "always say up" scores 70% direction accuracy. `scripts/debias_split.py`
   already balances direction for the forecasting metrics; the trading evaluation
   needs the same treatment.
