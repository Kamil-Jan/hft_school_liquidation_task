# liqsignal — liquidation-driven maker-trade filter

Research code for the task in [`description.md`](description.md): filter Binance
maker trades using trade / BBO / liquidation data (Binance + Bybit) so that the
*kept* trades have a better markout than the unfiltered baseline, subject to a
$500k/day kept-turnover floor. Universe: `perp:btcusdt`, `perp:ethusdt`; 90 days
(2025-12-01 → 2026-02-28 UTC).

## Layout

```
src/liqsignal/          installable package (the reusable core)
  config.py             paths, universe, frozen task-spec constants
  splits.py             train/validation assignment (NumPy + Polars, kept in sync)
  io.py                 data access: lazy scans, materialised arrays, batched trade iterator
  markout.py            spec-critical maker-PnL math (forward-filled mid, markout in bps)
  scoring.py            Score, PnL_all/kept/filtered, turnover constraint (ScoreResult)
  features.py           feature engineering (FeatureContext + compute_features) + panel assembly
  analysis.py           conditional-markout helpers + score thresholding (expected-value, Score-max sweep)
  model.py              per-tau sample-weighted markout model (HistGBR) + persistence
  baselines.py          full-data PnL_all + turnover/day
  signal.py             submission entry point (model + threshold; keep-all fallback; filter_fn hook)
  report.py             results report: metrics tables + plots

scripts/                thin runners (argparse -> package -> artifacts)
  compute_baselines.py  -> artifacts/baselines.parquet
  build_panel.py        -> artifacts/panel_<sym>.parquet
  run_study.py          conditional-markout study + single-feature filter sweep (reads panels)
  train_model.py        fit per-tau models + thresholds, score panels, write report
  eda/                  exploration producers (aggregates, event studies, notebook build)

notebooks/01_exploration.ipynb   executed EDA narrative (11 charts), reads artifacts
tests/                  pytest unit tests (markout / scoring / splits / features / thresholds / signal)
artifacts/              computed outputs incl. models + report/ (gitignored)
data/                   raw parquet, as shipped (gitignored)
```

Design intent: the **spec-critical math lives in small, pure, unit-tested modules**
(`markout`, `scoring`, `splits`) so the grading logic can't drift; features are pure
functions evaluated through a reused `FeatureContext` (so the same code serves panel
building and the chunked submission path); scripts/notebooks only orchestrate.

## Data — where to put it

The loaders read parquet from a data directory (default `data/`, resolved by
`config.dataset_path`). Place the **train** data in `data/` and any held-out **test**
data in a parallel `data_test/` with the *same* internal layout — both are gitignored.
File names are exact (note: Bybit liquidations have **no** `perp_` prefix):

```
data/                                          <- TRAIN (the shipped 90 days)
  binance_trades/        perp_btcusdt.parquet  perp_ethusdt.parquet
  binance_booktickers/   perp_btcusdt.parquet  perp_ethusdt.parquet
  binance_liquidations/  perp_btcusdt.parquet  perp_ethusdt.parquet
  bybit_liquidations/    btcusdt.parquet       ethusdt.parquet
data_test/                                     <- TEST (same layout; reviewer-supplied)
  binance_trades/ ...  binance_booktickers/ ...  binance_liquidations/ ...  bybit_liquidations/ ...
```

Schemas (timestamps are **int64 microseconds, UTC** throughout):

| Source | Columns |
|---|---|
| trades | `timestamp, ticker, side, price, amount` |
| bbo (book tickers) | `timestamp, ticker, bid_price, bid_amount, ask_price, ask_amount` |
| liquidations (both venues) | `timestamp, ticker, side, price, amount` |

`side` means different things per table: in **trades** it is the *taker* side
(buy = lifted ask); in **liquidations** it is the *liquidation-order* side
(buy = forced buy = upward pressure). Bybit's feed isn't time-sorted and is shifted
`+200 ms` before any Binance comparison — `io.liquidations_from_frame` handles both,
so raw files need no preprocessing.

To point the loaders at a different directory, set `LIQSIGNAL_DATA_DIR` (or use
`make ... DATA_DIR=data_test`, which exports it for you).

## Setup

```bash
make venv               # python3 -m venv .venv
make install            # pip install -e ".[dev,notebook]"  (editable)
make test               # run unit tests
```

Or without make: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,notebook]"`.

## Pipelines

```bash
make baselines   # full-data PnL_all + turnover/day per symbol/split/tau   (~4 min)
make panel       # sampled feature panels with markout + features          (~20 s)
make study       # conditional markout + train→val single-feature sweep
make train       # fit per-tau models + thresholds, write report           (~3 min)
make eda         # EDA precompute + (re)build & execute the notebook
make evaluate    # run signal() on a data dir, report Score/turnover per (sym,τ)
```

`make train` writes the trained models (`artifacts/model_<tau>.joblib`, each carrying
its fitted Score-maximising threshold) and a results report
(`artifacts/report/report.md` + PNG figures).

## For reviewers — evaluating on test data

`make evaluate` runs the submission `signal()` on a data directory, computes the spec
maker markout, and prints — per symbol and τ — `PnL_all`, `PnL_kept`,
**Score = PnL_kept − PnL_all**, the kept turnover/day, and whether the $500k/day
constraint holds. Higher Score is better.

```bash
# 1. install + put the TRAIN data in data/  (see "Data — where to put it")
make install

# 2a. reproduce the models from scratch (fits models + thresholds, ~4 min total)
make panel && make train

#     ...or 2b. skip if trained models already sit in artifacts/model_<tau>.joblib

# 3. put the TEST data in data_test/ (same layout) and score it
make evaluate DATA_DIR=data_test
```

Example output (per row): `Score=+2.750  PnL_kept=+3.10  PnL_all=+0.35  keep=8.4%
keptTurn/day=1,200,000,000  OK`. A full metrics table is also printed and can be
saved with `scripts/evaluate.py --out artifacts/test_metrics.parquet`.

> **Memory:** a full 90-day symbol won't fit in 16 GB read whole. Evaluate a bounded
> test window, or pass `--batch-size N` (e.g. `20000000`) to score the trades in
> memory-bounded chunks: `LIQSIGNAL_DATA_DIR=data_test .venv/bin/python
> scripts/evaluate.py --batch-size 20000000`.

Programmatic use:

```python
from liqsignal import io, markout, scoring
book = io.load_book_top("btc")
# build pnl, weights, filter f (1=drop, 0=keep) ...
result = scoring.evaluate_filter(pnl, w, f, n_days=62)   # -> ScoreResult
```

The submission entry point is `liqsignal.signal.signal(trades, bbo, liq_binance,
liq_bybit)` → `{30,120,300: 0/1 array}` (1 = filter out, 0 = keep). If trained
models are present it computes features on the passed frames (in memory-bounded
batches), predicts markout, and filters trades below the **fitted Score-maximising
threshold** persisted with each model (falling back to the expected-value cutoff,
predicted markout < 0, only for models saved before threshold persistence);
otherwise it falls back to keep-all. A `filter_fn` hook overrides the decision entirely.

## Findings so far

**Data (see notebook):** clean (no crossed books / NaNs), timestamps are int64
microseconds UTC, spreads ~1 tick (rebate dominates), trades ~50/50 while
liquidations skew sell-side. Liquidations mark a local extreme → tiny continuation
→ multi-minute **mean-reversion** of the Binance mid; **Bybit predicts that
reversion ~10× more strongly than Binance's own**, surviving the +200 ms delay.

**Signal pipeline:**
* Baseline making is ~break-even and regime-dependent (BTC validation negative).
* **The turnover constraint barely binds** — clipped turnover ≈ $11–15 B/day vs the
  $500k floor (~25,000× headroom) — so the problem is essentially prediction quality.
* The combined per-τ model (45 features, sample-weighted by `w_i`) + a
  **Score-maximising threshold** beats the previous single-feature keep-10% rule on
  **validation** in every cell (e.g. ETH τ=30: 3.07 vs 1.94 bps; BTC τ=30: 1.71 vs
  0.41), and beats the expected-value rule. All operating points clear the turnover
  floor by ~100×.
* Top features (permutation importance): **Bybit cascade size** (`bybit_liqabs_300s`),
  **time-of-day** (`hour`), the **taker×Bybit-liq-pressure interaction**
  (`bybit_liqalign_300s`), and **volatility amplitude** (`ampl_300s`) — confirming the
  cross-exchange-liquidation thesis (Bybit > Binance) and the regime dependence.

See `artifacts/report/report.md` (run `make panel && make train`) for the full
metrics tables and figures (threshold curves, calibration, per-month stability,
feature importance).

**Next:** tape-derived flow features (TFI/OFI, trade intensity, VPIN — need a 1s
trade-flow grid), purged/embargoed CV, probability calibration, and deeper regime
stress-testing. See [`.claude/docs/roadmap.md`](.claude/docs/roadmap.md) for the full,
prioritized list of feature / modeling / engineering improvements.
