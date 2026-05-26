# CLAUDE.md — orientation for AI sessions

Read this first. Deeper reference lives in [`.claude/docs/`](.claude/docs/) (linked at the bottom).

## What this project is
Build a signal that **filters Binance perpetual maker trades** (BTC & ETH) using
trade / BBO / liquidation data (Binance + Bybit) so the *kept* trades have a better
markout than the unfiltered baseline, subject to a $500k/day kept-turnover floor.
The deliverable is a function `signal(trades, bbo, liq_binance, liq_bybit)` returning,
for each τ ∈ {30,120,300}s, a 0/1 array (1 = filter out). Spec: [`description.md`](description.md)
(in Russian). Data is 179 days, 2025-11-01 → 2026-04-28 UTC. The train/val/test split
is configurable in `config.py` (four dates + a `USE_TEST` toggle; leak-safe embargo) —
see [`.claude/docs/data-and-conventions.md`](.claude/docs/data-and-conventions.md).

## Where things are
- **`src/liqsignal/`** — installable package (`pip install -e .`). Modules:
  `config` (paths + frozen spec constants + `MODEL_SPECS`), `splits` (incl. `walk_forward_folds`),
  `io` (data access), `markout` (spec PnL math), `scoring` (Score + turnover), `features`
  (feature engineering), `analysis` (study + thresholding), `model` (per-(sym,τ) estimators via
  `fit_model`), `backtest` (walk-forward OOS harness + experiment specs), `baselines`, `report`,
  `signal` (submission entry point).
- **`scripts/`** — thin runners: `compute_baselines.py`, `build_flow_grid.py`, `build_panel.py`,
  `train_model.py`, `walk_forward.py`, `regime_diagnostic.py`, `build_report.py`,
  `evaluate.py`, `build_feature_selection_nb.py`, `select_features.py` (leak-free N-sweep),
  `analyze_feature_sets.py` (why-they-help), `patch_lightgbm.py`; `scripts/eda/` builds the
  exploration notebook.
- **`notebooks/01_exploration.ipynb`** (EDA, 13 charts; §6.4 = liquidation-cascade microstructure) and
  **`02_feature_selection.ipynb`** (`make feature-nb`; §8 = leak-free selection) — executed narratives.
- **`tests/`** — pytest (45 tests; spec math + features + thresholding + signal + model + backtest + select_features).
- **`artifacts/`** — all computed outputs (gitignored): `baselines.parquet`,
  `flow_grid_<sym>.parquet`, `panel_<sym>.parquet`, `model_<sym>_<tau>.joblib`, `report/`
  (incl. `walkforward_*.parquet`, `regime_by_month.parquet`), EDA tables.
- **`.venv`** — Python 3.9 venv (system python; no uv/homebrew). Polars 1.36, sklearn 1.6.1,
  **lightgbm 4.6** (needs libomp → `make install` runs `patch_lightgbm.py`; see [`lightgbm note`](README.md)).

## How to run (Makefile)
```
make install     # editable install + dev/notebook extras (+ patch_lightgbm)
make test        # unit tests (fast)
make baselines   # full-data PnL_all + turnover/day        (~4 min)
make panel       # sampled feature panels (3M/symbol)        (~20 s)
make train       # fit per-(sym,τ) models + thresholds + report  (~15 min w/ lgbm)
make walkforward # expanding-window OOS backtest (WF_SPECS=baseline|regime|objective|shipped|features)
make feature-select  # leak-free N-sweep -> FEATURE_SETS dict + feature_selection_sweep.parquet
make feature-explain # why the chosen features help -> feature_explanations.parquet
make regime      # per-month edge/markout/vol diagnostic
make report      # regenerate report from trained models (no refit)
make eda         # rebuild + execute the exploration notebook
make feature-nb  # rebuild + execute notebooks/02_feature_selection.ipynb (leak-free study)
```
Pipeline order from scratch: `install → panel → train` (and `baselines` for reference).

## Conventions that bite (verify, don't assume)
- **Timestamps are int64 microseconds UTC.** `t/1e6` = epoch seconds. (ms→year 57000.)
- **`side` differs by table:** in *trades* it's the **taker** side (buy = lifted ask);
  in *liquidations* it's the **liq-order** side (buy = forced buy = upward pressure).
- **Bybit liquidations:** apply **+200 ms** before comparing to Binance time, AND
  **sort first** — the Bybit feed is *not* time-sorted and has µs-collisions.
  (`io.liquidations_from_frame` handles both.)
- **16 GB RAM.** Trade files are 800 M–1.4 B rows (BBO ~200 M); never load whole. Polars
  `join_asof` OOMs on ETH — use the chunked/`searchsorted` patterns in
  `baselines.py` / `io.iter_trade_batches` / `signal._model_signal`.
- **Spreads are ~1 tick** (median ≈0.01–0.03 bps) → the +0.5 bps rebate and the
  markout dominate maker PnL, not the spread.
- **Turnover constraint barely binds** (~$11–15 B/day vs $500k floor, ~25,000×) →
  the problem is prediction quality, not turnover budgeting.

## Current state (2026-05-26)
Signal pipeline + **per-(symbol, τ) estimators** shipped, chosen by a **walk-forward OOS study**
(`make walkforward`; the honest multi-month judge that replaced the single val/test read). 73
features (tape-flow/cascade/regime/funding), sample-weighted by `w_i`; `signal()` infers the
symbol and loads `model_<sym>_<tau>.joblib` (predict path uniform across estimator kinds).
Each cell's estimator is in `config.MODEL_SPECS` (`model.fit_model` dispatches):
**HGBR-MAE** for BTC τ30/120 + ETH τ30 (robust regression beats MSE broadly), **HGBR + recency**
(halflife 30 d) for BTC τ300, **LightGBM quantile** for ETH τ120/300. Each keeps its purged-CV
Score-max threshold (persisted, applied by `signal()`). **Key win:** ETH validation (March) τ120/300
went −0.82/−7.06 → **+1.02/+0.26** — the quantile models stop betting on the regime sign-flip.
Top features unchanged (`bybit_liqabs_300s`, `hour`, `bybit_liqalign_300s`, `ampl_300s`).
**Feature selection: all 73 kept** (`config.FEATURE_SETS = {}`) — a leak-free N-sweep
(`make feature-select`, train-internal ranker / deployed-estimator judge) was derived and judged on
the OOS gate (`walk_forward.py --specs features`); **all-73 won 5/6 cells** (only ETH τ30 marginally
passed), because the edge is spread across many consistent-sign features. See `make feature-explain`
+ features.md. See `artifacts/report/report.md` after `make train`; full results in
[`.claude/docs/findings.md`](.claude/docs/findings.md).

## Core thesis
A liquidation marks the local extreme of a fast move → tiny same-direction
continuation (~1–2 s) → multi-minute **mean-reversion** of the Binance mid. **Bybit
liquidations predict that reversion ~10× more strongly than Binance's own**, and it
survives the +200 ms delay. The filter keeps trades whose predicted markout is high.
**But the edge is regime-conditional** — its sign *flips* in some months (Dec for both
symbols; March for ETH), which is why ETH long-τ validation went negative and why the
conservative quantile objective (which won't bet on uncertain trades) helps there.

## Deeper docs
- [`.claude/docs/architecture.md`](.claude/docs/architecture.md) — package design, data flow, how to extend.
- [`.claude/docs/data-and-conventions.md`](.claude/docs/data-and-conventions.md) — schemas, scale, quirks, the conventions in detail.
- [`.claude/docs/features.md`](.claude/docs/features.md) — the 73 model features: definition, units, rationale.
- [`.claude/docs/findings.md`](.claude/docs/findings.md) — EDA + signal + model results.
- [`.claude/docs/workflows-and-gotchas.md`](.claude/docs/workflows-and-gotchas.md) — commands, recipes, memory patterns, pitfalls.
- [`.claude/docs/roadmap.md`](.claude/docs/roadmap.md) — done / next / open questions.
