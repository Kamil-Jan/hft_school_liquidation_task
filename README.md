# liqsignal — liquidation-driven maker-trade filter

Research code for the task in [`description.md`](description.md): **filter Binance perpetual
maker trades** (BTC & ETH) using trade / BBO / liquidation data (Binance + Bybit) so that the
*kept* trades have a **better markout than the unfiltered baseline**, subject to a $500k/day
kept-turnover floor. Universe: `perp:btcusdt`, `perp:ethusdt`; 179 days
(2025-11-01 → 2026-04-28 UTC).

The deliverable is one function:

```python
liqsignal.signal.signal(trades, bbo, liq_binance, liq_bybit) -> {30: arr, 120: arr, 300: arr}
```

returning, for each horizon τ ∈ {30, 120, 300} s, a 0/1 array over the input trades
(**1 = filter out, 0 = keep**). It is scored by **Score = PnL_kept − PnL_all** (spec maker
markout, in bps), with the constraint that kept turnover stays above $500k/day.

---

## Core thesis

A liquidation marks the **local extreme of a fast move** → a tiny same-direction continuation
(~1–2 s) → a multi-minute **mean-reversion of the Binance mid**. A maker fill that lands just
*before* the reversion earns; one that lands into the continuation loses. So the filter keeps
trades whose **predicted markout** is high.

Two empirical facts make this tradeable:

1. **Bybit liquidations predict the Binance reversion ~10× more strongly than Binance's own**,
   and the edge survives the +200 ms Bybit→Binance delay (genuine cross-exchange information).
2. **The edge is regime-conditional — its sign *flips* in some months** (December for both
   symbols; March for ETH). That flip, not feature count or thresholding, is the central
   modeling problem and the reason long-horizon ETH validation went negative under a naive fit.

---

## Architecture

**Design principles**
- **Spec-critical math is isolated and unit-tested** (`markout`, `scoring`, `splits`) — small,
  pure, no I/O — so the grading logic cannot silently drift from `description.md`.
- **One source of truth for the spec**: all constants (horizons, rebate, notional cap, Bybit
  delay, turnover floor, split dates, paths, per-cell model specs) live in `config.py`.
- **Features are pure functions** evaluated through a reused `FeatureContext`, so the *same* code
  computes features for the sampled training panel and for the full submission frames (the latter
  in memory-bounded batches).
- **Scripts only orchestrate** (argparse → package call → write artifact).

**Package (`src/liqsignal/`)**

| module | responsibility |
|---|---|
| `config` | paths, universe, frozen spec constants, `MODEL_SPECS`, `FEATURE_SETS` |
| `splits` | train/val/test assignment (NumPy + Polars, kept in sync); `walk_forward_folds` |
| `io` | data access: lazy scans, materialised sorted arrays, batched trade iterator, `*_from_frame` |
| `markout` | spec maker-PnL math (forward-filled mid, signed markout in bps) |
| `scoring` | `Score`, `PnL_all/kept/filtered`, turnover constraint (`ScoreResult`) |
| `features` | feature engineering (`FeatureContext` + `compute_features`) + panel assembly |
| `analysis` | conditional-markout study + score thresholding (expected-value, purged-CV Score-max) |
| `model` | per-`(sym,τ)` estimators via `fit_model` (HGBR / LightGBM) + persistence |
| `backtest` | walk-forward OOS harness + experiment specs (the judge for any model change) |
| `baselines` | full-data `PnL_all` + turnover/day reference |
| `report` | metrics tables + figures (threshold curves, calibration, per-month regime) |
| `signal` | **submission entry point** (loads models, batched feature compute, threshold filter) |

**Data flow**

```
raw parquet (data/)
   │  io.load_* / *_from_frame   (sort, +200 ms Bybit shift)
   ▼
BookTop, Liquidations (sorted numpy arrays)         sample_trades / iter_trade_batches
   │                                                        │
   └──────► features.build_context ──► compute_features(ctx, trade arrays) ──┐
                                                                             ▼
                  markout.compute_markout (label) ───────────────► panel (Polars DF)
                                                                             │
                              model.fit_model (per-(sym,τ) estimator, w-weighted)
                                                                             │
                  score = model.predict ; analysis.fit_score_threshold (purged CV)
                                                                             ▼
                  report.generate  /  signal._model_signal → 0/1 arrays per τ
```

---

## Data & processing

The loaders read parquet from a data directory (default `data/`, resolved by
`config.dataset_path`). Put the full dataset in `data/`; the **train / validation / test split is
carved by date inside the code**, not by separate folders. File names are exact (note: Bybit
liquidations have **no** `perp_` prefix):

```
data/                                          <- full dataset (2025-11-01 → 2026-04-28)
  binance_trades/        perp_btcusdt.parquet  perp_ethusdt.parquet
  binance_booktickers/   perp_btcusdt.parquet  perp_ethusdt.parquet
  binance_liquidations/  perp_btcusdt.parquet  perp_ethusdt.parquet
  bybit_liquidations/    btcusdt.parquet       ethusdt.parquet
data_test/                                     <- optional external test set (same layout)
```

**Schemas** (timestamps are **int64 microseconds, UTC** throughout — `t / 1e6` = epoch seconds):

| Source | Columns |
|---|---|
| trades | `timestamp, ticker, side, price, amount` |
| bbo (book tickers) | `timestamp, ticker, bid_price, bid_amount, ask_price, ask_amount` |
| liquidations (both venues) | `timestamp, ticker, side, price, amount` |

**Conventions that bite** (handled in code, but they explain the processing):
- **`side` differs by table.** In *trades* it is the **taker** side (buy = lifted ask, so the
  resting maker sold). In *liquidations* it is the **liq-order** side (buy = forced buy = upward
  pressure). The feature/markout code signs everything to the maker's perspective.
- **Bybit liquidations** are shifted **+200 ms** before any Binance comparison **and sorted
  first** — the Bybit feed is not time-ordered and has µs-collisions. `io.liquidations_from_frame`
  does both, so raw files need no preprocessing.
- **16 GB RAM.** Trade files are 800 M–1.4 B rows (BBO ~200 M); they are never loaded whole.
  Polars `join_asof` OOMs on ETH, so the code uses chunked `searchsorted` patterns
  (`io.iter_trade_batches`, `signal._model_signal`) and a precomputed **1-second mid grid** +
  **1-second trade-flow grid** inside `FeatureContext`. Training uses **sampled panels (~3 M
  rows/symbol)**; the submission path streams trades in 5 M-row batches (output is 1 byte/trade).

**Train / validation / test split** lives in `config.py` (single source of truth in `splits.py`):
four dates + a `USE_TEST` toggle. `USE_TEST=True` (default) → train `2025-11-01..2026-02-28`,
validation `2026-03`, test `2026-04`; `USE_TEST=False` folds April into validation for the final
model. A `SPLIT_EMBARGO_S` (= max τ = 300 s) gap is **purged before each boundary** so no trade's
markout window straddles two splits (leak-safe). After changing any of these, re-run
`make panel && make train`. Point the loaders elsewhere with `LIQSIGNAL_DATA_DIR` (or
`make ... DATA_DIR=data_test`).

---

## Features

**73 features per `(symbol, τ)`**, all pure functions of the BBO/liquidation arrays + the 1 s mid
and trade-flow grids. The model matrix is everything except meta/label columns (`timestamp`,
`side`, `price`, `notional`, `w`, `day`, `split`, `pnl_{30,120,300}`).

| Family | n | Examples |
|---|---:|---|
| Pre-trade top-of-book | 4 | `obi`, `obi_signed`, `micro_signed_bps`, `px_vs_mid_bps` |
| Momentum into the trade | 3 | `ret_{1,5,30}s_signed` |
| Realized vol / amplitude | 6 | `rv_{5,30,300}s`, `ampl_{5,30,300}s` |
| Top-of-book dynamics | 2 | `book_age_s`, `book_chg_rate_30s` |
| Liquidation pressure (venue × window) | 24 | `{binance,bybit}_liq{press,abs,cnt,align}_{5,30,300}s` |
| Time since last liquidation | 2 | `dt_last_{binance,bybit}_liq_s` |
| Cascade acceleration | 2 | `{binance,bybit}_liqaccel` |
| Cross-exchange liq divergence | 4 | `xexch_liq{press,align}_{30,300}s` |
| Cross-exchange basis | 2 | `basis_bps`, `basis_signed_bps` |
| Seasonality (incl. funding) | 4 | `hour`, `is_weekend`, `min_to_funding`, `in_funding_window` |
| Tape-derived flow | 10 | `tfi_{30,300}s`, `tfi_aligned_{30,300}s`, `trade_intensity_{30,300}s`, `signed_vol_mom_{30,300}s`, `flow_imbalance_mag_{30,300}s` |
| Cascade dynamics | 5 | `{binance,bybit}_liq_runlen`, `{binance,bybit}_liqz`, `liq_lead_s` |
| Regime descriptors | 5 | `rskew_{30,300}`, `varratio_300`, `vol_ts_ratio(_mid)` |

The `*_align*` features are **signed by the taker side** so one learned relationship serves both
buy and sell fills. Top features by permutation importance: `bybit_liqabs_300s` (Bybit cascade
size) ≫ `hour` > `bybit_liqalign_300s` (taker × Bybit pressure) > `ampl_300s` (vol amplitude) —
i.e. the cross-exchange-reversion thesis plus strong diurnal / volatility-regime dependence. Full
catalogue with units and rationale: [`.claude/docs/features.md`](.claude/docs/features.md).

**Feature selection:** all 73 are kept (`config.FEATURE_SETS = {}`). A leak-free selection study
(below) found that pruning loses more than it saves.

---

## Models

One estimator **per `(symbol, τ)` cell** (`config.MODEL_SPECS`, dispatched by `model.fit_model`),
each chosen by the walk-forward OOS study — not a single pooled model. All are sample-weighted by
`w = min(notional, $100k)` and trained to predict the spec markout; the submission keeps trades
whose predicted markout clears a **persisted, purged-CV Score-maximising threshold**.

| cell | estimator | why |
|---|---|---|
| BTC τ30, τ120 | **HGBR, MAE loss** (`absolute_error`) | robust to heavy-tailed markout outliers; beats MSE broadly |
| BTC τ300 | **HGBR + recency** (halflife 30 d) | leans on the nearest regime |
| ETH τ30 | **HGBR, MAE loss** | same robustness win |
| ETH τ120 | **LightGBM quantile** (α 0.50) | conservative — won't bet on regime-flip trades |
| ETH τ300 | **LightGBM quantile** (α 0.60) | same; fixes the March sign-flip |

The submission `predict` path is **uniform** — HGBR and LightGBM both expose `.predict` (higher ⇒
keep) — so `signal()` is unchanged across estimator kinds. `signal()` infers the symbol and loads
`artifacts/model_<sym>_<tau>.joblib` (keep-all fallback if absent). The two ETH long-τ cells make
the package import `lightgbm` when scoring ETH (see the OpenMP note under Setup).

**The judge — walk-forward OOS.** Every model change is evaluated on an **expanding-window
backtest** (`make walkforward`): train→Feb, train→Mar, train→Apr with a 300 s embargo, scored by
**mean Score across the three held-out months**. This replaced the single val-month / test-month
read that previously let changes overfit one month.

---

## Pipelines

```bash
make install     # editable install + dev/notebook extras (+ patch_lightgbm)
make test        # unit tests (45; spec math + features + thresholding + signal + model + backtest)
make baselines   # full-data PnL_all + turnover/day per symbol/split/τ        (~4 min)
make panel       # sampled feature panels with markout + features (~3 M/sym)   (~20 s)
make train       # fit per-(sym,τ) models + thresholds, write report          (~15 min w/ lgbm)
make walkforward # expanding-window OOS backtest (WF_SPECS=baseline|regime|objective|features|shipped)
make regime      # per-month edge / markout / vol diagnostic
make report      # regenerate report from trained models (no refit)
make feature-select   # leak-free N-sweep -> FEATURE_SETS dict + feature_selection_sweep.parquet
make feature-explain  # why the chosen features help -> feature_explanations.parquet
make eda         # rebuild + execute notebooks/01_exploration.ipynb
make feature-nb  # rebuild + execute notebooks/02_feature_selection.ipynb (leak-free study)
make evaluate    # run signal() on a data dir, report Score/turnover per (sym,τ)
```

From scratch: `make install → panel → train` (and `baselines` for reference). `make train` writes
`artifacts/model_<sym>_<tau>.joblib` (each carrying its fitted threshold + estimator spec) and a
report (`artifacts/report/report.md` + figures). Executed notebooks:
`notebooks/01_exploration.ipynb` (EDA, 13 charts; §6.4 = liquidation-cascade microstructure) and `02_feature_selection.ipynb` (incl. §8, the
leak-free selection study).

---

## Setup

```bash
make venv               # python3 -m venv .venv  (Python 3.9, system python; no uv/homebrew)
make install            # pip install -e ".[dev,notebook]"  (editable)
make test               # run unit tests
```

Stack: Polars 1.36, scikit-learn 1.6.1, **LightGBM 4.6**.

> **LightGBM / OpenMP.** Two shipped models (ETH τ120/τ300) are LightGBM quantile regressors, so
> the package imports `lightgbm` when scoring ETH. Its wheel needs an OpenMP runtime (`libomp`);
> on macOS without homebrew, `make install` runs `scripts/patch_lightgbm.py` to point it at
> scikit-learn's vendored copy. If `import lightgbm` ever fails with
> `Library not loaded: @rpath/libomp.dylib`, run `.venv/bin/python scripts/patch_lightgbm.py`
> (idempotent).

---

## For reviewers — evaluating on test data

`make evaluate` runs the submission `signal()` on a data directory, computes the spec maker
markout, and prints — per symbol and τ — `PnL_all`, `PnL_kept`, **Score = PnL_kept − PnL_all**,
the kept turnover/day, and whether the $500k/day constraint holds. Higher Score is better.

```bash
make install                          # 1. install + put TRAIN data in data/
make panel && make train              # 2a. reproduce models from scratch (~15 min)
#                                       2b. ...or skip if artifacts/model_<sym>_<tau>.joblib exist
make evaluate DATA_DIR=data_test      # 3. put TEST data in data_test/ (same layout) and score it
```

Example row: `Score=+2.750  PnL_kept=+3.10  PnL_all=+0.35  keep=8.4% keptTurn/day=1,200,000,000 OK`.

> **Memory:** a full 90-day symbol won't fit in 16 GB read whole. Evaluate a bounded window, or
> pass `--batch-size N` (e.g. `20000000`) to score in memory-bounded chunks:
> `LIQSIGNAL_DATA_DIR=data_test .venv/bin/python scripts/evaluate.py --batch-size 20000000`.

Programmatic use:

```python
from liqsignal import scoring
# build pnl, weights w, filter f (1=drop, 0=keep) ...
result = scoring.evaluate_filter(pnl, w, f, n_days=62)   # -> ScoreResult(score, pnl_kept, ...)
```

---

## Findings

**Data (see `notebooks/01_exploration.ipynb`).** Clean (no crossed books / NaNs); spreads are
~1 tick (median ≈ 0.01–0.03 bps), so the **+0.5 bps maker rebate and the markout dominate maker
PnL, not the spread**. Trades are ~50/50 buy/sell while liquidations skew sell-side. The
liquidation → mean-reversion pattern holds, and **Bybit predicts the Binance reversion ~10×
more strongly than Binance's own liquidations**, surviving the +200 ms delay.

**The problem is prediction quality, not turnover.** Clipped kept turnover is ≈ $11–15 B/day vs
the $500k floor (~25,000× headroom) — the constraint barely binds, so all the work is in ranking
which fills will revert.

**Per-(symbol, τ) estimators, chosen by walk-forward OOS** (mean Score over Feb/Mar/Apr; baseline
= HistGBR-MSE on all 73):
- **Robust MAE regression** beats MSE almost everywhere (it stops chasing heavy-tailed markout
  outliers): BTC τ30 1.30→1.61, τ120 1.32→2.14; ETH τ30 2.31→2.78.
- **LightGBM quantile** turns ETH's long-horizon **March** positive — the headline regime fix:
  ETH τ120 March −0.82 → **+1.02** (mean 2.27→2.72); ETH τ300 March −7.06 → **+0.26**
  (mean 0.46→2.70, worst-month std collapses 6.68→2.17). The conservative loss won't rank a trade
  high unless even its lower markout-quantile is good, so it sidesteps the regime-flip trades.
- **Recency-weighted HGBR** (halflife 30 d) is best for BTC τ300 (mean 1.56→1.99).
- **Monotonic constraints** on aligned features helped ETH means but hurt BTC variance — *not*
  adopted. The tradeoff accepted for the quantile cells: they give up some upside in the favorable
  April month for March downside protection — judged worth it on the 3-month mean.

**Leak-free feature selection — all 73 features kept.** `scripts/select_features.py`
(`make feature-select`) ranks importance with a stable MSE-HGBR on a *train-internal* selection
fold (val/test never touched) and picks N with the *deployed* estimator + a parsimony knee; the
walk-forward gate (`--specs features`) then compares the shipped estimator on **all features vs the
curated set**. Verdict: **all-73 won 5 of 6 cells** (mean OOS all→curated: BTC 1.61→1.48,
2.14→1.55, 1.99→1.24; ETH 2.78→2.83, 2.72→2.04, 2.70→1.69 with ETH τ300 breaking March to −2.43).
Only ETH τ30 passed and only marginally (+0.05, within noise), so nothing is adopted. **Why all-73
wins:** the edge is *spread* across many sign-stable features (`make feature-explain` shows the
liquidation-alignment family with train quintile edges of +1.7…+4.2 bps, sign-consistent in 5–6 of
6 months; plus signed momentum, cascade size, and volatility-regime gates), so pruning to ~25
loses signal without a variance payoff. A notable methodological result: the clean "BTC wants few
features / ETH wants many" shape from the older *validation* sweep **did not reproduce leak-free** —
it was partly a val-selection artifact.

**Caveat.** Three OOS months is modest and this is one mean-reverting *drawdown* quarter; the
reversion edge could weaken in a strong trend. Treat magnitudes as directional and re-judge on more
data. Full results: [`.claude/docs/findings.md`](.claude/docs/findings.md); the full
done/next/open list: [`.claude/docs/roadmap.md`](.claude/docs/roadmap.md).
