# Architecture

## Design principles
- **Spec-critical math is isolated and unit-tested.** `markout`, `scoring`, `splits`
  are small, pure (no I/O), and tested against hand-computed examples, so the grading
  logic cannot silently drift from `description.md`.
- **One source of truth for the spec.** All constants (horizons, rebate, notional cap,
  Bybit delay, turnover floor, split boundaries, paths) live in `config.py`.
- **Features are pure functions** evaluated through a reused `FeatureContext`, so the
  *same* code computes features for the training sample and for the full submission
  frames (the latter in memory-bounded batches).
- **Scripts only orchestrate** (argparse → package call → write artifact). No logic
  hides in scripts or the notebook.

## Module map (`src/liqsignal/`)
| module | responsibility | key API |
|---|---|---|
| `config.py` | paths, universe, frozen spec constants | `dataset_path(source, sym)`, `TAUS`, `TRAIN_START/VAL_START/VAL_END`, `NOTIONAL_CAP`, `BYBIT_DELAY_US`, `TURNOVER_MIN_PER_DAY` |
| `splits.py` | train/val/other assignment | `assign_split(ts)` (numpy), `split_expr()` (polars) — kept in sync |
| `io.py` | data access | `BookTop`, `Liquidations` dataclasses; `book_top_from_frame`, `liquidations_from_frame` (work on in-memory frames); `load_book_top`, `load_liquidations`, `sample_trades`, `iter_trade_batches`, `scan` |
| `markout.py` | spec maker-PnL math | `trade_sign`, `forward_fill_mid`, `markout_bps`, `compute_markout(ts,sign,price,bbo_ts,bbo_mid,tau)`, `last_index_at` |
| `scoring.py` | Score + turnover | `evaluate_filter(pnl,w,f,n_days,turnover_scale) -> ScoreResult`, `weighted_mean` |
| `features.py` | feature engineering | pure fns (`order_book_imbalance`, `microprice_adjustment_bps`, `windowed_liq`, `basis_proxy_bps`, …); `FeatureContext` + `build_context`; `compute_features(ctx,ts,sign,price) -> dict`; `feature_columns`; `build_feature_panel(sym)` |
| `analysis.py` | studies + thresholding | `conditional_markout`, `fit_keep_best`/`apply_keep_best` (old single-feature rule), `expected_value_threshold`, `fit_score_threshold` (CV Score-max), `apply_threshold`, `score_split` |
| `model.py` | per-(sym,τ) predictor | `fit_model(panel,tau,symbol)` dispatches `config.MODEL_SPECS` → HistGBR (`train_markout_model`, optional `loss`/`recency_weight`) or `fit_lgbm_quantile`; `predict_markout`, `predict_from_features` (uniform `.predict`), `save`(+`kind`)/`load`/`load_threshold` |
| `backtest.py` | walk-forward OOS judge | `run_walk_forward(panel,step,tau,fit_fn)`, `evaluate_specs`/`summarize`; `fit_fn` factories (hgbr/recency/monotonic/clf/lgbm/shipped) + `SPEC_SETS` |
| `baselines.py` | full-data references | `compute_baselines()` → PnL_all + turnover/day per (sym,split,tau) |
| `report.py` | results report | `generate(panels,steps,models,features,thresholds)` → `artifacts/report/`; `regime_by_month`, walk-forward + spec sections |
| `signal.py` | **submission entry point** | `signal(trades,bbo,liq_binance,liq_bybit, *, filter_fn=None, thresholds=None, cost=0.0)` |

## Data flow
```
raw parquet (data/)
   │  io.load_* / *_from_frame
   ▼
BookTop, Liquidations (sorted numpy arrays)        sample_trades / iter_trade_batches
   │                                                       │
   └────────────► features.build_context ──► compute_features(ctx, trade arrays) ──┐
                                                                                    ▼
                       markout.compute_markout (label) ───────────► panel (polars DF)
                                                                                    │
                                          model.fit_model (per (sym,τ) estimator spec, w-weighted)
                                                                                    │
                                          score = model.predict ; analysis.fit_score_threshold
                                                                                    ▼
                                          report.generate  /  signal._model_signal → 0/1 arrays
```

## The submission path (`signal._model_signal`)
1. Infer the symbol (`ticker` col or price level) and load its per-τ models from
   `artifacts/model_<sym>_<tau>.joblib` (keep-all fallback + warning if absent). A model may
   be HistGBR or LightGBM — `predict_from_features` calls `.predict` either way (loading a
   LightGBM blob requires `lightgbm` importable; see README/`patch_lightgbm.py`).
2. Build one `FeatureContext` from the passed frames (`io.book_top_from_frame`,
   `io.liquidations_from_frame`, `io.flow_grid_from_trades`). The context precomputes the
   1-second mid grid (rolling vol/amplitude/regime), the flow grid, and mid-change timestamps once.
3. Iterate trades in `BATCH = 5M` chunks: `compute_features` → `model.predict_from_features` per τ →
   threshold (each model's persisted Score-max cutoff, or supplied `thresholds[tau]`, else
   expected-value `score<cost`) → fill the 0/1 output. Memory stays bounded (output is 1 byte/trade;
   feature matrix is per-batch).

## How to extend
- **Add a feature:** write a pure fn in `features.py`, add it inside `compute_features`
  (must be computable from `FeatureContext` only — BBO/liq arrays + grid), add a unit
  test, rebuild panels (`make panel`), retrain (`make train`). It auto-enters the model
  via `feature_columns` (anything not in `NON_FEATURE_COLUMNS`).
- **Tape-derived features (TFI/VPIN/intensity)** need a streamed **1s trade-flow grid**
  (the 700M-row tape can't be held in RAM) — see roadmap; build it like the EDA
  aggregates and join by prefix-sum + `searchsorted`.
- **Change the model/threshold:** add a `fit_fn` spec in `backtest.py`, judge it with
  `make walkforward --specs ...`, then if it wins set `config.MODEL_SPECS[(sym,τ)]` and teach
  `model.fit_model` the new `kind`. Thresholding lives in `analysis`. Submission wiring in
  `signal.py` is unchanged as long as the estimator exposes `.predict` (higher ⇒ keep).
- **Per-(symbol, τ) models:** one model per `(symbol, τ)` (`model_<sym>_<tau>.joblib`),
  each with its own estimator (`config.MODEL_SPECS`) and persisted threshold; `signal()`
  infers the symbol and loads the matching pair. (The old pooled-across-symbols model is gone.)
