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
| `model.py` | per-τ predictor | `train_markout_model(panel,tau)` (HistGBR, `sample_weight=w`), `predict_markout`, `predict_from_features`, `save`/`load` |
| `baselines.py` | full-data references | `compute_baselines()` → PnL_all + turnover/day per (sym,split,tau) |
| `report.py` | results report | `generate(panels,steps,models,features,thresholds)` → `artifacts/report/` |
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
                                          model.train_markout_model (per τ, w-weighted)
                                                                                    │
                                          score = model.predict ; analysis.fit_score_threshold
                                                                                    ▼
                                          report.generate  /  signal._model_signal → 0/1 arrays
```

## The submission path (`signal._model_signal`)
1. Load per-τ models from `artifacts/model_<tau>.joblib` (keep-all fallback + warning if absent).
2. Build one `FeatureContext` from the passed frames (`io.book_top_from_frame`,
   `io.liquidations_from_frame`). The context precomputes the 1-second mid grid
   (rolling vol/amplitude) and mid-change timestamps once.
3. Iterate trades in `BATCH = 20M` chunks: `compute_features` → `model.predict` per τ →
   threshold (expected-value `score<cost`, or supplied `thresholds[tau]`) → fill the 0/1 output.
   Memory stays bounded (output is 1 byte/trade; feature matrix is per-batch).

## How to extend
- **Add a feature:** write a pure fn in `features.py`, add it inside `compute_features`
  (must be computable from `FeatureContext` only — BBO/liq arrays + grid), add a unit
  test, rebuild panels (`make panel`), retrain (`make train`). It auto-enters the model
  via `feature_columns` (anything not in `NON_FEATURE_COLUMNS`).
- **Tape-derived features (TFI/VPIN/intensity)** need a streamed **1s trade-flow grid**
  (the 700M-row tape can't be held in RAM) — see roadmap; build it like the EDA
  aggregates and join by prefix-sum + `searchsorted`.
- **Change the model/threshold:** swap the estimator in `model.train_markout_model`;
  thresholding lives in `analysis`. Submission wiring in `signal.py` is unchanged.
- **Pooling:** models are pooled across symbols (features are scale-free); one model per
  τ, applied regardless of symbol at submission time.
