# Roadmap

## Done
- **EDA** — 4 datasets characterised; conventions verified by hand; cross-source event
  studies; executed notebook (`notebooks/01_exploration.ipynb`).
- **Package** — `liqsignal` (src layout, editable install), spec-critical math isolated +
  unit-tested (22 tests), thin scripts, Makefile, pyproject.
- **Scoring foundation** — `markout`/`scoring` exactly per spec; full-data baselines.
- **Features A1–A3** (45) — microprice, multi-window OBI/momentum, realized vol/amplitude
  (1s mid grid), book age/rate, multi-window liq net/abs/count + `s·liqpress` interaction,
  Bybit basis proxy, time-of-day.
- **Model + thresholding** — per-τ sample-weighted HistGBR; expected-value and CV
  Score-maximising thresholds; wired into `signal()`; results report with plots.
- **Threshold persistence (fix)** — the fitted Score-maximising threshold is now stored in
  each `model_<tau>.joblib` (`model.save(..., threshold=)`) and loaded by `signal()` by
  default, so the submission applies the operating point the report measures (previously it
  silently used the weaker expected-value cutoff). Legacy models without a threshold fall
  back to the expected-value rule.
- **Cascade / cross-exchange features** — per-venue cascade acceleration
  (`{exch}_liqaccel`, 30s-vs-300s burst rate) and Bybit−Binance liquidation-pressure
  divergence (`xexch_liqpress_{30,300}s` + taker-aligned `xexch_liqalign_*`), encoding the
  core lead-lag thesis directly. Derived from already-computed windowed quantities (no new I/O).
- **Reviewer harness** — `scripts/evaluate.py` + `make evaluate DATA_DIR=...` runs `signal()`
  on a (test) tree and reports Score / turnover per (sym, τ); `LIQSIGNAL_DATA_DIR` repoints
  the loaders so test data lives beside the train data (`data_test/`).
- **Tape-flow features (#1)** — a streamed 1s trade-flow grid (`scripts/build_flow_grid.py` →
  `artifacts/flow_grid_<sym>.parquet`, `io.FlowGrid`/`load_flow_grid`/`flow_grid_from_trades`)
  drives windowed `tfi`/`tfi_aligned`/`trade_intensity`/`flow_imbalance_mag` (VPIN-lite)/
  `signed_vol_mom` over {30,300}s. Panel uses the full-tape grid; `signal()` builds it from the
  passed trades. (True OFI/VPIN/sweep-runs still want tick/volume-bucket data — deferred.)
- **Deeper cascade dynamics (#2)** — signed `{exch}_liq_runlen`, cascade-size z-score
  `{exch}_liqz`, and Bybit→Binance `liq_lead_s`.
- **Regime descriptors (#3)** — `rskew_{30,300}`, `varratio_300`, `vol_ts_ratio(_mid)` from the
  1s mid grid.
- **Funding seasonality (#4)** — `min_to_funding` / `in_funding_window` (8h marks).
- **Purged + embargoed CV (#5)** — `analysis.fit_score_threshold` now uses time-contiguous folds
  with a max-τ embargo at boundaries (replaces random k-fold) for an honest threshold/Score.
- **Configurable feature pruning (#10)** — `train_model --n-features N` (config `N_FEATURES`):
  rank by permutation importance on validation, keep top-N, refit per (sym,τ). Opt-in (keep-all default).
- **Per-symbol models (#2, now default)** — one model per `(symbol, τ)` (`model_<sym>_<tau>.joblib`);
  the pooled path was removed. `signal()` infers the symbol (ticker/price) and loads the matching
  model. Lifted BTC validation (val τ120/300 +0.64/+0.48 vs pooled +0.20/+0.03) and test. The
  symbol×τ feature-importance grid shows BTC vs ETH rank features differently.
- **Feature-selection study** — `notebooks/02_feature_selection.ipynb` (`make feature-nb`):
  missingness, univariate corr/MI, correlation clustering, train→val importance stability, PCA, and
  a top-N validation-Score sweep. Findings (see `features.md`): weak univariate signal (max
  |corr|≈0.17, led by `*_liqalign`/`signed_vol_mom`/`ret_*_signed`), several redundant blocks, PCA
  unhelpful, and **optimal N is opposite by symbol** — BTC wants very few features (τ30 +0.49→+1.44
  at N=5, Score falls as N grows) while ETH is *hurt* by small N and peaks at N≈25–40. So
  **per-`(sym,τ)` N**, not a global cutoff; ETH's τ120/300 weakness is regime shift, not excess
  features. Pruning is opt-in via `make train N_FEATURES=N`.
  **Caveat (important):** a first cut populated `config.FEATURE_SETS` by ranking on *validation*
  importance (redundancy-filtered, `scripts/select_features.py`); retraining on it **improved
  validation but degraded the held-out test** (ETH τ300 test 6.00→2.18) — a val-selection leak.
  Reverted to all features (`FEATURE_SETS={}`); the script is kept. See `features.md`.

## Done (2026-05-26 — model-improvement cycle)
- **Walk-forward OOS harness** (`src/liqsignal/backtest.py`, `make walkforward`) — expanding-window
  backtest judging any model spec on three held-out months (Feb/Mar/Apr); reproduces the shipped
  validation Scores on the matching fold. Plus a per-month regime diagnostic (`make regime`).
- **Regime sign-flip characterised** — the liquidation edge flips sign in Dec (both) and March (ETH);
  that, not features/threshold, is the source of ETH's negative validation.
- **Objective reframed (#4) + ETH regime fixed (#5)** — per-`(symbol, τ)` estimators in
  `config.MODEL_SPECS` / `model.fit_model`: HGBR **MAE** (robust, beats MSE broadly), HGBR **recency**
  weights (BTC τ300), and **LightGBM quantile** for ETH τ120/300 (turns the −7.06 March blow-up to
  +0.26). Judged on mean OOS Score; the submission `predict` path stays uniform.
- **Monotonic constraints (#3)** — tested in the harness; helped ETH means but hurt BTC variance, so
  **not adopted** (left available as a `backtest` spec).
- **Leak-free feature selection — derived, judged, decided** — `scripts/select_features.py` ranks
  importance on a train-internal fit/selection split (RANKER = MSE-HGBR), picks N by a train-internal
  sweep + knee scored with the *deployed* estimator (JUDGE), and emits `feature_selection_sweep` +
  `feature_importance_rank` parquets. The `features` walk-forward spec is now shipped-all vs
  shipped-curated. **Verdict: all-73 won 5/6 cells on mean OOS (only ETH τ30 marginally passed);
  `FEATURE_SETS` stays `{}`.** Why-they-help analysis in `scripts/analyze_feature_sets.py`
  (`make feature-explain`) + notebook §8 + features.md: the edge is spread across many
  consistent-sign features, so pruning loses signal without a variance payoff.

## Next (highest value first)

### Modeling / robustness
1. **Probability calibration** — only matters for a fixed/expected-value cutoff: the score-max
   threshold is invariant to any monotone calibration, so the classifier/quantile scores need
   isotonic/Platt only if we want them to read as probabilities for an EV rule.
2. **Combine levers** — e.g. quantile **+** recency for ETH long-τ, or MAE **+** recency — untested
   interactions that might beat the single-lever per-cell winners on the harness.
3. **Confirm out-of-regime** — three OOS months is modest and the per-cell picks carry some
   selection optimism; re-judge when more data (or a non-drawdown regime) is available.

### Features (deferred / harder)
6. **True OFI / VPIN / sweep-runs** — BBO-based Cont-style order-flow imbalance, volume-bucketed
   VPIN, and tick-level sweep/run-length (need finer-than-1s or volume-bucket structure).

### Engineering
11. **Ship/download trained models** so reviewers can `make evaluate` without retraining
    (models are gitignored; provide them out-of-band or a fetch step).
12. **End-to-end signal smoke test** on a tiny fixture dataset under `tests/fixtures/` so the
    full `signal()` model path is exercised in CI, not just the pure functions.
13. **`git init`** — the repo is currently untracked.

## Open questions
- Is the turnover constraint per-symbol or pooled across the universe? (Ambiguous in spec;
  we treat per-symbol, which is stricter/safe. Either way it barely binds.)
- Does the hidden test call `signal()` once per symbol (assumed) or on a mixed frame?
- How much of the Bybit edge is real cross-exchange information vs. coincident timing? (The
  +200 ms shift is honoured; the event study suggests it's genuine lead, but worth re-checking
  out-of-regime.)

## Where the bodies are buried
- 16 GB RAM ⇒ all the chunked/searchsorted patterns are deliberate; don't refactor to
  whole-file loads or `join_asof` (see workflows-and-gotchas.md).
- Bybit feed unsorted + needs +200 ms; trades/liq `side` semantics differ; µs timestamps.
