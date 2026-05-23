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

## Next (highest value first)

### Features (most are computable from the four submission frames alone)
1. **A4 tape-derived flow features.** TFI/OFI, trade intensity, trade-sign autocovariance,
   VPIN toxicity, sweep/run-length. Blocked on a **1s trade-flow grid**
   (`artifacts/flow_grid_<sym>.parquet`) because the 700M-row tape can't be held in RAM —
   build it with a streamed Polars group-by (like the EDA aggregates), then derive windowed
   features by prefix-sum + `searchsorted` (same trick as `features.windowed_liq`). Add the
   grid build to `build_panel` / a new script and a parallel path in `signal._model_signal`.
2. **Liquidation-cascade dynamics (deeper).** Beyond the new `liqaccel`/`xexch` features:
   cascade-size z-score vs a trailing distribution, directional run-length of consecutive
   same-side liquidations, and an explicit Bybit→Binance lead-lag timing feature. All cheap —
   the liq feeds are only hundreds of thousands of rows.
3. **Regime descriptors.** Variance ratio (trend vs mean-revert), short/long realized-vol
   term-structure, realized skew. Directly targets the regime dependence that is the main
   hidden-test risk (this is one drawdown quarter).
4. **Funding-cycle seasonality.** Perps fund every 8h (00:00/08:00/16:00 UTC); add
   minutes-to-funding / in-funding-window indicators. Cheap; purely time-derived.

### Modeling / robustness
5. **Purged + embargoed CV.** Markout windows (up to 300s) overlap between nearby trades, so
   plain k-fold leaks — the train-CV Scores (9–32 bps) ≫ validation (1–3 bps) gap is the
   symptom. Purge/embargo around fold boundaries to get an honest threshold and Score estimate.
6. **Probability calibration** — isotonic/Platt on a held-out fold so the score reads as a
   probability and the expected-value cutoff is principled; compare to score-max.
7. **Per-symbol vs pooled models** — currently pooled (one model per τ). BTC underperforms
   ETH (BTC val sometimes negative); test per-symbol specialisation.
8. **Monotonic constraints** in HistGBR for sign-known features (e.g. `*_liqalign`,
   `xexch_liqalign`) — cheaper variance, more robust out-of-regime, more interpretable.
9. **Reframe the objective.** The goal is *ranking* trades by markout, not minimising MSE —
   try a classification (good/bad) or quantile/asymmetric loss; add a minimum-keep-fraction
   floor above the turnover floor so the operating point can't degenerate.
10. **Feature pruning** — 45→ fewer; importance is concentrated, so drop redundant columns
    to cut variance.

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
