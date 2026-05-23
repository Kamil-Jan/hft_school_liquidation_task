# Findings

## EDA (see `notebooks/01_exploration.ipynb`)
- **Shape:** 8 tables, identical 90-day span. ≈1.1B trades, ≈207M BBO ticks, ~0.63M
  liquidations. Microsecond timestamps. Clean (no crossed books / NaNs).
- **Diurnal:** activity peaks ~15:00 UTC (US open / macro hour), troughs ~05–10:00 UTC;
  ~3.8× swing for trades, up to ~9× for Bybit liquidations.
- **The core relationship (cross-source event study):** around a liquidation the Binance
  mid (a) *runs into* the event for the prior 10–60 s (price rising into a buy-liq /
  falling into a sell-liq → liquidations mark a **local extreme**), (b) shows a tiny
  same-direction **continuation** for ~1–2 s, then (c) **reverts** over tens of seconds
  to minutes. Two asymmetries: **Bybit ≫ Binance** (Bybit's own-liq reversion ~10× larger;
  ETH sell-liqs ≈ +15 bps at 300 s vs Binance ≈ +0.4 bps), and **ETH ≫ BTC**, sell ≫ buy
  (this falling-market sample). The +200 ms handicap costs ~nothing of this multi-minute move.
- **Trade price impact:** taker trades have persistent, size-scaling impact in their own
  direction (adverse selection) — but mostly *informational*, since trades are tiny vs the
  ~$320k top-of-book depth.

## Baselines (`artifacts/baselines.parquet`, full data)
Maker collecting **all** trades (rebate included) is ~break-even and **regime-dependent**:
BTC validation PnL_all is negative at all horizons (−0.15 to −0.19 bps — the drawdown
month), ETH mostly positive. Adverse selection dominates at 30 s; rebate + reversion make
longer horizons less bad → the maker edge grows with horizon. Clipped turnover ≈$11–15
B/day vs the $500k floor (**~25,000× headroom** → constraint essentially non-binding).

## Single-feature study (`scripts/run_study.py`)
Conditional w-weighted markout cleanly separates trades by feature. Robust, generalising
single features (fit on train, applied to val): `ret_5s_signed` (momentum-into-trade →
exhaustion/reversion, strongest), `px_vs_mid_bps` (aggressive sweep reverts), `obi_signed`
(contrarian book = paper-1 "reversal"). Liquidation-pressure features are sparse
specialists (only fire on the ~10% of trades near a cascade) but generalise.

## Model + threshold (`scripts/train_model.py`, `artifacts/report/`)
Combined per-τ `HistGradientBoostingRegressor` (45 features, sample-weighted by `w_i`,
pooled BTC+ETH train) predicting markout, then a threshold on the predicted score.

**Validation Score (bps) — model+Score-max beats the old keep-10% everywhere:**

| method | BTC τ30 | τ120 | τ300 | ETH τ30 | τ120 | τ300 |
|---|---|---|---|---|---|---|
| model + score-max | **1.71** | **0.99** | **1.52** | **3.07** | **3.04** | **3.14** |
| model + expected-value | 0.90 | 0.71 | 0.45 | 1.52 | 1.41 | 1.26 |
| keep-10% `ret_5s` (old) | 0.41 | 0.85 | 1.21 | 1.94 | 2.40 | 2.98 |

All operating points clear the turnover floor by ~100×. score-max keeps ~7–12% of trades;
expected-value keeps ~27–36%.

**Top features (permutation importance, τ=120):** `bybit_liqabs_300s` (Bybit cascade size),
`hour` (time-of-day), `bybit_liqalign_300s` (taker × Bybit-liq-pressure interaction),
`ampl_300s` (volatility amplitude). Confirms the cross-exchange thesis (Bybit > Binance) and
regime dependence. Calibration (predicted vs realised by decile) is monotonic out-of-sample;
slope <1 at the tails (over-predicts magnitude) but ordering holds — which is what
thresholding needs.

## Risks / caveats
- **Regime risk is the main threat.** This is one mean-reverting *drawdown* quarter; the
  reversion edge could weaken in a strong trend. The hidden test is other dates → check
  per-month stability (`report.fig_monthly_stability`).
- Train CV Scores (9–32 bps) are far above validation (1–3 bps) — optimism; the val numbers
  are what matter and they're solidly positive.
- We have no order-book depth, no Bybit book, no own-order/fill data — caps some
  paper-1/paper-2 features.
