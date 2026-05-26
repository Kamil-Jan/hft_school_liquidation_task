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
- **Cascade anatomy (§6.4, `precompute_cascades.py`):** clustering liquidations into cascades
  (both venues, gap < 10 s, ≥ 5 prints; ~21k each symbol, median span ≈ 13 s, ~55% sell-driven)
  and tracking the Binance microstructure `pre → begin → middle → end → after`. The onset is a
  **flow + volatility burst**: at `begin` trade intensity ≈ **2×** baseline, signed taker flow ≈
  **3–3.6×**, and the **spread widens ≈ 1.5–1.6×**; the mid runs into the cluster and peaks near the
  end. The **book imbalance flips** from leaning *with* the pressure (pre/begin) to *against* it
  (end/after) — the microstructure signature of the turn. The maker edge is a **sign flip in
  realized markout**: `pnl_120` is **negative in `pre`/`begin`** (≈ −1.5 BTC / −2.1 ETH bps — fills
  run over by the continuation) and turns **positive in `middle`/`end`/`after`** (+0.06–0.14 BTC,
  +0.2–0.36 ETH — fills catch the reversion). Bybit carries ~60–65% of cascade notional throughout.
  ⇒ a good filter must read the cascade's **phase** (via the windowed flow/liq/vol features), not
  just its presence.
- **Trade price impact:** taker trades have persistent, size-scaling impact in their own
  direction (adverse selection) — but mostly *informational*, since trades are tiny vs the
  ~$320k top-of-book depth.

## Baselines (`artifacts/baselines.parquet`, full data)
Maker collecting **all** trades (rebate included) is ~break-even and **regime-dependent**:
BTC validation PnL_all is negative at all horizons (−0.15 to −0.19 bps — the drawdown
month), ETH mostly positive. Adverse selection dominates at 30 s; rebate + reversion make
longer horizons less bad → the maker edge grows with horizon. Clipped turnover ≈$11–15
B/day vs the $500k floor (**~25,000× headroom** → constraint essentially non-binding).

## Single-feature study (historical; tooling now `make feature-explain`)
Conditional w-weighted markout cleanly separates trades by feature. Robust, generalising
single features: `ret_5s_signed` (momentum-into-trade → exhaustion/reversion, strongest),
`px_vs_mid_bps` (aggressive sweep reverts), `obi_signed` (contrarian book = paper-1 "reversal").
Liquidation-pressure features are sparse specialists (only fire on the ~10% of trades near a
cascade) but generalise. (The standalone `run_study.py` / `make study` was retired — its
conditional-markout view is subsumed by `make feature-explain`, which reports the same quintile
edge plus importance rank and per-month regime survival for the actually-selected features.)

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

## Walk-forward OOS study + per-cell estimators (2026-05-26)
The single val-month / test-month read was too thin to judge regime robustness (and had let
an earlier feature-selection pass overfit validation). `src/liqsignal/backtest.py` +
`make walkforward` now judge any model spec on an **expanding-window backtest** (train→Feb,
train→Mar, train→Apr; 300 s embargo) by **mean OOS Score across the three held-out months**.
Gate check: the March fold (train Nov–Feb) reproduces the shipped validation Scores exactly.

**The regime problem, quantified (`make regime`, `report.regime_by_month`).** The
liquidation→reversion edge (top−bottom-quintile w-markout of `bybit_liqalign_300s`) is real
but **its sign flips by month**: positive in Nov/Jan/Feb/Apr, **negative in December for both
symbols**, and **negative across all horizons for ETH in March** (−1.57 at τ300). That flip —
not excess features or a bad threshold — is why ETH validation (=March) goes negative
(baseline ETH τ300 March Score −7.06).

**What helps (mean OOS Score, baseline = HistGBR-MSE all-73):**
- **Robust MAE regression** (`loss="absolute_error"`) beats MSE almost everywhere (it stops
  chasing heavy-tailed markout outliers): BTC τ30 1.30→1.61, τ120 1.32→2.14; ETH τ30 2.31→2.78.
- **LightGBM quantile** turns ETH's long-horizon March *positive*: ETH τ120 (α=0.5) March
  −0.82→**+1.02** (mean 2.27→2.72); ETH τ300 (α=0.6) March −7.06→**+0.26** (mean 0.46→2.70).
  The conservative loss simply won't rank a trade high unless even its lower-markout-quantile
  is good, so it sidesteps the regime-flip trades.
- **Recency-weighted** HistGBR (halflife 30 d) is best for BTC τ300 (mean 1.56→1.99, March
  stays positive); also more than halves ETH τ300's March loss but quantile dominates there.
- **Monotonic +1 constraints** on aligned features help ETH means but hurt BTC variance — not adopted.

**Shipped per-(symbol, τ) estimators** (`config.MODEL_SPECS`, `model.fit_model`):
BTC τ30/τ120 = HGBR-MAE, BTC τ300 = HGBR + recency(30 d); ETH τ30 = HGBR-MAE,
ETH τ120 = LGBM-quantile α0.5, ETH τ300 = LGBM-quantile α0.6. The submission's `predict`
path is uniform (HGBR and LightGBM both expose `.predict`, higher ⇒ keep); the two ETH
long-τ cells make the shipped `signal()` import lightgbm (see README / `patch_lightgbm.py`).
*Caveat:* picking the best-of-N spec per cell on three OOS months carries some selection
optimism — the adopted specs were chosen for a clear mechanism + consistent (not single-month)
gains, but treat the magnitudes as directional.

### Leak-free feature selection — derived & judged (2026-05-26)
The deferred redo is done (`scripts/select_features.py` → `make feature-select`): a **RANKER**
(MSE-HGBR permuted on a train-internal selection block) orders features; a **JUDGE** (the deployed
per-cell estimator) picks N by a train-internal sweep + parsimony knee — no validation touched, and
the old val-derived `NMAP` is gone. Adoption was decided on the OOS gate
(`walk_forward.py --specs features`: shipped-estimator all vs curated).
**Verdict: all-73 won 5/6 cells** (mean OOS, all→curated: BTC 1.61→1.48, 2.14→1.55, 1.99→1.24;
ETH 2.78→**2.83**, 2.72→2.04, 2.70→1.69 with March −2.43). Only ETH τ30 passed and only marginally
(+0.05 mean, within noise), so **`config.FEATURE_SETS` stays `{}` — keep all features.** The
train-internal sweep looked optimistic for several cells (the OOS gate caught it), and the
validation sweep's clean BTC-few/ETH-many shape did **not** reproduce leak-free (flat/noisy curves →
that asymmetry was partly a val artifact). **Why all-73 is hard to beat:** the edge is *spread* —
the why-they-help analysis (`make feature-explain` → `feature_explanations.parquet`, notebook §8)
shows the liquidation-alignment family (`bybit_liqalign_5s` train edge +1.7…+4.2 bps, sign-consistent
5–6/6 months), signed momentum (`ret_*_signed`, `signed_vol_mom_*`), cascade size (`bybit_liqabs_300s`)
and the volatility-regime gates (`vol_ts_ratio`, `ampl_*`) all carry stable-sign edges; the prunable
features are a minority of flippy/weak ones (2/6 months), so cutting to N≈25 loses signal without a
variance payoff. Full table + per-feature evidence in [`features.md`](features.md).

## Risks / caveats
- **Regime risk is the main threat.** This is one mean-reverting *drawdown* quarter; the
  reversion edge could weaken in a strong trend. The hidden test is other dates → check
  per-month stability (`report.fig_monthly_stability`).
- Train CV Scores (9–32 bps) are far above validation (1–3 bps) — optimism; the val numbers
  are what matter and they're solidly positive.
- We have no order-book depth, no Bybit book, no own-order/fill data — caps some
  paper-1/paper-2 features.
