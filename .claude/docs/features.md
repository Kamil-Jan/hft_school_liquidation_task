# Feature reference

The complete catalog of the **73 model features**, all produced by
`features.compute_features` (`src/liqsignal/features.py`). They are pure functions
evaluated against a prebuilt `FeatureContext` (full BBO + both liquidation feeds +
a 1-second mid grid + a 1-second trade-flow grid), so a trade is featurised
**identically** whether it is in the training panel (`make panel`) or scored live in
the submission path (`signal()`).

> Training can keep only the top-N most important features per horizon — set
> `N_FEATURES` (or `make train N_FEATURES=30`); ranking is permutation importance on
> validation. Default keeps all. Each τ's model stores (and `signal()` applies) its own list.

Anything a feature can't define is left **NaN** (e.g. before the BBO starts, or no
prior liquidation) — `HistGradientBoostingRegressor` routes NaN natively, so no
sentinel values pollute the distributions.

### Conventions

- **Timestamps** are int64 µs UTC; lookups are forward-fills (`np.searchsorted` into
  sorted arrays), never `join_asof`.
- **Taker sign** `s ∈ {+1, −1}`: `+1` = taker buy (the maker *sold* at the ask),
  `−1` = taker sell (the maker *bought* at the bid). The model predicts the **maker**
  markout, so features named `*_signed` / `*_align` / `micro_signed` / `px_vs_mid`
  multiply a raw market quantity by `s`, expressing it in the trade's own frame so one
  learned relationship serves both buy- and sell-initiated trades.
- **Liquidation sign**: a *buy* liquidation is forced short-covering (**upward**
  pressure, `+notional`); a *sell* liquidation is forced long-liquidation (**downward**,
  `−notional`). Bybit timestamps are shifted `+200 ms` and re-sorted before use.
- **Lookback windows** (`features.py:36`): momentum `(1, 5, 30)s`, realized vol /
  amplitude `(5, 30, 300)s`, liquidity `(5, 30, 300)s`, basis staleness gate `300s`.
- **`feature_columns`** = all panel columns minus `NON_FEATURE_COLUMNS`, so **any new
  feature key flows into the model automatically** — no registry to update.

---

## 1. Pre-trade top-of-book (4)
Forward-filled best bid/ask at the trade time (`book_state_at`).

| Feature | Definition | Units | Why |
|---|---|---|---|
| `obi` | `(bid_amt − ask_amt) / (bid_amt + ask_amt)` | [−1, 1] | Queue imbalance; `+1` = bid-heavy = latent buy pressure. |
| `obi_signed` | `s · obi` | [−1, 1] | Same imbalance in the trade's frame. |
| `micro_signed_bps` | `s · (spread/2)/mid · obi · 1e4` | bps | Microprice tilt — the OBI scaled by the half-spread; matters most when the spread widens beyond one tick. |
| `px_vs_mid_bps` | `s · (price − mid)/mid · 1e4` | bps | Trade aggressiveness: how far through the mid the taker reached. |

## 2. Momentum into the trade (3)
Signed mid return over each lookback ending at the trade.

| Feature | Definition | Units | Why |
|---|---|---|---|
| `ret_1s_signed`, `ret_5s_signed`, `ret_30s_signed` | `s · (mid_t − mid_{t−w})/mid_{t−w} · 1e4` for `w ∈ {1,5,30}s` | bps | The short-horizon move the maker is trading into — the continuation-vs-reversal context. |

## 3. Realized volatility & amplitude (6)
Computed from the 1-second forward-filled mid grid; **unsigned** (regime gates).

| Feature | Definition | Units | Why |
|---|---|---|---|
| `rv_5s`, `rv_30s`, `rv_300s` | std of 1s log-returns over the window `· 1e4` | bps | Local volatility regime — markout dispersion scales with it. |
| `ampl_5s`, `ampl_30s`, `ampl_300s` | `(max − min of mid)/mid` over the window `· 1e4` | bps | Peak-to-trough range; flags the local extreme a cascade prints into (`ampl_300s` is a top feature). |

## 4. Top-of-book dynamics (2)

| Feature | Definition | Units | Why |
|---|---|---|---|
| `book_age_s` | seconds since the last mid change (NaN before the book starts) | s | Quote staleness — a stale top-of-book is weaker evidence. |
| `book_chg_rate_30s` | (# mid changes in the last 30s) / 30 | changes/s | Quote-churn intensity / activity regime. |

## 5. Liquidation pressure per venue × window (24)
For each venue `{binance, bybit}` and window `w ∈ {5, 30, 300}s`, over `(t−w, t]`
via prefix-sum + `searchsorted` (`windowed_liq`).

| Feature | Definition | Units | Why |
|---|---|---|---|
| `{exch}_liqpress_{w}s` | Σ signed notional of liquidations in the window | USD | **Net** directional liquidation pressure (`+` upward). |
| `{exch}_liqabs_{w}s` | Σ \|notional\| of liquidations in the window | USD | **Cascade size** regardless of side (`bybit_liqabs_300s` is the #1 feature). |
| `{exch}_liqcnt_{w}s` | # liquidation events in the window | count | Cascade event count / clustering. |
| `{exch}_liqalign_{w}s` | `s · {exch}_liqpress_{w}s` | USD | Taker direction × net pressure — does the taker trade **with or against** the liquidation-driven move? (`bybit_liqalign_300s` is a top feature). |

→ 2 venues × 3 windows × 4 metrics = **24**. The cross-exchange thesis lives here:
the Bybit columns dominate the Binance ones in importance.

## 6. Time since last liquidation (2)

| Feature | Definition | Units | Why |
|---|---|---|---|
| `dt_last_binance_liq_s`, `dt_last_bybit_liq_s` | seconds since the most recent liquidation on that venue (NaN if none) | s | Recency of liquidation activity — the reversion edge is strongest just after a cascade. |

## 7. Cascade acceleration (2)
`cascade_acceleration(cnt_30s, cnt_300s, 30, 300)` per venue.

| Feature | Definition | Units | Why |
|---|---|---|---|
| `binance_liqaccel`, `bybit_liqaccel` | `(cnt_30s/30) / (cnt_300s/300)` (NaN if no 300s liqs) | ratio | Is the cascade **speeding up**? `>1` = recent burst denser than the slower baseline; `~1` = steady. |

## 8. Cross-exchange liquidation divergence (4)
Bybit-minus-Binance net pressure, for `w ∈ {30, 300}s` — encodes the lead-lag thesis directly.

| Feature | Definition | Units | Why |
|---|---|---|---|
| `xexch_liqpress_{w}s` | `bybit_liqpress_{w}s − binance_liqpress_{w}s` | USD | The Bybit-vs-Binance pressure gap; Bybit tends to lead. |
| `xexch_liqalign_{w}s` | `s · xexch_liqpress_{w}s` | USD | That gap in the taker's frame. |

## 9. Cross-exchange basis proxy (2)
`basis_proxy_bps` — a mean-divergence stand-in (we have Bybit liquidation prints but
not its book).

| Feature | Definition | Units | Why |
|---|---|---|---|
| `basis_bps` | `(last fresh Bybit liq price − Binance mid)/mid · 1e4`, **zeroed** if the last Bybit liq is older than 300s | bps | Cross-exchange price divergence — how far Bybit's last forced print sits from the Binance mid. |
| `basis_signed_bps` | `s · basis_bps` | bps | Basis in the trade's frame. |

## 10. Seasonality (4)

| Feature | Definition | Units | Why |
|---|---|---|---|
| `hour` | UTC hour of day | [0, 24) | Diurnal regime — a **top feature**; markout quality varies strongly by hour. |
| `is_weekend` | `1` on Sat/Sun UTC, else `0` | {0, 1} | Weekday/weekend regime. |
| `min_to_funding` | minutes to the next 8h funding mark (00/08/16 UTC) | [0, 480) | Funding-cycle effects on flow/price. |
| `in_funding_window` | `1` within ±5 min of a funding mark | {0, 1} | The minutes around funding behave differently. |

## 11. Tape-derived flow (10)
From the 1s **trade-flow grid** (`FlowGrid`: per-second signed/total volume + count), windowed by
prefix-sum over the whole seconds *strictly before* the trade (no intra-second look-ahead), for
`w ∈ {30, 300}s`.

| Feature | Definition | Units | Why |
|---|---|---|---|
| `tfi_{w}s` | net signed volume / total volume | [−1, 1] | Trade-flow imbalance (aggressor pressure). |
| `tfi_aligned_{w}s` | `s · tfi_{w}s` | [−1, 1] | Flow imbalance in the taker's frame. |
| `trade_intensity_{w}s` | trade count / `w` | trades/s | Activity regime. |
| `flow_imbalance_mag_{w}s` | \|net\| / total volume | [0, 1] | Order-flow toxicity (VPIN-lite), unsigned. |
| `signed_vol_mom_{w}s` | `s ·` net signed volume | volume | Taker-aligned directional volume. |

## 12. Cascade dynamics (5)
Per-venue, looked up at the last liquidation event ≤ trade time.

| Feature | Definition | Units | Why |
|---|---|---|---|
| `{exch}_liq_runlen` | signed run-length of consecutive same-side liqs (`+` buy / `−` sell) | count | Cascade persistence/direction. |
| `{exch}_liqz` | z-score of the last cascade's \|notional\| vs a trailing-event distribution | σ | Is this an unusually large liquidation? |
| `liq_lead_s` | `dt_last_binance_liq_s − dt_last_bybit_liq_s` | s | Bybit→Binance lead-lag (>0 ⇒ Bybit more recent). |

## 13. Regime descriptors (5)
From the 1s mid grid (rolling), the direct generalization lever.

| Feature | Definition | Units | Why |
|---|---|---|---|
| `rskew_30s`, `rskew_300s` | rolling skew of 1s log-returns | — | Asymmetry of recent returns. |
| `varratio_300s` | `Var(10s ret) / (10 · Var(1s ret))` over 300s | — | <1 mean-reverting, >1 trending. |
| `vol_ts_ratio` | `rv_5s / rv_300s` | — | Vol term structure (short vs long). |
| `vol_ts_ratio_mid` | `rv_30s / rv_300s` | — | Vol term structure (mid vs long). |

---

## Count by family

| Family | Count |
|---|---:|
| Pre-trade top-of-book | 4 |
| Momentum | 3 |
| Realized vol / amplitude | 6 |
| Top-of-book dynamics | 2 |
| Liquidation pressure (venue × window) | 24 |
| Time since last liquidation | 2 |
| Cascade acceleration | 2 |
| Cross-exchange divergence | 4 |
| Cross-exchange basis | 2 |
| Seasonality (incl. funding) | 4 |
| Tape-derived flow | 10 |
| Cascade dynamics | 5 |
| Regime descriptors | 5 |
| **Total** | **73** |

## Not features (meta / labels)
`NON_FEATURE_COLUMNS` (`features.py:43`) — excluded from the model matrix:
`timestamp`, `side`, `s` (sign), `price`, `notional`, `w` (= `min(notional, $100k)`,
the sample weight), `day`, `split`, `dt`, and the labels `pnl_30` / `pnl_120` /
`pnl_300` (the spec markout per horizon).

## Top features (permutation importance, τ=120 — see findings.md)
`bybit_liqabs_300s` (cascade size) ≫ `hour` > `bybit_liqalign_300s` (taker × Bybit
pressure) > `ampl_300s` (volatility amplitude) — i.e. Bybit liquidations, time-of-day,
and the volatility regime, confirming the cross-exchange-reversion thesis.

## Feature-selection findings (notebook `02_feature_selection.ipynb`, `make feature-nb`)

A per-`(symbol, τ)` study (missingness, univariate corr/MI, correlation clustering,
train→val permutation-importance stability, PCA, and a top-N validation-Score sweep):

- **Missingness is tiny.** Only `bybit_liqaccel` (~38%) and `binance_liqaccel` (~12–16%) are
  NaN-heavy (no liq in the 300 s window); every other feature is ~0% NaN. So sparsity is *not*
  what widens the importance error bars — redundancy and the noisy target are.
- **Univariate signal is weak.** Max |weighted corr| with markout ≈ **0.17**; most features ≪5%.
  The strongest single features are the **liquidation-alignment** (`*_liqalign`,
  `xexch_liqalign_*`) and **signed flow/return momentum** (`signed_vol_mom_*`, `ret_*_signed`)
  families — `binance_liqabs_30s` for ETH; brightest in BTC τ120, dimmer for ETH. The edge is
  interaction-driven, not marginal.
- **Redundancy is real.** Tight clusters (|r|>0.75): short-vol `{rv_5s, rv_30s, ampl_5s, ampl_30s}`,
  long-window activity `{rv_300s, ampl_300s, trade_intensity_30s/300s, book_chg_rate_30s}`, the
  bybit-liq blocks (`liqabs`≈`liqcnt`; `{bybit_liqabs_300s, bybit_liqcnt_300s, bybit_liqpress_300s,
  xexch_liqpress_300s}`; `xexch_*`≈`bybit_*` since Binance liq is tiny), and
  `dt_last_bybit_liq_s`≈`liq_lead_s`. One representative per block suffices; keeping all splits
  permutation credit → the wide error bars.
- **Train→val importance decays.** Most features sit well below the train=val diagonal —
  importance collapses out-of-sample (the overfitting signature behind the train-CV ≫ val gap),
  worst for ETH τ120/300.
- **PCA does not help.** 40/47 of 73 components are needed for 90%/95% variance (little
  compression), and the top PC↔target correlation is only ≈0.10 (and not PC1) — variance ≠
  signal. A tree gains nothing from decorrelation and loses interpretability → **don't use PCA**.
- **Optimal feature count is opposite by symbol** (top-N by val-stable importance, validation Score):

  | model | best val Score | at N | all-73 |
  |---|---|---|---|
  | BTC τ30 | **+1.44** | 5 | +0.49 |
  | BTC τ120 | +0.45 | 10 | +0.05 |
  | BTC τ300 | +0.22 | 15 | +0.12 |
  | ETH τ30 | +0.54 | 40 | +0.40 |
  | ETH τ120 | −0.04 | 40 | −0.29 |
  | ETH τ300 | −0.29 | 25 | −0.75 |

  **BTC Score falls as N grows** — it wants very few features (τ30 nearly triples at N=5). **ETH
  is the reverse** — Score is *worst* at small N and peaks around **N≈25–40** (small N *hurts* it,
  likely because ETH's near-zero val signal makes the importance ranking itself noisy). So there
  is **no single best N**: prune BTC aggressively (~5–15) but keep ETH rich (~25–40). A global
  `N_FEATURES` can't serve both (e.g. N=15 helps BTC but hurts ETH τ120) — the right knob is
  **per-`(sym,τ)` N**. PCA: no. ETH's τ120/300 weakness is **regime shift, not excess features** —
  address it with calibration / monotonic constraints / recency-weighting, not pruning.

### ⚠️ Caveat: selecting features on validation overfits it (val-selection leak)

A first cut built per-`(sym,τ)` `FEATURE_SETS` (redundancy-filtered to one feature per
|corr|>0.75 cluster, then top-N by **validation** permutation importance;
`scripts/select_features.py`) and retrained per-symbol on them. Result vs all-73:

| split | BTC τ30 | BTC τ120 | BTC τ300 | ETH τ30 | ETH τ120 | ETH τ300 |
|---|---|---|---|---|---|---|
| **validation** | 1.06→**1.18** | 0.64→**0.88** | 0.48→**1.71** | 0.70→0.36 | −0.82→**−3.06** | −7.06→−3.33 |
| **test (held out)** | 2.03→**0.99** | 3.25→**2.84** | 3.85→**2.33** | 2.69→2.28 | 3.90→3.76 | 6.00→**2.18** |

**Validation improves (BTC strongly) but the held-out April test degrades almost everywhere**
(ETH τ300 6.00→2.18). Because features were *ranked on validation*, the selection fit the
validation set — val rises because we optimised it, but the truly-out-of-sample test falls. (BTC
τ30 val rose while test fell — if it were underfitting, val would fall too → confirms val-overfit,
not too-few-features.) The dedup-first method is sound; the flaw is **selecting on val**.

**Decision (reverted):** ship **all features** per model; `config.FEATURE_SETS = {}`. The notebook
+ `select_features.py` are kept. Feature selection is **deferred** — redo it **leak-free**: rank
importance on a *train-internal* fold (split train into fit/selection), leaving val and test as
honest checks, then populate `FEATURE_SETS` and confirm on test. **→ Done; see next section.**

## Leak-free feature selection — chosen sets and why they help (2026-05-26)

The redo (`scripts/select_features.py`, `make feature-select`) ranks importance leak-free and is
judged on the walk-forward OOS harness — never on validation. Two estimators, two jobs (see
[[feature-selection]] memory): a **RANKER** (MSE-HGBR, permuted on the later-20% *train-internal
selection block*, embargoed) only *orders* features; a **JUDGE** (the deployed per-`(sym,τ)`
estimator via `model.fit_model`) picks N by a train-internal sweep + parsimony knee (`pick_n`). The
old hardcoded, val-derived `NMAP` is gone. Adoption is then decided **only** on the OOS gate
(`python scripts/walk_forward.py --specs features`: `shipped_all_features` vs
`shipped_curated_features`, identical except the feature set).

**OOS verdict — all-73 wins 5/6 cells** (mean Score over Feb/Mar/Apr, all-73 → curated):

| cell | all-73 | curated | Δ | curated worst-month |
|---|---|---|---|---|
| BTC τ30 | 1.61 | 1.48 | −0.13 | 1.11 |
| BTC τ120 | 2.14 | 1.55 | −0.59 | 0.03 |
| BTC τ300 | 1.99 | 1.24 | −0.74 | 0.36 |
| ETH τ30 | 2.78 | **2.83** | +0.05 | 0.91 (vs 0.55) |
| ETH τ120 | 2.72 | 2.04 | −0.68 | 0.58 |
| ETH τ300 | 2.70 | 1.69 | −1.01 | **−2.43** (breaks March) |

Only ETH τ30 passed, and only marginally (+0.05 mean — within 3-month noise — while giving up April
3.60→3.25). **Decision: keep all features everywhere; `config.FEATURE_SETS = {}`.** The train-internal
sweep had looked optimistic for BTC τ120/300 and ETH τ120/300 (curated > all internally) — exactly
the optimism the OOS gate exists to catch. Note too that the clean **BTC-few / ETH-many** shape from
the *validation* sweep above did **not** reproduce leak-free: the train-internal N-curves are flat and
noisy (weak signal), so that asymmetry was itself partly a validation artifact.

**Why all-73 is hard to beat (the "why they help" analysis,** `make feature-explain` →
`feature_explanations.parquet`, notebook §8). For each leak-free-chosen feature we measured its TRAIN
top−bottom-quintile edge (bps) and how many of the 6 calendar months keep that sign (regime
survival). The edge is **spread across many consistent-sign features**, so dropping ~half (even
keeping cluster representatives) loses signal without a variance payoff:

- **Liquidation alignment is the strongest, most regime-robust family** (the core thesis, leak-free):
  `bybit_liqalign_5s` edge **+1.7 / +2.3 / +3.1 / +4.2 bps** (BTC τ30 / BTC τ300 / ETH τ30 / ETH τ120),
  `binance_liqalign_{5,300}s` and `bybit_liqalign_30s` similar — almost all hold their sign **5–6/6
  months**. A liquidation aligned with the taker side reliably precedes the reversion the model keeps.
- **Signed momentum into the trade** is the other heavy hitter: `ret_30s_signed` +2.8 (ETH τ120, 6/6)
  / +3.5 (ETH τ300), `signed_vol_mom_{30,300}s` +1.9–3.5 (BTC τ300, ETH τ300), `ret_5s_signed` ~+2.9
  — large positive edges, 5–6/6 survival.
- **Cascade size / count**: `bybit_liqabs_300s` (+1.0 ETH τ30 6/6, +1.5 ETH τ300), `bybit_liqcnt_300s`,
  `binance_liqabs_300s` — the report's #1 feature, confirmed.
- **Volatility-regime gates carry a robust *negative* edge** (high vol ⇒ worse maker markout):
  `vol_ts_ratio` −0.7 (BTC τ30/120, 6/6), `ampl_5s` −0.6 (BTC τ30, 6/6); `ampl_300s` flips *positive*
  for ETH (+1.2/+1.4). Their sign-stability across all 6 months makes them reliable conditioners.
- **Microstructure** `px_vs_mid_bps` is small but unusually stable (6/6 in several cells).
- The features the selection *would* cut are a minority of flippy/weak ones — `trade_intensity_300s`,
  `basis_bps`, `bybit_liq_runlen`, `micro_signed_bps` (only **2/6** months consistent). Cutting them
  helps a little; cutting down to N≈25 also drops consistent-sign contributors, and the net is
  negative OOS. **The signal is broad, not concentrated** — so the all-73 tree, which can lean on the
  stable core and largely ignore the noise, is the right default. ETH τ300's curated 5-feature set is
  the cautionary case: stripped to 5 it *amplifies* the March regime sign-flip (worst month −2.43).
