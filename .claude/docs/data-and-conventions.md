# Data & conventions

## The four datasets (per symbol: `btc`, `eth`; 90 days 2025-12-01 → 2026-02-28 UTC)
| source (`config` key) | file | columns | rows (BTC / ETH) |
|---|---|---|---|
| `trades` | `data/binance_trades/perp_<sym>usdt.parquet` | timestamp, ticker, side, price, amount | 402M / 706M |
| `bbo` | `data/binance_booktickers/perp_<sym>usdt.parquet` | timestamp, ticker, bid_price, bid_amount, ask_price, ask_amount | 99M / 108M |
| `liq_binance` | `data/binance_liquidations/perp_<sym>usdt.parquet` | timestamp, ticker, side, price, amount | 114K / 132K |
| `liq_bybit` | `data/bybit_liquidations/<sym>usdt.parquet` | timestamp, ticker, side, price, amount | 229K / 160K |

`ticker` is `perp:btcusdt`/`perp:ethusdt` (Binance) or `btcusdt`/`ethusdt` (Bybit).
**BBO is top-of-book only** — no depth beyond level 1. No funding/OI data exists; any
feature must be computable from these four frames alone (the submission gets only these).

## The spec (from `description.md`, encoded in `config.py`)
- **Markout** for trade i, horizon τ: `m_i(τ)` = forward-filled Binance mid at `t_i+τ`;
  `s_i = +1` taker-buy / `-1` taker-sell; `w_i = min(notional_i, 100_000)`;
  `pnl_i(τ) = -s_i·(m_i(τ)-p_i)/p_i·1e4 + 0.5` (bps; +0.5 = maker rebate).
  If `t_i+τ` is beyond the last BBO tick, the trade is **excluded** (NaN).
- **Filter** `f_i(τ) ∈ {0,1}`, 1 = filter out.
- **Score** `= PnL_kept − PnL_all` (w-weighted means; maximise). Also report `PnL_filtered`.
- **Constraint** `KeptTurnoverPerDay = Σ(1-f)·w / n_days ≥ 500_000`.
- **Split:** train 2025-12-01..2026-01-31 (62 days), val 2026-02-01..2026-02-28 (28 days),
  test hidden (other dates).
- **Submission:** `signal(trades, bbo, liq_binance, liq_bybit) -> {30,120,300: 0/1 array}`,
  each length `len(trades)`. Called per symbol.

## Conventions that bite (each verified by hand — see notebook §5)
1. **Timestamps = int64 microseconds since UNIX epoch, UTC.** Divide by 1e6 for seconds.
   Sanity: `1764547200047000` → 2025-12-01 00:00:00.047. (ms → year ~57000; ns → 1970.)
2. **`side` means different things.** *trades*: taker side — a `buy` lifts the ask and
   prints **≥ mid 98%** of the time (maker sold). *liquidations*: the liquidation order
   side — `buy` = a short force-closed by buying = **upward** pressure.
3. **Bybit cross-exchange delay.** Treat Bybit events as available only **+200 ms** after
   their timestamp; shift before any Binance-time comparison. Empirically the two clocks
   are aligned (no large offset) — 200 ms is a network handicap, and it costs almost none
   of the (multi-minute) reversion signal. `io.liquidations_from_frame(..., 'bybit')`
   applies the shift.
4. **Bybit liquidations are NOT time-sorted** (≈0.03% backward steps, up to ~22 ms) and
   have thousands of same-µs collisions. **Always sort** before as-of joins (the loaders do).

## Data quality / quirks (notebook §2, §7)
- **Clean where it counts:** zero crossed/locked books, zero/negative prices, zero
  zero-size trades, zero NaN prices across 207M BBO rows and 1.1B trades.
- **BBO ~99.99% complete:** only 6 (BTC) / 9 (ETH) missing minutes in 90 days, on the
  *same* wall-clock minutes for both symbols (collector/venue outages, e.g. 2026-02-26 13:30–32).
- **Binance liquidations:** strictly monotonic, no dups/collisions.
- **Trades are bursty but never zero** (busiest minute 74–87× the median; min 29–58/min).
- **BBO feed is ~50 ms-coalesced** (≈14 updates/s; ~88% are size-only, price unchanged) —
  so trades, not BBO, set the finest time resolution; forward-filled mid has ~50 ms granularity.

## Distributions (notebook §4)
- Trade notional heavy-tailed/log-normal: median ≈$255 (BTC) / $40 (ETH); p99.99 ≈$0.5M/$0.33M.
  The $100k weight cap clips only the top ~1%.
- Spreads ≈1 tick almost always: BTC ~0.011 bps, ETH ~0.034 bps median.
- Trades ~50/50 taker buy/sell. Liquidations skew **sell-side** (long liquidations),
  Bybit most of all (~30% buy) — consistent with the BTC drawdown over the window.
- Market regime: BTC fell ~90k→~60k, ETH ~3450→~1750 over the quarter (a drawdown).
