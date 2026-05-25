"""Feature engineering for the maker-trade filter.

Three layers:

* small, pure functions for each feature (top-of-book imbalance, microprice
  adjustment, signed momentum, realized vol / amplitude, windowed liquidation
  pressure, cross-exchange basis, recency, seasonality) — each testable alone;
* :class:`FeatureContext` + :func:`compute_features`, which evaluate the whole
  feature set for an arbitrary set of trade times against shared, pre-built BBO /
  liquidation arrays (the context is built once and reused across batches, so the
  same code serves both panel building and the chunked submission path); and
* :func:`build_feature_panel`, which draws a per-symbol trade sample, attaches the
  spec markout, and returns a model-ready panel.

Feature design follows ``research_papers/``: order-book imbalance and its
contrarian/"reversal" reading and the microprice (Albers, "Market Maker's
Dilemma"); trade-flow / mean-divergence and cross-exchange lead (Albers,
"Fragmentation"); transient-impact exhaustion / momentum (Lillo). Everything is
computable from the four submission frames alone (top-of-book BBO, trades,
Binance & Bybit liquidations) — no depth, funding, or open interest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from . import config, io
from .io import BookTop, Liquidations
from .markout import compute_markout, last_index_at, trade_sign

US = config.US

# Lookback windows (seconds).
MOMENTUM_WINDOWS_S: tuple[int, ...] = (1, 5, 30)
REALIZED_WINDOWS_S: tuple[int, ...] = (5, 30, 300)
LIQUIDITY_WINDOWS_S: tuple[int, ...] = (5, 30, 300)
FLOW_WINDOWS_S: tuple[int, ...] = (30, 300)   # tape-flow feature windows
BASIS_MAX_STALE_S: float = 300.0   # ignore the Bybit basis proxy if no liq within this

# Columns that are NOT model features (meta / label / identifiers).
NON_FEATURE_COLUMNS: frozenset[str] = frozenset({
    "timestamp", "side", "s", "price", "notional", "w", "day", "split", "dt",
    "pnl_30", "pnl_120", "pnl_300",
})


# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------
def order_book_imbalance(bid_amount: np.ndarray, ask_amount: np.ndarray) -> np.ndarray:
    """(bid - ask) / (bid + ask) in [-1, 1]; +1 = bid-heavy (latent buy pressure)."""
    total = bid_amount + ask_amount
    return np.where(total > 0, (bid_amount - ask_amount) / total, 0.0)


def microprice_adjustment_bps(spread: np.ndarray, mid: np.ndarray, obi: np.ndarray) -> np.ndarray:
    """(microprice - mid) in bps. For a 1-tick book microprice = mid + (spread/2)·OBI,
    so this is the OBI tilt scaled by the half-spread (matters most when spread widens)."""
    return (spread / 2.0) / mid * obi * 1e4


def signed_distance_to_mid(price: np.ndarray, mid: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Trade aggressiveness: signed (price - mid)/mid in bps. >0 = paid up to cross."""
    return sign * (price - mid) / mid * 1e4


def signed_return_bps(mid_now: np.ndarray, mid_prev: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Momentum into the trade: signed mid return over a lookback, in bps."""
    return sign * (mid_now - mid_prev) / mid_prev * 1e4


def windowed_liq(liq: Liquidations, query_ts: np.ndarray,
                 window_us: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Net signed notional, absolute notional (cascade size) and event count in ``(t-window, t]``."""
    net_cs = np.concatenate([[0.0], np.cumsum(liq.signed_notional)])
    abs_cs = np.concatenate([[0.0], np.cumsum(np.abs(liq.signed_notional))])
    hi = np.searchsorted(liq.ts, query_ts, side="right")
    lo = np.searchsorted(liq.ts, query_ts - window_us, side="right")
    net = net_cs[hi] - net_cs[lo]
    absn = abs_cs[hi] - abs_cs[lo]
    count = (hi - lo).astype(np.int32)
    return net, absn, count


def cascade_acceleration(cnt_short: np.ndarray, cnt_long: np.ndarray,
                         short_s: float, long_s: float) -> np.ndarray:
    """Liquidation burst-rate ratio ``(cnt_short/short_s) / (cnt_long/long_s)``.

    ``>1`` ⇒ liquidations are accelerating into the trade (recent burst denser than
    the slower baseline); ``~1`` ⇒ steady. NaN where the long window has no events
    (handled natively by the GBM, per the project's missing-value convention).
    """
    rate_short = cnt_short / short_s
    rate_long = cnt_long / long_s
    out = np.full(len(cnt_short), np.nan, dtype=np.float64)
    nz = rate_long > 0
    out[nz] = rate_short[nz] / rate_long[nz]
    return out


def liq_run_length(side: np.ndarray) -> np.ndarray:
    """Signed run-length of consecutive same-side liquidations ending at each event.

    `+k` for a run of k consecutive buy-side liqs, `−k` for sell-side. Vectorised.
    """
    n = len(side)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    change = np.concatenate([[True], side[1:] != side[:-1]])
    idx = np.arange(n)
    last_change = np.where(change, idx, 0)
    np.maximum.accumulate(last_change, out=last_change)
    run = (idx - last_change + 1).astype(np.float64)
    return np.where(side == "buy", run, -run)


def liq_zscore(abs_notional: np.ndarray, window: int = 200, min_periods: int = 20) -> np.ndarray:
    """Rolling z-score of each liquidation's |notional| vs the trailing-event distribution.

    NaN until ``min_periods`` events have accrued (handled natively by the GBM).
    """
    if len(abs_notional) == 0:
        return np.zeros(0, dtype=np.float64)
    s = pd.Series(abs_notional)
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - m) / sd).to_numpy()


def windowed_flow_sums(cs_signed: np.ndarray, cs_tot: np.ndarray, cs_cnt: np.ndarray,
                       s0: int, n: int, query_ts: np.ndarray, window_s: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Windowed (net signed vol, total vol, count) over the ``window_s`` whole seconds
    *strictly before* each query time, from contiguous-grid prefix sums.

    Excluding the trade's own (partial) second avoids intra-second look-ahead. Returns
    ``(net, tot, cnt, valid)``; ``valid`` is False where there is no in-grid history.
    """
    qsec = (np.asarray(query_ts) // US).astype(np.int64)
    hi = qsec - s0                      # prefix index up to (not incl.) the current second
    lo = hi - window_s
    valid = (lo >= 0) & (hi <= n)
    hi_c = np.clip(hi, 0, n); lo_c = np.clip(lo, 0, n)
    net = cs_signed[hi_c] - cs_signed[lo_c]
    tot = cs_tot[hi_c] - cs_tot[lo_c]
    cnt = cs_cnt[hi_c] - cs_cnt[lo_c]
    return net, tot, cnt, valid


def seconds_since_last(event_ts: np.ndarray, query_ts: np.ndarray,
                       default: float = np.nan) -> np.ndarray:
    """Seconds since the most recent event at-or-before each query time.

    ``default`` (NaN by default = "missing", handled natively by the GBM) is used
    where no event precedes the query — avoids a giant sentinel polluting stats.
    """
    if len(event_ts) == 0:
        return np.full(len(query_ts), default)
    idx = np.searchsorted(event_ts, query_ts, side="right") - 1
    out = np.full(len(query_ts), default, dtype=np.float64)
    ok = idx >= 0
    out[ok] = (query_ts[ok] - event_ts[idx[ok]]) / US
    return out


def book_state_at(book: BookTop, query_ts: np.ndarray):
    """Forward-filled (mid, spread, bid_amount, ask_amount) at each query time; NaN mid if before book."""
    idx, valid = last_index_at(book.ts, query_ts)
    mid = book.mid[idx].astype(np.float64)
    spread = book.spread[idx].astype(np.float64)
    bid = book.bid_amount[idx].astype(np.float64)
    ask = book.ask_amount[idx].astype(np.float64)
    mid[~valid] = np.nan
    return mid, spread, bid, ask


def basis_proxy_bps(liq_bybit: Liquidations, query_ts: np.ndarray, mid: np.ndarray,
                    max_stale_s: float = BASIS_MAX_STALE_S) -> np.ndarray:
    """Cross-exchange divergence proxy: (last recent Bybit liq price - Binance mid)/mid in bps.

    A mean-divergence stand-in (paper 2) given we only have Bybit liquidation prints,
    not its book. Zeroed when the last Bybit liquidation is older than ``max_stale_s``.
    """
    if len(liq_bybit.ts) == 0:
        return np.zeros(len(query_ts))
    idx = np.searchsorted(liq_bybit.ts, query_ts, side="right") - 1
    out = np.zeros(len(query_ts), dtype=np.float64)
    ok = idx >= 0
    fresh = ok.copy()
    fresh[ok] &= (query_ts[ok] - liq_bybit.ts[idx[ok]]) <= max_stale_s * US
    last_price = liq_bybit.price[np.clip(idx, 0, len(liq_bybit.price) - 1)]
    out[fresh] = (last_price[fresh] - mid[fresh]) / mid[fresh] * 1e4
    return out


def hour_of_day(ts: np.ndarray) -> np.ndarray:
    """Hour of day in UTC [0, 24)."""
    return ((ts // (3600 * US)) % 24).astype(np.float64)


def is_weekend(ts: np.ndarray) -> np.ndarray:
    """1.0 on Sat/Sun (UTC). Unix epoch day 0 = Thursday, so Monday=0 ⇒ +3 offset."""
    dow = ((ts // config.DAY_US) + 3) % 7
    return (dow >= 5).astype(np.float64)


FUNDING_PERIOD_S: int = 8 * 3600          # Binance USDT-perp funds every 8h: 00/08/16 UTC


def minutes_to_funding(ts: np.ndarray) -> np.ndarray:
    """Minutes until the next 8-hour funding mark (00:00/08:00/16:00 UTC), in [0, 480)."""
    sec_in = (ts // US) % FUNDING_PERIOD_S
    return ((FUNDING_PERIOD_S - sec_in) % FUNDING_PERIOD_S).astype(np.float64) / 60.0


def in_funding_window(ts: np.ndarray, margin_min: float = 5.0) -> np.ndarray:
    """1.0 within ``margin_min`` of a funding mark (just before or just after)."""
    m = minutes_to_funding(ts)
    period_min = FUNDING_PERIOD_S / 60.0
    return ((m <= margin_min) | (m >= period_min - margin_min)).astype(np.float64)


# ---------------------------------------------------------------------------
# 1-second mid grid for realized volatility / amplitude
# ---------------------------------------------------------------------------
def _second_grid(book: BookTop) -> tuple[int, np.ndarray]:
    """Contiguous 1s grid of the last mid per second (forward-filled). Returns (start_sec, mid_grid)."""
    sec = (book.ts // US).astype(np.int64)
    g = (pl.DataFrame({"sec": sec, "mid": book.mid})
         .group_by("sec").agg(pl.col("mid").last()).sort("sec"))
    s0, s1 = int(g["sec"][0]), int(g["sec"][-1])
    mid_grid = np.full(s1 - s0 + 1, np.nan)
    mid_grid[g["sec"].to_numpy() - s0] = g["mid"].to_numpy()
    fill = np.where(~np.isnan(mid_grid), np.arange(len(mid_grid)), 0)
    np.maximum.accumulate(fill, out=fill)        # forward-fill gaps
    return s0, mid_grid[fill]


def _grid_vol_range(mid_grid: np.ndarray, windows_s: tuple[int, ...]
                    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Rolling realized vol (std of 1s log-returns, bps) and amplitude ((max-min)/mid, bps)."""
    s = pd.Series(mid_grid)
    ret = np.log(s).diff()
    vol, ampl = {}, {}
    for w in windows_s:
        vol[w] = (ret.rolling(w).std() * 1e4).to_numpy()
        ampl[w] = ((s.rolling(w).max() - s.rolling(w).min()) / s * 1e4).to_numpy()
    return vol, ampl


def _grid_regime(mid_grid: np.ndarray, skew_windows_s: tuple[int, ...] = (30, 300),
                 vr_window_s: int = 300, vr_k: int = 10) -> dict[str, np.ndarray]:
    """Regime descriptors on the 1s mid grid: rolling realized skew of 1s log-returns,
    and a variance ratio (Var(k-step ret) / (k · Var(1-step ret)) — <1 mean-reverting,
    >1 trending). All rolling, computed once; looked up per trade like vol/amplitude."""
    s = pd.Series(mid_grid)
    ret1 = np.log(s).diff()
    out: dict[str, np.ndarray] = {}
    for w in skew_windows_s:
        out[f"rskew_{w}"] = ret1.rolling(w).skew().to_numpy()
    retk = np.log(s).diff(vr_k)
    var1 = ret1.rolling(vr_window_s).var().to_numpy()
    vark = retk.rolling(vr_window_s).var().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = vark / (vr_k * var1)
    vr[~np.isfinite(vr)] = np.nan
    out[f"varratio_{vr_window_s}"] = vr
    return out


# ---------------------------------------------------------------------------
# Feature context (built once; reused across trade batches)
# ---------------------------------------------------------------------------
@dataclass
class FeatureContext:
    book: BookTop
    liq: dict[str, Liquidations]
    grid_s0: int
    grid_vol: dict[int, np.ndarray]
    grid_ampl: dict[int, np.ndarray]
    change_ts: np.ndarray   # timestamps where the mid changed (for book "age")
    grid_regime: dict[str, np.ndarray]              # rolling skew / variance ratio on the mid grid
    liq_runlen: dict[str, np.ndarray]               # per-venue signed cascade run-length, aligned to liq.ts
    liq_z: dict[str, np.ndarray]                    # per-venue cascade |notional| z-score, aligned to liq.ts
    flow: "io.FlowGrid | None" = None               # 1s trade-flow grid (None ⇒ flow features are NaN)
    flow_cs_signed: np.ndarray | None = None        # prefix sums for windowed flow lookups
    flow_cs_tot: np.ndarray | None = None
    flow_cs_cnt: np.ndarray | None = None


def build_context(book: BookTop, liq_binance: Liquidations, liq_bybit: Liquidations,
                  flow: "io.FlowGrid | None" = None) -> FeatureContext:
    s0, mid_grid = _second_grid(book)
    vol, ampl = _grid_vol_range(mid_grid, REALIZED_WINDOWS_S)
    regime = _grid_regime(mid_grid)
    change_ts = book.ts[1:][np.diff(book.mid) != 0]

    liq = {"binance": liq_binance, "bybit": liq_bybit}
    runlen = {e: liq_run_length(liq[e].side) for e in liq}
    zsc = {e: liq_zscore(np.abs(liq[e].signed_notional)) for e in liq}

    cs_s = cs_t = cs_c = None
    if flow is not None:
        cs_s = np.concatenate([[0.0], np.cumsum(flow.signed_vol)])
        cs_t = np.concatenate([[0.0], np.cumsum(flow.tot_vol)])
        cs_c = np.concatenate([[0.0], np.cumsum(flow.cnt)])

    return FeatureContext(book, liq, s0, vol, ampl, change_ts, regime, runlen, zsc,
                          flow, cs_s, cs_t, cs_c)


def compute_features(ctx: FeatureContext, trade_ts: np.ndarray, sign: np.ndarray,
                     price: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate the full feature set for the given trades against a prepared context."""
    book = ctx.book
    feats: dict[str, np.ndarray] = {}

    # --- pre-trade top-of-book
    mid_pre, spread_pre, bid_pre, ask_pre = book_state_at(book, trade_ts)
    obi = order_book_imbalance(bid_pre, ask_pre)
    feats["obi"] = obi
    feats["obi_signed"] = sign * obi
    feats["micro_signed_bps"] = sign * microprice_adjustment_bps(spread_pre, mid_pre, obi)
    feats["px_vs_mid_bps"] = signed_distance_to_mid(price, mid_pre, sign)

    # --- momentum into the trade
    for w in MOMENTUM_WINDOWS_S:
        mid_prev, _, _, _ = book_state_at(book, trade_ts - w * US)
        feats[f"ret_{w}s_signed"] = signed_return_bps(mid_pre, mid_prev, sign)

    # --- realized volatility / amplitude (grid lookup; regime gates, unsigned)
    gi = np.clip((trade_ts // US) - ctx.grid_s0, 0, len(next(iter(ctx.grid_vol.values()))) - 1)
    before_grid = (trade_ts // US) < ctx.grid_s0
    for w in REALIZED_WINDOWS_S:
        rv = ctx.grid_vol[w][gi]; am = ctx.grid_ampl[w][gi]
        rv[before_grid] = np.nan; am[before_grid] = np.nan
        feats[f"rv_{w}s"] = rv
        feats[f"ampl_{w}s"] = am

    # --- regime descriptors: vol term-structure + grid skew / variance ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        vtr = feats["rv_5s"] / feats["rv_300s"]
        vtrm = feats["rv_30s"] / feats["rv_300s"]
    vtr[~np.isfinite(vtr)] = np.nan; vtrm[~np.isfinite(vtrm)] = np.nan
    feats["vol_ts_ratio"] = vtr          # short/long realized-vol ratio (term structure)
    feats["vol_ts_ratio_mid"] = vtrm
    for k, arr in ctx.grid_regime.items():
        v = arr[gi].copy(); v[before_grid] = np.nan
        feats[k] = v                     # rskew_30, rskew_300, varratio_300

    # --- top-of-book dynamics
    feats["book_age_s"] = seconds_since_last(ctx.change_ts, trade_ts)
    hi = np.searchsorted(ctx.change_ts, trade_ts, side="right")
    lo = np.searchsorted(ctx.change_ts, trade_ts - 30 * US, side="right")
    feats["book_chg_rate_30s"] = (hi - lo).astype(np.float64) / 30.0

    # --- liquidation pressure / cascade size / alignment with the taker, per venue & window
    for exch in ("binance", "bybit"):
        liq = ctx.liq[exch]
        for w in LIQUIDITY_WINDOWS_S:
            net, absn, cnt = windowed_liq(liq, trade_ts, w * US)
            feats[f"{exch}_liqpress_{w}s"] = net
            feats[f"{exch}_liqabs_{w}s"] = absn
            feats[f"{exch}_liqcnt_{w}s"] = cnt.astype(np.float64)
            feats[f"{exch}_liqalign_{w}s"] = sign * net   # taker trades with/against the pressure
        feats[f"dt_last_{exch}_liq_s"] = seconds_since_last(liq.ts, trade_ts)
        # cascade acceleration: is the burst speeding up (30s vs 300s rate)?
        feats[f"{exch}_liqaccel"] = cascade_acceleration(
            feats[f"{exch}_liqcnt_30s"], feats[f"{exch}_liqcnt_300s"], 30.0, 300.0)
        # deeper cascade dynamics: signed run-length + cascade-size z (last event ≤ t)
        idxl = np.searchsorted(liq.ts, trade_ts, side="right") - 1
        okl = idxl >= 0
        rl = np.full(len(trade_ts), np.nan); rl[okl] = ctx.liq_runlen[exch][idxl[okl]]
        feats[f"{exch}_liq_runlen"] = rl
        zz = np.full(len(trade_ts), np.nan); zz[okl] = ctx.liq_z[exch][idxl[okl]]
        feats[f"{exch}_liqz"] = zz

    # --- cross-exchange liquidation divergence (Bybit leads Binance — core thesis)
    for w in (30, 300):
        div = feats[f"bybit_liqpress_{w}s"] - feats[f"binance_liqpress_{w}s"]
        feats[f"xexch_liqpress_{w}s"] = div
        feats[f"xexch_liqalign_{w}s"] = sign * div   # taker aligned with the Bybit-vs-Binance gap
    # Bybit→Binance lead-lag: >0 ⇒ the Bybit liq is more recent (Bybit leads)
    feats["liq_lead_s"] = feats["dt_last_binance_liq_s"] - feats["dt_last_bybit_liq_s"]

    # --- tape-derived flow features (1s trade-flow grid; prefix-sum windows)
    nrow = len(trade_ts)
    for w in FLOW_WINDOWS_S:
        if ctx.flow is None:
            net = tot = cnt = np.zeros(nrow); valid = np.zeros(nrow, dtype=bool)
        else:
            net, tot, cnt, valid = windowed_flow_sums(
                ctx.flow_cs_signed, ctx.flow_cs_tot, ctx.flow_cs_cnt,
                ctx.flow.s0, ctx.flow.n, trade_ts, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            tfi = np.where(tot > 0, net / tot, 0.0)
            mag = np.where(tot > 0, np.abs(net) / tot, 0.0)
        intens = cnt / w
        svm = sign * net
        for arr in (tfi, mag, intens, svm):
            arr[~valid] = np.nan
        feats[f"tfi_{w}s"] = tfi
        feats[f"tfi_aligned_{w}s"] = sign * tfi          # taker aligned with the flow imbalance
        feats[f"trade_intensity_{w}s"] = intens          # trades/sec (regime)
        feats[f"flow_imbalance_mag_{w}s"] = mag          # |imbalance| (VPIN-lite toxicity)
        feats[f"signed_vol_mom_{w}s"] = svm              # taker-aligned signed volume

    # --- cross-exchange basis proxy + seasonality
    basis = basis_proxy_bps(ctx.liq["bybit"], trade_ts, mid_pre)
    feats["basis_bps"] = basis
    feats["basis_signed_bps"] = sign * basis
    feats["hour"] = hour_of_day(trade_ts)
    feats["is_weekend"] = is_weekend(trade_ts)
    feats["min_to_funding"] = minutes_to_funding(trade_ts)
    feats["in_funding_window"] = in_funding_window(trade_ts)
    return feats


def feature_columns(panel_columns) -> list[str]:
    """Model feature columns = panel columns minus meta/label columns."""
    return [c for c in panel_columns if c not in NON_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------
def build_feature_panel(symbol: str, *, target_rows: int = 3_000_000,
                        taus: tuple[int, ...] = config.TAUS) -> tuple[pl.DataFrame, int]:
    """Assemble a sampled feature+markout panel for ``symbol``.

    Returns ``(panel, step)`` where ``step`` is the trade-sampling factor (needed
    to rescale turnover during scoring).
    """
    from .splits import split_expr

    book = io.load_book_top(symbol)
    ctx = build_context(book, io.load_liquidations("binance", symbol),
                        io.load_liquidations("bybit", symbol),
                        flow=io.load_flow_grid(symbol))
    sample, step = io.sample_trades(symbol, target_rows)

    t = sample["timestamp"].to_numpy()
    side = sample["side"].to_numpy()
    price = sample["price"].to_numpy()
    amount = sample["amount"].to_numpy()
    sign = trade_sign(side)
    notional = price * amount

    cols: dict[str, np.ndarray] = {
        "timestamp": t, "side": side, "s": sign, "price": price,
        "notional": notional, "w": np.minimum(notional, config.NOTIONAL_CAP),
        "day": (t // config.DAY_US),
    }
    for tau in taus:
        cols[f"pnl_{tau}"] = compute_markout(t, sign, price, book.ts, book.mid, tau)
    cols.update(compute_features(ctx, t, sign, price))

    panel = pl.DataFrame(cols).with_columns(split_expr())
    return panel, step
