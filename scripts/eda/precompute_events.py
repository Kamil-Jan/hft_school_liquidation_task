#!/usr/bin/env python
"""EDA pass 2: cross-source event studies (forward-filled Binance mid around events).

Writes per-liquidation markouts, average mid-response profiles, and a sampled
trade price-impact table for the exploration notebook.

Bybit convention: the RAW timestamp is offset 0 in the response profile (so the
charts line up across exchanges and the +200ms availability line is visible),
while the markout table measures from the shifted (available) time.

Usage:  python scripts/eda/precompute_events.py
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from liqsignal import config, io
from liqsignal.markout import forward_fill_mid

ART = config.ensure_artifacts()
US = config.US

OFFSETS_S = np.array([-60, -30, -20, -10, -5, -2, -1, -0.5, -0.2, -0.1, 0,
                      0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180, 240, 300])
HORIZONS = {"30": 30, "120": 120, "300": 300}


def liq_study(exchange: str, sym: str, bts: np.ndarray, bmid: np.ndarray) -> None:
    d = pl.read_parquet(config.dataset_path(f"liq_{exchange}", sym))
    t = d["timestamp"].to_numpy()
    side = d["side"].to_numpy()
    price = d["price"].to_numpy()
    notional = price * d["amount"].to_numpy()
    sgn = np.where(side == "buy", 1.0, -1.0)              # +1 = upward pressure
    t_ref = t + (config.BYBIT_DELAY_US if exchange == "bybit" else 0)

    m0, v0 = forward_fill_mid(bts, bmid, t_ref)
    cols = dict(timestamp=t, side=side, price=price, notional=notional, m0=m0)
    for name, hs in HORIZONS.items():
        mh, vh = forward_fill_mid(bts, bmid, t_ref + hs * US)
        ret = sgn * (mh - m0) / m0 * 1e4
        ret[~(v0 & vh)] = np.nan
        cols[f"ret_{name}"] = ret
    pl.DataFrame(cols).write_parquet(ART / f"liq_markout_{exchange}_{sym}.parquet")

    m0_raw, v0_raw = forward_fill_mid(bts, bmid, t)       # raw-time reference for profile
    prof = []
    for off in OFFSETS_S:
        mo, vo = forward_fill_mid(bts, bmid, t + int(off * US))
        rel = (mo - m0_raw) / m0_raw * 1e4
        for sd in ("buy", "sell"):
            mask = (side == sd) & v0_raw & vo
            if mask.sum():
                prof.append(dict(offset_s=float(off), side=sd, n=int(mask.sum()),
                                 mean_bps=float(np.nanmean(rel[mask])),
                                 median_bps=float(np.nanmedian(rel[mask]))))
    pl.DataFrame(prof).write_parquet(ART / f"liq_profile_{exchange}_{sym}.parquet")
    print(f"  [liq {exchange} {sym}] n={len(t):,} buy_frac={(side=='buy').mean():.3f}")


def trade_study(sym: str, bts: np.ndarray, bmid: np.ndarray, n_sample: int = 300_000) -> None:
    s, _ = io.sample_trades(sym, n_sample)
    t = s["timestamp"].to_numpy()
    side = s["side"].to_numpy()
    price = s["price"].to_numpy()
    notional = price * s["amount"].to_numpy()
    sgn = np.where(side == "buy", 1.0, -1.0)

    m_pre, _ = forward_fill_mid(bts, bmid, t)
    out = dict(timestamp=t, side=side, price=price, notional=notional, m_pre=m_pre,
               px_vs_mid_bps=sgn * (price - m_pre) / m_pre * 1e4)
    for off in (-1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5, 10, 30):
        mo, _ = forward_fill_mid(bts, bmid, t + int(off * US))
        out[f"imp_{off}"] = sgn * (mo - m_pre) / m_pre * 1e4
    pl.DataFrame(out).write_parquet(ART / f"trade_impact_{sym}.parquet")
    buy = side == "buy"
    print(f"  [trades {sym}] sample={len(t):,} "
          f"buy@>=mid={(price[buy] >= m_pre[buy]).mean():.3f}")


def main() -> None:
    t0 = time.time()
    for sym in config.SYMBOLS:
        book = io.load_book_top(sym)
        bts, bmid = book.ts, book.mid
        for exch in ("binance", "bybit"):
            liq_study(exch, sym, bts, bmid)
        trade_study(sym, bts, bmid)
        del book
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
