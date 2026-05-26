#!/usr/bin/env python
"""EDA pass 3: anatomy of a liquidation cascade (begin / middle / end).

A *cascade* is a cluster of liquidations (both venues, Bybit shifted +200 ms to its
available time) separated by gaps < ``GAP_S`` and containing >= ``MIN_EVENTS`` prints.
We anchor at the cascade **begin** (first print) and, in event time, average over all
cascades what the **Binance microstructure** does — so we can see what is worth watching
*before*, *during*, and *after* a cascade:

  * **mid move** in the pressure direction (bps, rel. to the mid at begin),
  * **trade intensity** (trades/s) and **signed taker flow** (pressure-signed vol/s) from
    the 1 s trade-flow grid,
  * **BBO** spread (bps), top-of-book depth ($), and book imbalance (pressure-signed),
  * per-venue liquidation **notional** (Bybit vs Binance).

Outputs (per symbol):
  * ``cascade_profile_<sym>.parquet`` — every series in fixed 5 s bins from −60 s to +300 s.
  * ``cascade_phase_<sym>.parquet``   — the same series + the realized maker markout
    ``pnl_120`` averaged over the sampled-panel trades in five phases (pre/begin/middle/end/after).
  * ``cascade_meta_<sym>.parquet``    — cascade count, sell fraction, median climax/end offsets.

Usage:  python scripts/eda/precompute_cascades.py
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from liqsignal import analysis, config, io
from liqsignal.markout import forward_fill_mid

ART = config.ensure_artifacts()
US = config.US

GAP_S = 10              # a gap > this (s) between prints starts a new cascade
MIN_EVENTS = 5          # a cascade needs at least this many liquidation prints
BIN_S = 5               # event-time resolution of the profile (s)
LO_S, HI_S = -60, 300   # profile window around begin (s)
# windows bracket the typical cascade (median span ≈ 13 s)
PHASES = ["pre", "begin", "middle", "end", "after"]
PHASE_WIN = {"pre": (-30, 0), "begin": (0, 5), "middle": (5, 12),
             "end": (12, 30), "after": (30, 300)}
PHASE_CTR = {"pre": -12.0, "begin": 2.5, "middle": 8.0, "end": 18.0, "after": 120.0}
FEATS = ["rv_300s", "ampl_300s", "bybit_liqabs_300s", "binance_liqabs_300s"]


def _load_events(sym: str):
    """Union of both venues' liquidations in *available* time, sorted.
    Returns (t_us, is_bybit, notional, sign)."""
    parts = []
    for exch, shift in (("binance", 0), ("bybit", config.BYBIT_DELAY_US)):
        d = pl.read_parquet(config.dataset_path(f"liq_{exch}", sym))
        t = d["timestamp"].to_numpy() + shift
        notional = (d["price"] * d["amount"]).to_numpy()
        sgn = np.where(d["side"].to_numpy() == "buy", 1.0, -1.0)
        parts.append((t, np.full(t.shape, exch == "bybit"), notional, sgn))
    t = np.concatenate([p[0] for p in parts])
    order = np.argsort(t, kind="stable")
    return (t[order], np.concatenate([p[1] for p in parts])[order],
            np.concatenate([p[2] for p in parts])[order],
            np.concatenate([p[3] for p in parts])[order])


def _cascades(t, notional, signed):
    if len(t) == 0:
        return
    brk = np.where(np.diff(t) > GAP_S * US)[0] + 1
    for lo, hi in zip(np.r_[0, brk], np.r_[brk, len(t)]):
        if hi - lo < MIN_EVENTS:
            continue
        yield {"lo": int(lo), "hi": int(hi), "begin": int(t[lo]), "end": int(t[hi - 1]),
               "dominant": 1.0 if signed[lo:hi].sum() >= 0 else -1.0}


def _dense_flow(sym: str):
    """Dense per-second trade-flow arrays + prefix sums for windowed (rate) lookups."""
    g = pl.read_parquet(ART / f"flow_grid_{sym}.parquet").sort("sec")
    sec = g["sec"].to_numpy(); s0 = int(sec[0]); n = int(sec[-1]) - s0 + 1
    cnt = np.zeros(n); vol = np.zeros(n); sig = np.zeros(n)
    idx = sec - s0
    cnt[idx] = g["cnt"].to_numpy(); vol[idx] = g["tot_vol"].to_numpy(); sig[idx] = g["signed_vol"].to_numpy()
    return s0, np.r_[0.0, cnt.cumsum()], np.r_[0.0, vol.cumsum()], np.r_[0.0, sig.cumsum()]


def _bbo_series(book, idx, dominant):
    """spread (bps), top-of-book depth ($), pressure-signed book imbalance at indices ``idx``."""
    mid = book.mid[idx]; bid = book.bid_amount[idx].astype(float); ask = book.ask_amount[idx].astype(float)
    spread_bps = book.spread[idx] / mid * 1e4
    depth_usd = (bid + ask) * mid
    obi = (bid - ask) / np.maximum(bid + ask, 1e-9)
    return spread_bps, depth_usd, dominant * obi


def _window_rate(cs, s0, begin_sec, lo_s, hi_s):
    """Per-second rate of a cumulative-sum series over [begin+lo, begin+hi)."""
    a = int(np.clip(begin_sec - s0 + lo_s, 0, len(cs) - 1))
    b = int(np.clip(begin_sec - s0 + hi_s, 0, len(cs) - 1))
    return (cs[b] - cs[a]) / max(hi_s - lo_s, 1)


def process(sym: str, book) -> int:
    ev = _load_events(sym)
    casc = list(_cascades(ev[0], ev[2], ev[3]))
    t_ev, by_ev, n_ev = ev[0], ev[1], ev[2]
    s0, cs_cnt, cs_vol, cs_sig = _dense_flow(sym)
    centers = np.arange(LO_S, HI_S, BIN_S) + BIN_S / 2.0
    nb = len(centers)

    # accumulators (profile, by offset bin) and per-phase
    acc = {k: np.zeros(nb) for k in ("mid", "spread", "depth", "obi", "trate", "flow", "bn", "yn")}
    cnts = {k: np.zeros(nb) for k in ("mid", "bbo")}
    ph = {p: {"mid": [0.0, 0], "spread": [0.0, 0], "depth": [0.0, 0], "obi": [0.0, 0],
              "trate": 0.0, "flow": 0.0, "bn": 0.0, "yn": 0.0, "n": 0} for p in PHASES}
    edges = np.arange(LO_S, HI_S + BIN_S, BIN_S)
    climax_off, end_off, sells, n = [], [], 0, 0

    for c in casc:
        m0, v0 = forward_fill_mid(book.ts, book.mid, np.array([c["begin"]]))
        if not v0[0]:
            continue
        begin_sec = c["begin"] // US
        qt = c["begin"] + (centers * US).astype(np.int64)
        # mid displacement (pressure-signed)
        mids, valid = forward_fill_mid(book.ts, book.mid, qt)
        disp = c["dominant"] * (mids - m0[0]) / m0[0] * 1e4
        acc["mid"][valid] += disp[valid]; cnts["mid"][valid] += 1
        # BBO series at each offset
        idx = np.searchsorted(book.ts, qt, side="right") - 1
        ok = idx >= 0
        sp, dp, ob = _bbo_series(book, np.clip(idx, 0, len(book.ts) - 1), c["dominant"])
        acc["spread"][ok] += sp[ok]; acc["depth"][ok] += dp[ok]; acc["obi"][ok] += ob[ok]; cnts["bbo"][ok] += 1
        # trade-flow rate per offset bin (windowed over the bin)
        for k, lo_e, hi_e in zip(range(nb), edges[:-1], edges[1:]):
            acc["trate"][k] += _window_rate(cs_cnt, s0, begin_sec, int(lo_e), int(hi_e))
            acc["flow"][k] += c["dominant"] * _window_rate(cs_sig, s0, begin_sec, int(lo_e), int(hi_e))
        # per-venue liq notional by offset bin
        off = (t_ev[c["lo"]:c["hi"]] - c["begin"]) / US
        vb = by_ev[c["lo"]:c["hi"]].astype(bool); nt = n_ev[c["lo"]:c["hi"]]
        acc["bn"] += np.histogram(off[~vb], edges, weights=nt[~vb])[0]
        acc["yn"] += np.histogram(off[vb], edges, weights=nt[vb])[0]
        # phase aggregates
        for p in PHASES:
            lo_w, hi_w = PHASE_WIN[p]; ctr = PHASE_CTR[p]
            qc = np.array([c["begin"] + int(ctr * US)])
            mc, vc = forward_fill_mid(book.ts, book.mid, qc)
            if vc[0]:
                ph[p]["mid"][0] += c["dominant"] * (mc[0] - m0[0]) / m0[0] * 1e4; ph[p]["mid"][1] += 1
            ic = np.searchsorted(book.ts, qc[0], side="right") - 1
            if ic >= 0:
                sp1, dp1, ob1 = _bbo_series(book, np.array([ic]), c["dominant"])
                ph[p]["spread"][0] += sp1[0]; ph[p]["spread"][1] += 1
                ph[p]["depth"][0] += dp1[0]; ph[p]["obi"][0] += ob1[0]
            ph[p]["trate"] += _window_rate(cs_cnt, s0, begin_sec, lo_w, hi_w)
            ph[p]["flow"] += c["dominant"] * _window_rate(cs_sig, s0, begin_sec, lo_w, hi_w)
            m = (off >= lo_w) & (off < hi_w)
            ph[p]["bn"] += nt[m & ~vb].sum(); ph[p]["yn"] += nt[m & vb].sum(); ph[p]["n"] += 1
        hist = np.histogram(off, edges, weights=nt)[0]
        climax_off.append(centers[hist.argmax()]); end_off.append((c["end"] - c["begin"]) / US)
        sells += (c["dominant"] < 0); n += 1

    cm = np.maximum(cnts["mid"], 1); cb = np.maximum(cnts["bbo"], 1)
    prof = pl.DataFrame({
        "offset_s": centers,
        "mid_disp_bps": np.where(cnts["mid"] > 0, acc["mid"] / cm, np.nan),
        "spread_bps": np.where(cnts["bbo"] > 0, acc["spread"] / cb, np.nan),
        "depth_usd": np.where(cnts["bbo"] > 0, acc["depth"] / cb, np.nan),
        "obi_signed": np.where(cnts["bbo"] > 0, acc["obi"] / cb, np.nan),
        "trade_rate": acc["trate"] / max(n, 1),
        "flow_signed": acc["flow"] / max(n, 1),
        "binance_notional": acc["bn"] / max(n, 1), "bybit_notional": acc["yn"] / max(n, 1)})
    prof.write_parquet(ART / f"cascade_profile_{sym}.parquet")
    pl.DataFrame([{"sym": sym, "n_cascades": n, "sell_frac": sells / max(n, 1),
                   "median_climax_off_s": float(np.median(climax_off)) if climax_off else np.nan,
                   "median_end_off_s": float(np.median(end_off)) if end_off else np.nan,
                   }]).write_parquet(ART / f"cascade_meta_{sym}.parquet")

    # panel-derived markout / features per phase (tag each sampled trade to a phase)
    panel, _ = analysis.load_panel(sym)
    begins = np.array([c["begin"] for c in casc])
    tt = panel["timestamp"].to_numpy()
    ka = np.searchsorted(begins, tt, side="right") - 1
    kb = np.searchsorted(begins, tt, side="left")
    off_after = np.where(ka >= 0, (tt - begins[np.clip(ka, 0, len(begins) - 1)]) / US, np.inf)
    off_before = np.where(kb < len(begins), (begins[np.clip(kb, 0, len(begins) - 1)] - tt) / US, np.inf)
    phase = np.full(len(tt), "quiet", dtype=object)
    phase[off_after < 300] = "after"
    phase[(off_after >= 12) & (off_after < 30)] = "end"
    phase[(off_after >= 5) & (off_after < 12)] = "middle"
    phase[(off_after >= 0) & (off_after < 5)] = "begin"
    phase[off_before <= 30] = "pre"
    panel = panel.with_columns(pl.Series("phase", phase))
    pf = (panel.filter(pl.col("phase") != "quiet").group_by("phase")
          .agg(n_trades=pl.len(), pnl_120=pl.col("pnl_120").fill_nan(None).mean(),
               **{f: pl.col(f).fill_nan(None).mean() for f in FEATS}))

    out = []
    for p in PHASES:
        md = ph[p]["mid"]; sd = ph[p]["spread"]; nn = max(ph[p]["n"], 1)
        rec = {"sym": sym, "phase": p,
               "mid_disp_bps": md[0] / md[1] if md[1] else np.nan,
               "spread_bps": sd[0] / sd[1] if sd[1] else np.nan,
               "depth_usd": ph[p]["depth"][0] / sd[1] if sd[1] else np.nan,
               "obi_signed": ph[p]["obi"][0] / sd[1] if sd[1] else np.nan,
               "trade_rate": ph[p]["trate"] / nn, "flow_signed": ph[p]["flow"] / nn,
               "binance_notional": ph[p]["bn"], "bybit_notional": ph[p]["yn"]}
        fr = pf.filter(pl.col("phase") == p)
        rec["n_trades"] = int(fr["n_trades"][0]) if fr.height else 0
        rec["pnl_120"] = float(fr["pnl_120"][0]) if fr.height else np.nan
        for f in FEATS:
            rec[f] = float(fr[f][0]) if fr.height else np.nan
        out.append(rec)
    pl.DataFrame(out).write_parquet(ART / f"cascade_phase_{sym}.parquet")
    return n


def main() -> None:
    t0 = time.time()
    for sym in config.SYMBOLS:
        book = io.load_book_top(sym)
        n = process(sym, book)
        print(f"  [{sym}] {n} cascades (>= {MIN_EVENTS} prints, gap < {GAP_S}s)")
        del book
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
