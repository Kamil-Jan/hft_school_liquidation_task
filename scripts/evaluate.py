#!/usr/bin/env python
"""Evaluate the submission ``signal()`` on a dataset and report the spec Score.

For each symbol this reads the four frames (trades / BBO / Binance & Bybit
liquidations) from the data directory, calls ``signal()`` to get the per-horizon
0/1 filter, computes the spec maker markout, and prints — for each tau —
``PnL_all``, ``PnL_kept``, **Score = PnL_kept − PnL_all**, the kept turnover per
day, and whether the $500k/day constraint holds.

The data directory defaults to ``data/``; point it at a held-out **test** tree
with ``--data-dir`` or the ``LIQSIGNAL_DATA_DIR`` env var (``make evaluate
DATA_DIR=data_test`` sets it for you). Trained models must be present in
``artifacts/`` (run ``make train`` first); otherwise ``signal()`` keeps all
trades and the Score is ~0.

Memory: a full 90-day symbol will not fit in 16 GB read whole — evaluate a bounded
test window, or pass ``--batch-size N`` to score the trades in chunks.

Usage:
  python scripts/evaluate.py --data-dir data_test
  python scripts/evaluate.py --data-dir data_test --batch-size 20000000
  make evaluate DATA_DIR=data_test
"""
from __future__ import annotations

import argparse
import os

# Resolve --data-dir before importing liqsignal: config.DATA_DIR reads the env at
# import time, so the override must be set first.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--data-dir")
_known, _ = _pre.parse_known_args()
if _known.data_dir:
    os.environ["LIQSIGNAL_DATA_DIR"] = _known.data_dir

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from liqsignal import config, io, scoring  # noqa: E402
from liqsignal.markout import compute_markout, trade_sign  # noqa: E402
from liqsignal.signal import signal  # noqa: E402


def _read_context(sym: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, io.BookTop]:
    """BBO + both liquidation frames (read whole) and the BookTop for markout."""
    bbo = pl.read_parquet(config.dataset_path("bbo", sym),
                          columns=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"])
    liq_b = pl.read_parquet(config.dataset_path("liq_binance", sym),
                            columns=["timestamp", "side", "price", "amount"])
    liq_y = pl.read_parquet(config.dataset_path("liq_bybit", sym),
                            columns=["timestamp", "side", "price", "amount"])
    return bbo, liq_b, liq_y, io.book_top_from_frame(bbo)


def _metrics(pnl_all: float, pnl_kept: float, kept_turn: float,
             frac_kept: float, n: int) -> dict:
    return dict(pnl_all=round(pnl_all, 4), pnl_kept=round(pnl_kept, 4),
                score=round(pnl_kept - pnl_all, 4),
                kept_turnover_per_day=round(kept_turn, 0),
                constraint_ok=kept_turn >= config.TURNOVER_MIN_PER_DAY,
                frac_kept=round(frac_kept, 4), n=n)


def _score_single(sym: str, book: io.BookTop, bbo, liq_b, liq_y) -> dict[int, dict]:
    """Whole-frame path: read trades, call signal() once, score each tau."""
    trades = pl.read_parquet(config.dataset_path("trades", sym),
                             columns=["timestamp", "side", "price", "amount"])
    t = trades["timestamp"].to_numpy()
    sign = trade_sign(trades["side"].to_numpy())
    price = trades["price"].to_numpy()
    w = np.minimum(price * trades["amount"].to_numpy(), config.NOTIONAL_CAP)
    n_days = int(np.unique(t // config.DAY_US).size)

    f_by_tau = signal(trades, bbo, liq_b, liq_y)
    out = {}
    for tau in config.TAUS:
        pnl = compute_markout(t, sign, price, book.ts, book.mid, tau)
        r = scoring.evaluate_filter(pnl, w, f_by_tau[tau], n_days=n_days)
        out[tau] = _metrics(r.pnl_all, r.pnl_kept, r.kept_turnover_per_day,
                            1.0 - r.frac_filtered_n, r.n)
    return out


def _score_batched(sym: str, book: io.BookTop, bbo, liq_b, liq_y, batch_size: int) -> dict[int, dict]:
    """Memory-bounded path: chunk the trades, call signal() per chunk, accumulate.

    signal() is stateless across trade rows (features depend only on each trade's
    timestamp and the shared BBO/liq context), so per-chunk scoring is exact.
    """
    # acc[tau] = [sum_w, sum_w*pnl, sum_w_keep, sum_w_keep*pnl, n_valid, n_keep]
    acc = {tau: [0.0, 0.0, 0.0, 0.0, 0, 0] for tau in config.TAUS}
    days: set[int] = set()
    for batch in io.iter_trade_batches(sym, batch_size=batch_size):
        t, price, amount = batch["timestamp"], batch["price"], batch["amount"]
        side = np.where(batch["is_buy"], "buy", "sell")
        sign = np.where(batch["is_buy"], 1, -1).astype(np.int8)
        w = np.minimum(price * amount, config.NOTIONAL_CAP)
        days.update(np.unique(t // config.DAY_US).tolist())
        chunk = pl.DataFrame({"timestamp": t, "side": side, "price": price, "amount": amount})
        f_by_tau = signal(chunk, bbo, liq_b, liq_y)
        for tau in config.TAUS:
            pnl = compute_markout(t, sign, price, book.ts, book.mid, tau)
            keep = 1.0 - f_by_tau[tau].astype(float)
            v = np.isfinite(pnl) & np.isfinite(w)
            a = acc[tau]
            a[0] += float(w[v].sum())
            a[1] += float((w[v] * pnl[v]).sum())
            a[2] += float((w[v] * keep[v]).sum())
            a[3] += float((w[v] * keep[v] * pnl[v]).sum())
            a[4] += int(v.sum())
            a[5] += int(keep[v].sum())

    n_days = max(1, len(days))
    out = {}
    for tau, (sw, swp, swk, swkp, nv, nk) in acc.items():
        pnl_all = swp / sw if sw > 0 else float("nan")
        pnl_kept = swkp / swk if swk > 0 else float("nan")
        out[tau] = _metrics(pnl_all, pnl_kept, swk / n_days,
                            (nk / nv) if nv else float("nan"), nv)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(config.SYMBOLS))
    p.add_argument("--data-dir", default=None,
                   help="directory holding the parquet tree (sets LIQSIGNAL_DATA_DIR)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="if >0, score trades in chunks of this many rows (memory-bounded)")
    p.add_argument("--out", default=None, help="optional parquet path for the metrics table")
    args = p.parse_args()

    print(f"data dir : {config.DATA_DIR}")
    print(f"symbols  : {args.symbols}   taus: {list(config.TAUS)}\n")

    rows = []
    for sym in args.symbols:
        bbo, liq_b, liq_y, book = _read_context(sym)
        res = (_score_batched(sym, book, bbo, liq_b, liq_y, args.batch_size)
               if args.batch_size > 0
               else _score_single(sym, book, bbo, liq_b, liq_y))
        for tau in config.TAUS:
            m = res[tau]
            flag = "OK" if m["constraint_ok"] else "VIOLATION"
            print(f"  {sym} tau={tau:>3}:  Score={m['score']:+.3f}  "
                  f"PnL_kept={m['pnl_kept']:+.3f}  PnL_all={m['pnl_all']:+.3f}  "
                  f"keep={m['frac_kept']:.1%}  keptTurn/day={m['kept_turnover_per_day']:,.0f}  {flag}")
            rows.append(dict(sym=sym, tau=tau, **m))

    table = pl.DataFrame(rows).sort(["sym", "tau"])
    pl.Config.set_tbl_rows(40)
    print("\n", table)
    if args.out:
        config.ensure_artifacts()
        table.write_parquet(args.out)
        print(f"\nwrote metrics -> {args.out}")


if __name__ == "__main__":
    main()
