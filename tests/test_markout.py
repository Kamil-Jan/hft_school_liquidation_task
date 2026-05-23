"""Spec-critical: forward-fill, trade sign, and the markout formula."""
import numpy as np

from liqsignal.config import REBATE_BPS, US
from liqsignal.markout import (compute_markout, forward_fill_mid, markout_bps,
                               trade_sign)

BBO_TS = np.array([10, 20, 30], dtype=np.int64)
BBO_MID = np.array([100.0, 101.0, 102.0])


def test_trade_sign():
    assert list(trade_sign(np.array(["buy", "sell", "buy"]))) == [1, -1, 1]


def test_forward_fill_basic():
    q = np.array([5, 10, 25, 30, 35], dtype=np.int64)
    mid, valid = forward_fill_mid(BBO_TS, BBO_MID, q)
    # 5 -> before first (invalid); 10 -> 100; 25 -> last<=25 is 101; 30 -> 102; 35 -> beyond (invalid)
    assert list(valid) == [False, True, True, True, False]
    np.testing.assert_allclose(mid[1:4], [100.0, 101.0, 102.0])
    assert np.isnan(mid[0]) and np.isnan(mid[4])


def test_markout_formula_signs():
    # taker buy (maker sell) at 100; mid falls to 99 => maker +100bps (+ rebate)
    pnl = markout_bps(np.array([100.0]), np.array([1]), np.array([99.0]))
    assert np.isclose(pnl[0], 100.0 + REBATE_BPS)
    # taker sell (maker buy) at 100; mid falls to 99 => maker loses 100bps (+ rebate)
    pnl = markout_bps(np.array([100.0]), np.array([-1]), np.array([99.0]))
    assert np.isclose(pnl[0], -100.0 + REBATE_BPS)


def test_compute_markout_excludes_beyond():
    # BBO ticks at 0s, 30s, 60s. A trade at t=0 with tau=30 lands on a valid tick;
    # tau=300 runs past the last tick and must be excluded (NaN).
    bts = np.array([0, 30 * US, 60 * US], dtype=np.int64)
    bmid = np.array([100.0, 101.0, 102.0])
    trade_ts = np.array([0, 0], dtype=np.int64)
    sign = np.array([1, 1])
    price = np.array([100.0, 100.0])

    pnl_30 = compute_markout(trade_ts, sign, price, bts, bmid, tau=30)
    pnl_300 = compute_markout(trade_ts, sign, price, bts, bmid, tau=300)
    # tau=30: mid 100 -> 101 (+1%), maker sell loses 100 bps net of rebate
    assert np.isclose(pnl_30[0], -100.0 + REBATE_BPS)
    # tau=300: beyond last tick -> excluded
    assert np.isnan(pnl_300[0])
