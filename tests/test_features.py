"""Unit tests for the pure feature functions and the feature-context pipeline."""
import numpy as np

from liqsignal import config
from liqsignal.io import BookTop, Liquidations
from liqsignal.features import (basis_proxy_bps, build_context, cascade_acceleration,
                                compute_features, hour_of_day, in_funding_window, is_weekend,
                                liq_run_length, microprice_adjustment_bps, minutes_to_funding,
                                order_book_imbalance, seconds_since_last, windowed_flow_sums,
                                windowed_liq)

US = config.US


def test_order_book_imbalance():
    obi = order_book_imbalance(np.array([3.0, 1.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(obi, [0.5, 0.0])


def test_microprice_adjustment():
    # (spread/2)/mid * obi * 1e4 ; spread=2, mid=100, obi=0.5 -> 1/100*0.5*1e4 = 50 bps
    adj = microprice_adjustment_bps(np.array([2.0]), np.array([100.0]), np.array([0.5]))
    assert np.isclose(adj[0], 50.0)


def test_windowed_liq_prefix_sums():
    liq = Liquidations(ts=np.array([10, 20, 30], dtype=np.int64),
                       side=np.array(["buy", "sell", "buy"]),
                       price=np.array([1.0, 1.0, 1.0]),
                       signed_notional=np.array([100.0, -50.0, 200.0]))
    # window covers (q-25, q] -> for q=30 covers ts 10? no: 30-25=5 -> ts 10,20,30 all in (5,30]
    net, absn, cnt = windowed_liq(liq, np.array([30], dtype=np.int64), window_us=25)
    assert np.isclose(net[0], 250.0) and np.isclose(absn[0], 350.0) and cnt[0] == 3
    # tighter window (q-15,q] for q=30 -> ts 20,30
    net, absn, cnt = windowed_liq(liq, np.array([30], dtype=np.int64), window_us=15)
    assert np.isclose(net[0], 150.0) and np.isclose(absn[0], 250.0) and cnt[0] == 2


def test_cascade_acceleration():
    cnt_short = np.array([1.0, 2.0, 0.0])
    cnt_long = np.array([10.0, 4.0, 0.0])
    out = cascade_acceleration(cnt_short, cnt_long, 30.0, 300.0)
    assert np.isclose(out[0], 1.0)   # uniform: (1/30)/(10/300) = 1
    assert np.isclose(out[1], 5.0)   # accelerating: (2/30)/(4/300) = 5
    assert np.isnan(out[2])          # no long-window events -> missing


def test_windowed_flow_sums():
    s0 = 100
    signed = np.array([2.0, -1.0, 3.0, 0.0, 5.0])
    tot = np.array([2.0, 3.0, 3.0, 4.0, 5.0])
    cnt = np.array([1.0, 2.0, 1.0, 2.0, 3.0])
    cs_s = np.concatenate([[0.0], np.cumsum(signed)])
    cs_t = np.concatenate([[0.0], np.cumsum(tot)])
    cs_c = np.concatenate([[0.0], np.cumsum(cnt)])
    n = len(signed)
    # query at second s0+3, window 2s -> the two whole seconds strictly before it (idx 1,2)
    q = np.array([(s0 + 3) * config.US + config.US // 2], dtype=np.int64)
    net, t, c, valid = windowed_flow_sums(cs_s, cs_t, cs_c, s0, n, q, 2)
    assert valid[0]
    assert np.isclose(net[0], signed[1] + signed[2])   # -1 + 3 = 2
    assert np.isclose(t[0], tot[1] + tot[2]) and np.isclose(c[0], cnt[1] + cnt[2])
    # no in-grid history -> invalid
    _, _, _, v0 = windowed_flow_sums(cs_s, cs_t, cs_c, s0, n, np.array([s0 * config.US]), 2)
    assert not v0[0]


def test_liq_run_length():
    side = np.array(["buy", "buy", "sell", "buy", "buy", "buy"])
    np.testing.assert_array_equal(liq_run_length(side), [1, 2, -1, 1, 2, 3])


def test_funding_seasonality():
    base = config._utc_us(2026, 1, 1)                  # 00:00 UTC is a funding mark
    assert np.isclose(minutes_to_funding(np.array([base]))[0], 0.0)
    assert in_funding_window(np.array([base]))[0] == 1.0
    t2 = base + 2 * 3600 * config.US                   # 2h after a mark -> 360 min to next
    assert np.isclose(minutes_to_funding(np.array([t2]))[0], 360.0)
    assert in_funding_window(np.array([t2]))[0] == 0.0


def test_seconds_since_last_nan_default():
    ev = np.array([100, 200], dtype=np.int64)
    out = seconds_since_last(ev, np.array([50, 250], dtype=np.int64))  # 50 before first
    assert np.isnan(out[0])
    assert np.isclose(out[1], (250 - 200) / US)


def test_hour_and_weekend():
    monday = config._utc_us(2025, 12, 1)   # 2025-12-01 00:00:00 UTC is a Monday
    assert hour_of_day(np.array([monday]))[0] == 0
    assert hour_of_day(np.array([monday + 13 * 3600 * config.US]))[0] == 13
    assert is_weekend(np.array([monday]))[0] == 0.0                       # Monday
    assert is_weekend(np.array([monday + 5 * config.DAY_US]))[0] == 1.0   # Saturday


def test_basis_proxy_recency_gate():
    liq = Liquidations(ts=np.array([0], dtype=np.int64), side=np.array(["sell"]),
                       price=np.array([99.0]), signed_notional=np.array([-1.0]))
    mid = np.array([100.0, 100.0])
    # query at 1s (fresh) and at 10min (stale -> 0)
    out = basis_proxy_bps(liq, np.array([1 * US, 600 * US + US], dtype=np.int64), mid, max_stale_s=300)
    assert np.isclose(out[0], (99 - 100) / 100 * 1e4)  # -100 bps
    assert out[1] == 0.0


def _tiny_book():
    ts = (np.arange(6) * US).astype(np.int64)
    mid = np.array([100., 100.1, 100., 100.2, 100.1, 100.])
    return BookTop(ts=ts, mid=mid, spread=np.full(6, 0.1, np.float32),
                   bid_amount=np.full(6, 2.0, np.float32), ask_amount=np.full(6, 1.0, np.float32))


def _tiny_liq():
    return Liquidations(ts=np.array([1 * US, 2 * US], dtype=np.int64),
                        side=np.array(["buy", "sell"]), price=np.array([100.0, 100.0]),
                        signed_notional=np.array([500.0, -300.0]))


def test_compute_features_contract():
    ctx = build_context(_tiny_book(), _tiny_liq(), _tiny_liq())
    t = np.array([3 * US + US // 2, 4 * US + US // 2], dtype=np.int64)
    sign = np.array([1, -1], dtype=np.int8)
    price = np.array([100.2, 100.1])
    feats = compute_features(ctx, t, sign, price)
    assert all(v.shape == (2,) for v in feats.values())
    # signed alignment feature uses the taker sign
    assert "bybit_liqalign_30s" in feats and "obi_signed" in feats and "hour" in feats
    # obi = (2-1)/(2+1) = 1/3; signed for buy(+1)/sell(-1)
    np.testing.assert_allclose(feats["obi"], [1 / 3, 1 / 3], atol=1e-9)
    np.testing.assert_allclose(feats["obi_signed"], [1 / 3, -1 / 3], atol=1e-9)
    # new cascade / cross-exchange features are present and well-formed
    assert {"binance_liqaccel", "bybit_liqaccel", "xexch_liqpress_30s",
            "xexch_liqalign_300s"} <= set(feats)
    # identical binance/bybit feeds in this fixture ⇒ zero cross-exchange divergence
    np.testing.assert_allclose(feats["xexch_liqpress_30s"], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(feats["xexch_liqpress_300s"], [0.0, 0.0], atol=1e-9)
    # roadmap #1–4 features are present
    assert {"tfi_30s", "tfi_aligned_300s", "trade_intensity_30s", "flow_imbalance_mag_300s",
            "signed_vol_mom_30s", "liq_lead_s", "binance_liq_runlen", "bybit_liqz",
            "rskew_30", "varratio_300", "vol_ts_ratio", "min_to_funding",
            "in_funding_window"} <= set(feats)
    # flow=None in this fixture ⇒ flow features are NaN
    assert np.all(np.isnan(feats["tfi_30s"])) and np.all(np.isnan(feats["signed_vol_mom_300s"]))
