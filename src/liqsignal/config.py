"""Central configuration: filesystem paths, the trading universe, and the frozen
task spec constants from ``description.md``.

Everything that the hidden test grades on (markout horizons, the rebate, the
notional cap, the Bybit delay, the turnover floor, the train/validation split)
lives here as a single source of truth so no script can drift from the spec.

Timestamps are int64 **microseconds since the UNIX epoch (UTC)** throughout.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout (resolved relative to the repo root, so paths work no
# matter which directory a script or notebook is launched from).
#
# ``DATA_DIR`` defaults to ``<repo>/data`` but can be repointed via the
# ``LIQSIGNAL_DATA_DIR`` environment variable — used to evaluate the signal on a
# held-out *test* set kept in a separate directory (e.g. ``data_test/``) without
# disturbing the shipped train data. See ``make evaluate DATA_DIR=...``.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.environ.get("LIQSIGNAL_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"


def ensure_artifacts() -> Path:
    """Create the artifacts directory if needed and return it (call from writers)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Universe and time units
# ---------------------------------------------------------------------------
SYMBOLS: tuple[str, ...] = ("btc", "eth")

US: int = 1_000_000               # microseconds per second
DAY_US: int = 86_400 * US         # microseconds per day

# Data sources and how their files are named on disk.
SOURCES: tuple[str, ...] = ("trades", "bbo", "liq_binance", "liq_bybit")


def dataset_path(source: str, symbol: str) -> Path:
    """Absolute path to a raw parquet file for ``source`` and ``symbol``.

    ``source`` is one of :data:`SOURCES`; ``symbol`` is one of :data:`SYMBOLS`.
    """
    sym = symbol.lower()
    layout = {
        "trades": DATA_DIR / "binance_trades" / f"perp_{sym}usdt.parquet",
        "bbo": DATA_DIR / "binance_booktickers" / f"perp_{sym}usdt.parquet",
        "liq_binance": DATA_DIR / "binance_liquidations" / f"perp_{sym}usdt.parquet",
        "liq_bybit": DATA_DIR / "bybit_liquidations" / f"{sym}usdt.parquet",
    }
    if source not in layout:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return layout[source]


# ---------------------------------------------------------------------------
# Task spec (description.md) — do not change without re-reading the spec.
# ---------------------------------------------------------------------------
TAUS: tuple[int, ...] = (30, 120, 300)        # markout horizons, seconds
REBATE_BPS: float = 0.5                        # maker rebate added to every markout
NOTIONAL_CAP: float = 100_000.0                # weight w_i = min(notional_i, cap)
BYBIT_DELAY_US: int = 200_000                  # cross-exchange availability handicap
TURNOVER_MIN_PER_DAY: float = 500_000.0        # kept-trade clipped turnover floor (USD/day)

# Feature pruning: keep the top-N features (by permutation importance on validation)
# per horizon, refitting on them. None ⇒ keep all. Override at train time with
# `make train N_FEATURES=30` / `train_model.py --n-features 30`.
N_FEATURES: int | None = None

# Curated per-(symbol, τ) feature sets. Used by `train_model` per model when non-empty
# (precedence: --n-features > FEATURE_SETS > all features). **Currently empty ⇒ all
# features.** A first cut (notebook 02 / `scripts/select_features.py`, redundancy-filtered
# then top-N by *validation* importance) was reverted: ranking on validation overfit it and
# **hurt the held-out test** (see .claude/docs/features.md). Re-derive leak-free (rank on a train-internal
# fold, val/test untouched) before populating this. The script is kept for that redo.
FEATURE_SETS: dict[tuple[str, int], list[str]] = {}

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
# To re-time the splits, edit the four dates below — that is the only place. All
# boundaries are UTC midnight, **right edge exclusive**. Data currently spans
# 2025-11-01 .. 2026-04-28 (inclusive).
#
#   USE_TEST = True   → 3-way: train | validation | test
#       train [TRAIN_START, VAL_START)  validation [VAL_START, TEST_START)  test [TEST_START, SPLIT_END)
#   USE_TEST = False  → 2-way: the April test window folds into validation
#       train [TRAIN_START, VAL_START)  validation [VAL_START, SPLIT_END)
#
# Leak safety: SPLIT_EMBARGO_S seconds are purged before each split boundary so no
# trade's markout window (≤ max τ) can straddle two splits — see splits.py.

def _utc_us(year: int, month: int, day: int) -> int:
    """Epoch microseconds at 00:00:00 UTC on the given date."""
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp()) * US


USE_TEST: bool = True
SPLIT_EMBARGO_S: int = max(TAUS)               # = 300 s; gap purged before each boundary

TRAIN_START: int = _utc_us(2025, 11, 1)        # start of train (first day of data)
VAL_START:   int = _utc_us(2026, 3, 1)         # train → validation
TEST_START:  int = _utc_us(2026, 4, 1)         # validation → test (only if USE_TEST)
SPLIT_END:   int = _utc_us(2026, 4, 29)        # end of data, exclusive (2026-04-28 inclusive)

# Derived (do not edit): right edge of the validation window.
VAL_END: int = TEST_START if USE_TEST else SPLIT_END
