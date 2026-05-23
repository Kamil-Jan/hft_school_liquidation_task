"""Central configuration: filesystem paths, the trading universe, and the frozen
task spec constants from ``description.md``.

Everything that the hidden test grades on (markout horizons, the rebate, the
notional cap, the Bybit delay, the turnover floor, the train/validation split)
lives here as a single source of truth so no script can drift from the spec.

Timestamps are int64 **microseconds since the UNIX epoch (UTC)** throughout.
"""
from __future__ import annotations

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

# Split boundaries (epoch microseconds, UTC), right edge exclusive.
#   train      2025-12-01 .. 2026-01-31   (62 days)
#   validation 2026-02-01 .. 2026-02-28   (28 days)
TRAIN_START: int = 1_764_547_200_000_000       # 2025-12-01 00:00:00 UTC
VAL_START: int = TRAIN_START + 62 * DAY_US      # 2026-02-01 00:00:00 UTC
VAL_END: int = TRAIN_START + 90 * DAY_US        # 2026-03-01 00:00:00 UTC (exclusive)
