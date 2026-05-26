"""Leak-free N-sweep helpers: the parsimony knee (``pick_n``) is a pure function and is
importable without touching any panel I/O (loading is deferred into ``load_panels``)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import select_features as sf  # noqa: E402


def test_pick_n_knee_returns_smallest_at_peak():
    # Score plateaus at 1.0 for N in {8, 12}; the knee rule keeps the *smallest* set there.
    rows = [{"N": 5, "n_selected": 5, "internal_score": 0.5},
            {"N": 8, "n_selected": 8, "internal_score": 1.0},
            {"N": 12, "n_selected": 12, "internal_score": 1.0},
            {"N": 40, "n_selected": 40, "internal_score": 0.9}]
    assert sf.pick_n(rows)["n_selected"] == 8


def test_pick_n_tolerance_prefers_fewer_features():
    # A slightly-lower (within tol) score at a much smaller N wins on parsimony.
    rows = [{"N": 5, "n_selected": 5, "internal_score": 0.99},
            {"N": 40, "n_selected": 40, "internal_score": 1.00}]
    assert sf.pick_n(rows, tol=0.02)["n_selected"] == 5
    assert sf.pick_n(rows, tol=0.005)["n_selected"] == 40   # tighter tol no longer ties


def test_helpers_importable_without_panel_io():
    # load_panels must NOT have run on import (FEATURES stays empty); sweep_n is importable.
    assert sf.FEATURES == []
    assert callable(sf.sweep_n)
