"""
tests/test_fmax_search.py
=========================
Comprehensive automated tests for the frequency search algorithms:
- Normal search inside the prediction window
- Upper-edge extension with stride doubling
- Lower-window fallback
- No passing frequency across the complete grid (reject)
- Reproducibility with a fixed random seed
- Exact count of all test evaluations
- last_pass = None initialization and confirmation
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fmax_search import (
    build_freq_grid, snap_to_grid, passes_at,
    binary_search_fmax, conventional_binary_search,
    windowed_binary_search, FREQ_MIN, FREQ_MAX, FREQ_STEP,
)

GRID = build_freq_grid()


# ---------------------------------------------------------------------------
# Grid and snapping tests
# ---------------------------------------------------------------------------
class TestFrequencyGrid:
    def test_grid_step_is_10mhz(self):
        diffs = np.diff(GRID)
        assert np.all(diffs == FREQ_STEP), "Grid step must be 10 MHz"

    def test_grid_starts_at_freq_min(self):
        assert GRID[0] == FREQ_MIN

    def test_grid_ends_at_freq_max(self):
        assert GRID[-1] == FREQ_MAX

    def test_snap_floor(self):
        assert snap_to_grid(1987, GRID, "floor") == 1980
        assert snap_to_grid(1980, GRID, "floor") == 1980

    def test_snap_ceil(self):
        assert snap_to_grid(1981, GRID, "ceil") == 1990
        assert snap_to_grid(1990, GRID, "ceil") == 1990


# ---------------------------------------------------------------------------
# Oracle tests
# ---------------------------------------------------------------------------
class TestOracle:
    def test_passes_deterministic(self):
        assert passes_at(1990, 2000.0, noise_sigma=0.0) is True
        assert passes_at(2000, 2000.0, noise_sigma=0.0) is True
        assert passes_at(2010, 2000.0, noise_sigma=0.0) is False

    def test_oracle_with_noise_reproducibility(self):
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        out1 = [passes_at(2000, 2000.0, noise_sigma=4.0, rng=rng1) for _ in range(20)]
        out2 = [passes_at(2000, 2000.0, noise_sigma=4.0, rng=rng2) for _ in range(20)]
        assert out1 == out2, "Oracle outcomes with same RNG seed must be identical"


# ---------------------------------------------------------------------------
# last_pass = None verification
# ---------------------------------------------------------------------------
class TestLastPassInitialization:
    def test_returns_none_when_all_fail(self):
        """When every frequency fails, binary search must return None, not the first frequency."""
        # Chip true Fmax = 1500 (below grid min 1600)
        measured, n_tests = binary_search_fmax(1600, 1700, 1500.0, GRID, noise_sigma=0.0)
        assert measured is None, f"Expected None when all fail, got {measured}"
        assert n_tests > 0

    def test_only_records_passing_frequency(self):
        """Verify that last_pass only updates on an actual pass."""
        measured, n_tests = binary_search_fmax(1900, 2000, 1945.0, GRID, noise_sigma=0.0)
        assert measured == 1940
        assert n_tests > 0


# ---------------------------------------------------------------------------
# Normal search inside window
# ---------------------------------------------------------------------------
class TestNormalSearchInWindow:
    def test_normal_search_inside_window(self):
        # Prediction 2000, true 2005. Window is ~[1960, 2040].
        res = windowed_binary_search(2000.0, 39.2, 2005.0, GRID, noise_sigma=0.0)
        assert res.search_type == "window_direct"
        assert res.measured_fmax_mhz == 2000
        assert res.n_tests == 3 or res.n_tests == 4
        assert res.success is True


# ---------------------------------------------------------------------------
# Upper-edge extension with stride doubling
# ---------------------------------------------------------------------------
class TestUpperEdgeExtension:
    def test_upper_edge_extension_stride_doubling(self):
        # Prediction 1986.2, window ~ [1940, 2030]. True = 2044.4 (exceeds window).
        res = windowed_binary_search(1986.2, 39.2, 2044.4, GRID, noise_sigma=0.0)
        assert res.search_type == "upper_extension"
        assert res.measured_fmax_mhz == 2040
        # 4 window tests + test 2040 (pass) + test 2060 (fail) + bracket test 2050 (fail) = 7 tests
        assert res.n_tests == 7
        assert res.success is True

    def test_upper_edge_extension_high_overshoot(self):
        # Prediction 1900, window ~ [1860, 1940]. True = 2105.0
        res = windowed_binary_search(1900.0, 39.2, 2105.0, GRID, noise_sigma=0.0)
        assert res.search_type == "upper_extension"
        assert res.measured_fmax_mhz == 2100
        assert res.success is True


# ---------------------------------------------------------------------------
# Lower-window fallback
# ---------------------------------------------------------------------------
class TestLowerWindowFallback:
    def test_lower_window_fallback(self):
        # Prediction 1857.5, window ~ [1810, 1900]. True = 1808.8 (below window).
        res = windowed_binary_search(1857.5, 39.2, 1808.8, GRID, noise_sigma=0.0)
        assert res.search_type == "lower_fallback"
        assert res.measured_fmax_mhz == 1800
        # 3 window tests (all fail) + 5 fallback binary search tests = 8 tests
        assert res.n_tests == 8
        assert res.success is True


# ---------------------------------------------------------------------------
# Complete grid failure (no passing frequency / reject)
# ---------------------------------------------------------------------------
class TestNoPassingFrequencyAcrossGrid:
    def test_reject_below_grid_minimum(self):
        # Chip has true Fmax of 1550 (below grid minimum 1600)
        res_win = windowed_binary_search(1650.0, 39.2, 1550.0, GRID, noise_sigma=0.0)
        assert res_win.search_type == "lower_fallback"
        assert res_win.measured_fmax_mhz is None
        assert res_win.success is False

        res_bin = conventional_binary_search(1550.0, GRID, noise_sigma=0.0)
        assert res_bin.measured_fmax_mhz is None
        assert res_bin.success is False


# ---------------------------------------------------------------------------
# Reproducibility with fixed random seed
# ---------------------------------------------------------------------------
class TestReproducibilityWithNoise:
    def test_search_reproducible_with_seed(self):
        seed = 42
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        res1 = [
            windowed_binary_search(2000.0, 39.2, 2003.5, GRID, noise_sigma=4.0, rng=rng1)
            for _ in range(50)
        ]
        res2 = [
            windowed_binary_search(2000.0, 39.2, 2003.5, GRID, noise_sigma=4.0, rng=rng2)
            for _ in range(50)
        ]

        meas1 = [r.measured_fmax_mhz for r in res1]
        meas2 = [r.measured_fmax_mhz for r in res2]
        tests1 = [r.n_tests for r in res1]
        tests2 = [r.n_tests for r in res2]

        assert meas1 == meas2, "Measured frequencies must be identical with same seed"
        assert tests1 == tests2, "Test counts must be identical with same seed"
