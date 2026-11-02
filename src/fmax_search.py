"""
fmax_search.py
==============
Implements the frequency search algorithms described in the technical document:
- Conventional binary search over full frequency grid [FREQ_MIN, FREQ_MAX]
- AI-Windowed Binary Search with Upper-Edge Extension and Lower-Window Fallback (Algorithm 1)

All simulated tests can include independent Gaussian test-repeatability noise (sigma = 4.0 MHz).
Every physical test evaluation is counted.
"""

from typing import NamedTuple, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frequency grid constants
# ---------------------------------------------------------------------------
FREQ_MIN  = 1600   # MHz — absolute lower bound (grid-aligned)
FREQ_MAX  = 2400   # MHz — absolute upper bound (grid-aligned)
FREQ_STEP = 10     # MHz — grid step size
DEFAULT_NOISE_SIGMA = 4.0  # MHz — Gaussian test-repeatability noise


def build_freq_grid(freq_min: int = FREQ_MIN,
                    freq_max: int = FREQ_MAX,
                    step: int = FREQ_STEP) -> np.ndarray:
    """Return the integer frequency grid as a numpy array."""
    return np.arange(freq_min, freq_max + step, step, dtype=int)


FREQ_GRID = build_freq_grid()


def snap_to_grid(freq_mhz: float,
                 grid: np.ndarray = FREQ_GRID,
                 direction: str = "nearest") -> int:
    """
    Snap a frequency to the nearest, floor, or ceil grid point.

    Parameters
    ----------
    freq_mhz  : frequency to snap
    grid      : sorted integer grid array
    direction : 'nearest' | 'floor' | 'ceil'
    """
    freq_mhz = float(freq_mhz)
    if direction == "nearest":
        idx = int(np.argmin(np.abs(grid - freq_mhz)))
    elif direction == "floor":
        idx = int(np.searchsorted(grid, freq_mhz, side="right")) - 1
        idx = max(0, idx)
    elif direction == "ceil":
        idx = int(np.searchsorted(grid, freq_mhz, side="left"))
        idx = min(idx, len(grid) - 1)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return int(grid[idx])


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
def passes_at(freq_mhz: int,
              fmax_true_mhz: float,
              noise_sigma: float = 0.0,
              rng: Optional[np.random.Generator] = None) -> bool:
    """
    Simulated test oracle.
    Returns True if the chip passes at the given frequency.
    
    If noise_sigma > 0, independent Gaussian test-repeatability noise is added
    for each test execution:
        passes if freq_mhz <= fmax_true_mhz + noise
    where noise ~ N(0, noise_sigma^2).
    """
    if noise_sigma > 0.0:
        if rng is not None:
            noise = float(rng.normal(0.0, noise_sigma))
        else:
            noise = float(np.random.normal(0.0, noise_sigma))
    else:
        noise = 0.0

    return freq_mhz <= (fmax_true_mhz + noise)


# ---------------------------------------------------------------------------
# Search result container
# ---------------------------------------------------------------------------
class SearchResult(NamedTuple):
    measured_fmax_mhz:  Optional[int]  # Highest passing frequency found (None if reject)
    n_tests:            int            # Total number of test steps taken
    success:            bool           # True if a passing frequency was found and verified
    window_low_mhz:     int            # Lower bound of search window used
    window_high_mhz:    int            # Upper bound of search window used
    fmax_true_mhz:      float          # Ground truth (for evaluation only)
    search_type:        str            # 'window_direct', 'upper_extension', 'lower_fallback', 'baseline'


# ---------------------------------------------------------------------------
# Binary search within a frequency range
# ---------------------------------------------------------------------------
def binary_search_fmax(search_low: int,
                       search_high: int,
                       fmax_true_mhz: float,
                       grid: np.ndarray = FREQ_GRID,
                       noise_sigma: float = 0.0,
                       rng: Optional[np.random.Generator] = None) -> tuple[Optional[int], int]:
    """
    Binary search for the highest passing frequency in [search_low, search_high].

    Uses the 'last passing' paradigm:
      - last_pass is initialized to None.
      - A frequency is only recorded as passing if an actual test succeeds.
      - Stop when search window collapses (lo_idx > hi_idx).

    In the presence of test noise, occasional inconsistencies may occur (e.g.
    a test near Fmax passes or fails spuriously). Binary search deterministically
    narrows down the search interval based on each test query outcome, and
    last_pass tracks the highest confirmed passing frequency tested so far.

    Returns
    -------
    (measured_fmax_mhz, n_tests)
      measured_fmax_mhz : highest frequency that passed (or None if none passed)
      n_tests           : number of oracle calls made
    """
    sub_grid = grid[(grid >= search_low) & (grid <= search_high)]
    if len(sub_grid) == 0:
        return None, 0

    lo_idx = 0
    hi_idx = len(sub_grid) - 1
    n_tests = 0
    last_pass: Optional[int] = None

    while lo_idx <= hi_idx:
        mid_idx = (lo_idx + hi_idx) // 2
        freq = int(sub_grid[mid_idx])
        n_tests += 1

        if passes_at(freq, fmax_true_mhz, noise_sigma=noise_sigma, rng=rng):
            last_pass = freq
            lo_idx = mid_idx + 1   # Try higher frequency
        else:
            hi_idx = mid_idx - 1   # Too high, try lower frequency

    return last_pass, n_tests


# ---------------------------------------------------------------------------
# Conventional binary search (baseline)
# ---------------------------------------------------------------------------
def conventional_binary_search(fmax_true_mhz: float,
                               grid: np.ndarray = FREQ_GRID,
                               noise_sigma: float = 0.0,
                               rng: Optional[np.random.Generator] = None) -> SearchResult:
    """
    Baseline: binary search over the full frequency grid [FREQ_MIN, FREQ_MAX].
    """
    measured, n_tests = binary_search_fmax(
        int(grid[0]), int(grid[-1]), fmax_true_mhz, grid,
        noise_sigma=noise_sigma, rng=rng
    )
    success = measured is not None
    return SearchResult(
        measured_fmax_mhz  = measured,
        n_tests            = n_tests,
        success            = success,
        window_low_mhz     = int(grid[0]),
        window_high_mhz    = int(grid[-1]),
        fmax_true_mhz      = fmax_true_mhz,
        search_type        = "baseline",
    )


# ---------------------------------------------------------------------------
# AI-windowed binary search (Algorithm 1)
# ---------------------------------------------------------------------------
def windowed_binary_search(pred_fmax_mhz: float,
                           window_half_mhz: float,
                           fmax_true_mhz: float,
                           grid: np.ndarray = FREQ_GRID,
                           noise_sigma: float = 0.0,
                           rng: Optional[np.random.Generator] = None) -> SearchResult:
    """
    AI-assisted windowed binary search following Algorithm 1:
      1. Calculate raw prediction window [pred - half, pred + half].
      2. Snap window boundaries to grid: floor for low, ceil for high.
         Clamp within [FREQ_MIN, FREQ_MAX].
      3. Perform binary search in [win_low, win_high].
      4. If all tests fail (last_pass is None):
         Execute Lower-Window Fallback: binary search on [FREQ_MIN, win_low - FREQ_STEP].
      5. If highest passing frequency in window is win_high:
         Execute Upper-Edge Extension:
           - Start stride = 1 (10 MHz).
           - Probe curr_pass + stride * FREQ_STEP.
           - If pass: double stride and repeat.
           - If fail or exceed FREQ_MAX: binary search within overshoot bracket.
      6. Return highest confirmed passing frequency and total test count.
    """
    grid_min = int(grid[0])
    grid_max = int(grid[-1])

    # 1. Raw window bounds
    raw_low  = pred_fmax_mhz - window_half_mhz
    raw_high = pred_fmax_mhz + window_half_mhz

    # 2. Clamped and grid-aligned bounds
    raw_low  = max(raw_low,  float(grid_min))
    raw_high = min(raw_high, float(grid_max))
    win_low  = snap_to_grid(raw_low,  grid, "floor")
    win_high = snap_to_grid(raw_high, grid, "ceil")

    if win_low >= win_high:
        win_low  = snap_to_grid(pred_fmax_mhz, grid, "floor")
        win_high = snap_to_grid(pred_fmax_mhz, grid, "ceil")
        if win_low == win_high:
            win_high = min(win_high + FREQ_STEP, grid_max)

    # 3. Binary search within prediction window
    last_pass, n_tests = binary_search_fmax(
        win_low, win_high, fmax_true_mhz, grid,
        noise_sigma=noise_sigma, rng=rng
    )
    search_type = "window_direct"

    # 4. Lower-window fallback rule
    # If all frequencies in window failed, search below win_low down to grid_min
    if last_pass is None:
        search_type = "lower_fallback"
        fallback_high = win_low - FREQ_STEP
        if fallback_high >= grid_min:
            fallback_pass, fallback_tests = binary_search_fmax(
                grid_min, fallback_high, fmax_true_mhz, grid,
                noise_sigma=noise_sigma, rng=rng
            )
            n_tests += fallback_tests
            last_pass = fallback_pass

    # 5. Upper-edge extension rule with stride doubling
    # If win_high passed, the true Fmax may exceed the upper prediction window
    elif last_pass == win_high:
        search_type = "upper_extension"
        stride = 1  # in multiples of FREQ_STEP (10 MHz)
        curr_pass = win_high

        while True:
            probe = curr_pass + stride * FREQ_STEP
            if probe > grid_max:
                bracket_lo = curr_pass + FREQ_STEP
                bracket_hi = grid_max
                break

            n_tests += 1
            if passes_at(probe, fmax_true_mhz, noise_sigma=noise_sigma, rng=rng):
                curr_pass = probe
                stride *= 2  # Double the stride
            else:
                # Probe failed: bracket is [curr_pass + FREQ_STEP, probe - FREQ_STEP]
                bracket_lo = curr_pass + FREQ_STEP
                bracket_hi = probe - FREQ_STEP
                break

        # Binary search inside the unresolved bracket (if valid)
        if bracket_lo <= bracket_hi:
            bracket_pass, bracket_tests = binary_search_fmax(
                bracket_lo, bracket_hi, fmax_true_mhz, grid,
                noise_sigma=noise_sigma, rng=rng
            )
            n_tests += bracket_tests
            if bracket_pass is not None:
                curr_pass = bracket_pass

        last_pass = curr_pass

    success = last_pass is not None

    return SearchResult(
        measured_fmax_mhz  = last_pass,
        n_tests            = n_tests,
        success            = success,
        window_low_mhz     = win_low,
        window_high_mhz    = win_high,
        fmax_true_mhz      = fmax_true_mhz,
        search_type        = search_type,
    )


# ---------------------------------------------------------------------------
# Batch evaluation of both search methods on the test set
# ---------------------------------------------------------------------------
def run_search_comparison(test_df: pd.DataFrame,
                          model_predictions: np.ndarray,
                          window_half_mhz: float,
                          grid: np.ndarray = FREQ_GRID,
                          noise_sigma: float = DEFAULT_NOISE_SIGMA,
                          random_seed: int = 42) -> pd.DataFrame:
    """
    Run both search algorithms on every row of the test set and return a
    comparison DataFrame.

    Each test execution uses independent Gaussian noise (sigma = noise_sigma)
    generated via a dedicated NumPy Generator with a fixed random_seed.
    """
    from src.data_loader import TARGET_COL, UNIT_ID_COL

    # Create dedicated RNGs for reproducibility
    rng_bin = np.random.default_rng(random_seed)
    rng_win = np.random.default_rng(random_seed)

    rows = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        fmax_true = float(row[TARGET_COL])
        fmax_pred = float(model_predictions[i])

        # Conventional binary search (baseline)
        bin_result = conventional_binary_search(
            fmax_true, grid, noise_sigma=noise_sigma, rng=rng_bin
        )

        # AI-windowed binary search (Algorithm 1)
        win_result = windowed_binary_search(
            fmax_pred, window_half_mhz, fmax_true, grid,
            noise_sigma=noise_sigma, rng=rng_win
        )

        rows.append({
            "unit_id":                     row.get(UNIT_ID_COL, i),
            "fmax_true_mhz":               fmax_true,
            "fmax_pred_mhz":               fmax_pred,
            "tests_binary_ai":             bin_result.n_tests,
            "measured_fmax_binary_ai":     bin_result.measured_fmax_mhz,
            "success_binary_ai":           bin_result.success,
            "tests_windowed_ai":           win_result.n_tests,
            "measured_fmax_windowed_ai":   win_result.measured_fmax_mhz,
            "success_windowed_ai":         win_result.success,
            "search_type_ai":              win_result.search_type,
            "win_low_ai":                  win_result.window_low_mhz,
            "win_high_ai":                 win_result.window_high_mhz,
            "true_bin_mhz":                row.get("true_bin_mhz", np.nan),
            # Reference values from dataset for comparison
            "tests_binary_csv":            row.get("tests_binary", np.nan),
            "tests_windowed_csv":          row.get("tests_windowed", np.nan),
            "fmax_measured_windowed_csv":  row.get("fmax_measured_windowed_mhz", np.nan),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    grid = build_freq_grid()
    print(f"Frequency grid: {grid[0]}-{grid[-1]} MHz, step={FREQ_STEP} MHz, {len(grid)} points")

    r_normal = windowed_binary_search(2000.0, 39.2, 2005.0, grid, noise_sigma=0.0)
    print("Normal search in window:", r_normal)

    r_upper = windowed_binary_search(1986.2, 39.2, 2044.4, grid, noise_sigma=0.0)
    print("Upper extension search:", r_upper)

    r_lower = windowed_binary_search(1857.5, 39.2, 1808.8, grid, noise_sigma=0.0)
    print("Lower fallback search:", r_lower)
