"""
search_service.py
=================
Service for executing and comparing conventional binary search and
AI-windowed binary search (Algorithm 1) strictly reusing original core code.

Uses a thread-safe, non-invasive execution tracer (via threading.local)
to intercept oracle evaluation calls during search execution without modifying
src/fmax_search.py or duplicating Algorithm 1 search loops.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import src.fmax_search as fms
from src.fmax_search import (
    conventional_binary_search,
    windowed_binary_search,
    snap_to_grid,
    FREQ_GRID,
    FREQ_MIN,
    FREQ_MAX,
    FREQ_STEP,
    DEFAULT_NOISE_SIGMA,
)
from src.metrics import map_freq_to_speed_bin
from src.data_loader import FEATURE_COLS

# ---------------------------------------------------------------------------
# Empirical Training Bounding Envelope (Statistical OOD Detection)
# ---------------------------------------------------------------------------
_dataset_path = PROJECT_ROOT / "synthetic_fmax_dataset.csv"
TRAIN_ENVELOPE: Dict[str, Tuple[float, float]] = {}
if _dataset_path.exists():
    try:
        _df_raw = pd.read_csv(_dataset_path)
        _df_train = _df_raw[_df_raw["split"] == "train"]
        for _col in FEATURE_COLS:
            if _col in _df_train.columns:
                TRAIN_ENVELOPE[_col] = (float(_df_train[_col].min()), float(_df_train[_col].max()))
    except Exception as _e:
        print(f"Warning: could not load training envelope: {_e}")

# ---------------------------------------------------------------------------
# Thread-Local Storage for Execution Tracing
# ---------------------------------------------------------------------------
_local = threading.local()
_orig_passes_at = fms.passes_at


def dispatching_passes_at(freq_mhz: int,
                          fmax_true_mhz: float,
                          noise_sigma: float = 0.0,
                          rng: Optional[np.random.Generator] = None) -> bool:
    """
    Hooked passes_at function that delegates to the active thread's tracer if registered.
    If no tracer is active on the current thread, directly delegates to original passes_at.
    Guarantees 100% thread isolation for concurrent requests.
    """
    tracer = getattr(_local, "current_tracer", None)
    if tracer is not None:
        return tracer.record_and_evaluate(freq_mhz, fmax_true_mhz, noise_sigma=noise_sigma, rng=rng)
    return _orig_passes_at(freq_mhz, fmax_true_mhz, noise_sigma=noise_sigma, rng=rng)


# Install thread-safe hook once into fmax_search module namespace
fms.passes_at = dispatching_passes_at


class Algorithm1ExecutionTracer:
    """
    Context manager that intercepts passes_at for the current thread during windowed_binary_search.
    Labels each test query with the exact Algorithm 1 phase:
      - Initial Window
      - Upper Extension
      - Lower Fallback
      - Bracket Search
      - Full Search Fallback
    """
    def __init__(self, win_low: int, win_high: int, is_full_fallback: bool = False):
        self.win_low = int(win_low)
        self.win_high = int(win_high)
        self.is_full_fallback = is_full_fallback
        self.history: List[Dict[str, Any]] = []
        self.best_pass: Optional[int] = None
        self.stride = 1
        self.in_extension = False
        self.in_bracket = False

    def __enter__(self):
        _local.current_tracer = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.current_tracer = None

    def record_and_evaluate(self,
                            freq: int,
                            fmax_true: float,
                            noise_sigma: float = 0.0,
                            rng: Optional[np.random.Generator] = None) -> bool:
        passed = _orig_passes_at(freq, fmax_true, noise_sigma=noise_sigma, rng=rng)
        if passed:
            if self.best_pass is None or freq > self.best_pass:
                self.best_pass = int(freq)

        if self.is_full_fallback:
            phase = "full_search_fallback"
            phase_label = "Full Search Fallback [1600–2400 MHz]"
        elif self.in_bracket:
            phase = "bracket_search"
            phase_label = f"Bracket Search"
        elif self.in_extension:
            phase = "upper_extension"
            phase_label = f"Upper Extension (Stride {self.stride}: +{self.stride * 10} MHz probe)"
            if passed:
                self.stride *= 2
            else:
                self.in_bracket = True
        elif freq < self.win_low:
            phase = "lower_fallback"
            phase_label = f"Lower Fallback (< {self.win_low} MHz)"
        elif freq > self.win_high:
            self.in_extension = True
            phase = "upper_extension"
            phase_label = f"Upper Extension (Stride {self.stride}: +{self.stride * 10} MHz probe)"
            if passed:
                self.stride *= 2
            else:
                self.in_bracket = True
        else:
            phase = "initial_window"
            phase_label = f"Initial Window [{self.win_low}–{self.win_high} MHz]"

        self.history.append({
            "step": len(self.history) + 1,
            "step_num": len(self.history) + 1,
            "freq_mhz": int(freq),
            "frequency_tested_mhz": float(freq),
            "passed": bool(passed),
            "phase": phase,
            "phase_label": phase_label,
            "best_pass_so_far": self.best_pass,
            "best_fmax": self.best_pass
        })
        return passed


class ConventionalExecutionTracer:
    """
    Context manager that intercepts passes_at for the current thread during conventional_binary_search.
    """
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.best_pass: Optional[int] = None

    def __enter__(self):
        _local.current_tracer = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.current_tracer = None

    def record_and_evaluate(self,
                            freq: int,
                            fmax_true: float,
                            noise_sigma: float = 0.0,
                            rng: Optional[np.random.Generator] = None) -> bool:
        passed = _orig_passes_at(freq, fmax_true, noise_sigma=noise_sigma, rng=rng)
        if passed:
            if self.best_pass is None or freq > self.best_pass:
                self.best_pass = int(freq)

        self.history.append({
            "step": len(self.history) + 1,
            "step_num": len(self.history) + 1,
            "freq_mhz": int(freq),
            "frequency_tested_mhz": float(freq),
            "passed": bool(passed),
            "phase": "Full Grid Binary Search",
            "phase_label": "Conventional Full Grid [1600–2400 MHz]",
            "best_pass_so_far": self.best_pass,
            "best_fmax": self.best_pass
        })
        return passed


# ---------------------------------------------------------------------------
# Search Service
# ---------------------------------------------------------------------------
class SearchService:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.train_envelope = TRAIN_ENVELOPE

    def check_ood(self, features: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Check if any feature is outside the empirical training bounding envelope.
        Returns: (is_ood, list_of_violations)
        """
        if not features or not self.train_envelope:
            return False, []
        violations = []
        for col, (lo, hi) in self.train_envelope.items():
            if col in features and features[col] is not None:
                try:
                    val = float(features[col])
                    if val < lo or val > hi:
                        violations.append(f"{col}: {val:.4f} outside [{lo:.4f}, {hi:.4f}]")
                except (ValueError, TypeError):
                    pass
        return len(violations) > 0, violations

    def trace_conventional_search(self,
                                  fmax_true_mhz: float,
                                  noise_sigma: float = DEFAULT_NOISE_SIGMA,
                                  random_seed: int = 42) -> Dict[str, Any]:
        """
        Execute baseline conventional binary search reusing fmax_search.conventional_binary_search directly.
        """
        rng = np.random.default_rng(random_seed)
        with ConventionalExecutionTracer() as tracer:
            actual_res = conventional_binary_search(
                fmax_true_mhz, FREQ_GRID, noise_sigma=noise_sigma, rng=rng
            )

        speed_bin = map_freq_to_speed_bin(actual_res.measured_fmax_mhz)

        return {
            "measured_fmax_mhz": actual_res.measured_fmax_mhz,
            "final_fmax_mhz": actual_res.measured_fmax_mhz,
            "n_tests": actual_res.n_tests,
            "test_steps_count": actual_res.n_tests,
            "success": actual_res.success,
            "speed_bin": speed_bin,
            "search_range": [int(FREQ_MIN), int(FREQ_MAX)],
            "steps": tracer.history,
            "test_trace": tracer.history
        }

    def trace_windowed_search(self,
                               pred_fmax_mhz: float,
                               window_half_mhz: float,
                               fmax_true_mhz: float,
                               noise_sigma: float = DEFAULT_NOISE_SIGMA,
                               random_seed: int = 42,
                               is_ood: bool = False,
                               ood_reasons: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute AI-windowed binary search (Algorithm 1) reusing fmax_search.windowed_binary_search.
        If is_ood is True, executes safe fallback (conventional full grid search) with phase label Full Search Fallback.
        """
        rng = np.random.default_rng(random_seed)
        grid_min = int(FREQ_GRID[0])
        grid_max = int(FREQ_GRID[-1])

        raw_low = pred_fmax_mhz - window_half_mhz
        raw_high = pred_fmax_mhz + window_half_mhz
        clamped_low = max(raw_low, float(grid_min))
        clamped_high = min(raw_high, float(grid_max))
        win_low = snap_to_grid(clamped_low, FREQ_GRID, "floor")
        win_high = snap_to_grid(clamped_high, FREQ_GRID, "ceil")

        if win_low >= win_high:
            win_low = snap_to_grid(pred_fmax_mhz, FREQ_GRID, "floor")
            win_high = snap_to_grid(pred_fmax_mhz, FREQ_GRID, "ceil")
            if win_low == win_high:
                win_high = min(win_high + FREQ_STEP, grid_max)

        if is_ood:
            # Safe Fallback: Execute full conventional binary search under Algorithm 1 fallback rules
            with Algorithm1ExecutionTracer(win_low, win_high, is_full_fallback=True) as tracer:
                actual_res = conventional_binary_search(
                    fmax_true_mhz, FREQ_GRID, noise_sigma=noise_sigma, rng=rng
                )
            search_type = "full_search_fallback"
        else:
            # Normal Algorithm 1: Execute windowed binary search
            with Algorithm1ExecutionTracer(win_low, win_high, is_full_fallback=False) as tracer:
                actual_res = windowed_binary_search(
                    pred_fmax_mhz, window_half_mhz, fmax_true_mhz, FREQ_GRID,
                    noise_sigma=noise_sigma, rng=rng
                )
            search_type = actual_res.search_type

        speed_bin = map_freq_to_speed_bin(actual_res.measured_fmax_mhz)

        return {
            "measured_fmax_mhz": actual_res.measured_fmax_mhz,
            "final_fmax_mhz": actual_res.measured_fmax_mhz,
            "n_tests": actual_res.n_tests,
            "test_steps_count": actual_res.n_tests,
            "success": actual_res.success,
            "speed_bin": speed_bin,
            "search_type": search_type,
            "is_ood": is_ood,
            "ood_reasons": ood_reasons or [],
            "search_window": [int(actual_res.window_low_mhz), int(actual_res.window_high_mhz)],
            "raw_window": [round(raw_low, 2), round(raw_high, 2)],
            "grid_window": [int(win_low), int(win_high)],
            "raw_window_text": f"[{raw_low:.1f}, {raw_high:.1f}] MHz",
            "grid_window_text": f"[{int(win_low)}, {int(win_high)}] MHz",
            "steps": tracer.history,
            "test_trace": tracer.history
        }

    def compare_searches(self,
                         pred_fmax_mhz: float,
                         window_half_mhz: float,
                         fmax_true_mhz: float,
                         noise_sigma: float = DEFAULT_NOISE_SIGMA,
                         random_seed: int = 42,
                         is_synthetic_unit: bool = False,
                         unit_id: Optional[str] = None,
                         features: Optional[Dict[str, Any]] = None,
                         simulate_ood_drift: bool = False) -> Dict[str, Any]:
        """
        Compare conventional binary search and AI-windowed search for a chip.
        """
        # Step 1: OOD check (automatic training bounding envelope OR manual testing toggle)
        auto_ood, ood_violations = self.check_ood(features)
        is_ood = bool(simulate_ood_drift or auto_ood)
        ood_reasons = ood_violations if auto_ood else (["Manual testing mode: simulated feature drift"] if simulate_ood_drift else [])

        conventional = self.trace_conventional_search(
            fmax_true_mhz=fmax_true_mhz,
            noise_sigma=noise_sigma,
            random_seed=random_seed
        )

        windowed = self.trace_windowed_search(
            pred_fmax_mhz=pred_fmax_mhz,
            window_half_mhz=window_half_mhz,
            fmax_true_mhz=fmax_true_mhz,
            noise_sigma=noise_sigma,
            random_seed=random_seed,
            is_ood=is_ood,
            ood_reasons=ood_reasons
        )

        tests_bin = conventional["n_tests"]
        tests_win = windowed["n_tests"]
        conventional["total_tests"] = tests_bin
        conventional["trace"] = conventional["steps"]
        windowed["total_tests"] = tests_win
        windowed["trace"] = windowed["steps"]

        if tests_win < tests_bin:
            status = "reduction"
            diff_tests = tests_bin - tests_win
            diff_pct = round(((tests_bin - tests_win) / tests_bin) * 100.0, 2) if tests_bin > 0 else 0.0
            status_title = "TEST TIME REDUCTION"
            status_subtitle = f"{diff_tests} Fewer ATE Steps"
            display_reduction_pct = diff_pct
        elif tests_win > tests_bin:
            status = "increase"
            diff_tests = tests_win - tests_bin
            diff_pct = round(((tests_win - tests_bin) / tests_bin) * 100.0, 2) if tests_bin > 0 else 0.0
            status_title = "TEST TIME OVERHEAD (RECOVERY)"
            status_subtitle = f"+{diff_tests} Additional Tests (Recovery)"
            display_reduction_pct = -diff_pct
        else:
            status = "equal"
            diff_tests = 0
            diff_pct = 0.0
            status_title = "TEST STEP COMPARISON"
            status_subtitle = "Equal Test Steps (0 Saved)"
            display_reduction_pct = 0.0

        tests_saved = tests_bin - tests_win
        raw_reduction_pct = round((tests_saved / tests_bin) * 100.0, 2) if tests_bin > 0 else 0.0

        conv_fmax = conventional.get("final_fmax_mhz")
        ai_fmax = windowed.get("final_fmax_mhz")
        fmax_diff_mhz = abs(conv_fmax - ai_fmax) if (conv_fmax is not None and ai_fmax is not None) else 0.0
        repeatability_variance = bool(noise_sigma > 0 and fmax_diff_mhz > 0)

        true_speed_bin = map_freq_to_speed_bin(fmax_true_mhz)
        conv_error_mhz = round(abs(conv_fmax - fmax_true_mhz), 1) if conv_fmax is not None else None
        ai_error_mhz = round(abs(ai_fmax - fmax_true_mhz), 1) if ai_fmax is not None else None
        speed_bin_match = bool(conventional.get("speed_bin") == windowed.get("speed_bin") == true_speed_bin)

        type_descriptions = {
            "window_direct": "Direct Window Search (true Fmax resolved within the predicted window).",
            "upper_extension": "Upper-Edge Extension with Stride Doubling (true Fmax exceeded predicted upper bound).",
            "lower_fallback": "Lower-Window Fallback Search (all window frequencies failed; searched below window).",
            "full_search_fallback": "Full Search Fallback (Out-of-Distribution feature drift detected; safely searched full grid)."
        }

        return {
            "unit_id": unit_id,
            "is_synthetic_unit": is_synthetic_unit,
            "ground_truth_label": "Synthetic Ground Truth Fmax" if is_synthetic_unit else "User-Supplied Simulated Target Fmax",
            "fmax_true_mhz": fmax_true_mhz,
            "fmax_pred_mhz": pred_fmax_mhz,
            "noise_sigma_mhz": noise_sigma,
            "random_seed": random_seed,
            "raw_window": windowed["raw_window"],
            "grid_window": windowed["grid_window"],
            "raw_window_text": windowed["raw_window_text"],
            "grid_window_text": windowed["grid_window_text"],
            "fmax_diff_mhz": round(fmax_diff_mhz, 1),
            "repeatability_variance": repeatability_variance,
            "true_speed_bin": true_speed_bin,
            "conv_error_mhz": conv_error_mhz,
            "ai_error_mhz": ai_error_mhz,
            "speed_bin_match": speed_bin_match,
            "is_ood": is_ood,
            "ood_reasons": ood_reasons,
            "conventional_search": conventional,
            "ai_windowed_search": windowed,
            "comparison": {
                "tests_conventional": tests_bin,
                "tests_windowed": tests_win,
                "tests_saved": tests_saved,
                "saved_tests": tests_saved,
                "reduction_pct": raw_reduction_pct,
                "step_reduction_pct": raw_reduction_pct,
                "diff_tests": diff_tests,
                "diff_pct": diff_pct,
                "status": status,
                "status_title": status_title,
                "status_subtitle": status_subtitle,
                "search_type": windowed["search_type"],
                "search_type_description": type_descriptions.get(windowed["search_type"], "Standard search.")
            },
            "disclaimer": (
                "NOTE: These search steps are simulated using the original test oracle and noise model. "
                "In a production wafer fab, physical tester hardware executes electrical tests at each step."
            )
        }


search_service = SearchService()
