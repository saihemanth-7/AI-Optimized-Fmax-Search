"""
metrics.py
==========
Centralized metric calculations for the Fmax project.
All metrics are computed from actual data — no hard-coded values.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               label: str = "") -> dict:
    """
    Compute MAE, RMSE, R², and error statistics.

    Parameters
    ----------
    y_true : array of true Fmax values (MHz)
    y_pred : array of predicted Fmax values (MHz)
    label  : optional label for the metric set

    Returns
    -------
    dict with keys: mae_mhz, rmse_mhz, r2, mean_error_mhz, std_error_mhz, n
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)

    return {
        "label":          label,
        "n":              len(y_true),
        "mae_mhz":        round(float(mae),  4),
        "rmse_mhz":       round(float(rmse), 4),
        "r2":             round(float(r2),   6),
        "mean_error_mhz": round(float(errors.mean()), 4),
        "std_error_mhz":  round(float(errors.std()),  4),
        "p5_error_mhz":   round(float(np.percentile(errors,  5)), 2),
        "p95_error_mhz":  round(float(np.percentile(errors, 95)), 2),
    }


def map_freq_to_speed_bin(freq_mhz: float | None) -> str:
    """
    Map an Fmax frequency (MHz) to its standardized commercial speed bin.
    Speed bins: [1800, 1900), [1900, 2000), [2000, 2100), [2100, 2200), [2200+), or reject (< 1800).
    """
    if freq_mhz is None or np.isnan(freq_mhz) or freq_mhz < 1800:
        return "reject"
    elif freq_mhz < 1900:
        return "1800"
    elif freq_mhz < 2000:
        return "1900"
    elif freq_mhz < 2100:
        return "2000"
    elif freq_mhz < 2200:
        return "2100"
    else:
        return "2200"


def compute_search_metrics(comparison_df: pd.DataFrame) -> dict:
    """
    Compute search-step statistics from the comparison DataFrame.

    Parameters
    ----------
    comparison_df : output of fmax_search.run_search_comparison()

    Returns
    -------
    dict with search metrics and reduction statistics
    """
    bin_tests = comparison_df["tests_binary_ai"].values.astype(float)
    win_tests = comparison_df["tests_windowed_ai"].values.astype(float)

    avg_binary   = float(np.mean(bin_tests))
    avg_windowed = float(np.mean(win_tests))

    p50_binary   = float(np.percentile(bin_tests, 50))
    p90_binary   = float(np.percentile(bin_tests, 90))
    p95_binary   = float(np.percentile(bin_tests, 95))
    max_binary   = int(np.max(bin_tests))

    p50_windowed = float(np.percentile(win_tests, 50))
    p90_windowed = float(np.percentile(win_tests, 90))
    p95_windowed = float(np.percentile(win_tests, 95))
    max_windowed = int(np.max(win_tests))

    if avg_binary == 0:
        pct_reduction = 0.0
        abs_reduction = 0.0
    else:
        abs_reduction = avg_binary - avg_windowed
        pct_reduction = (abs_reduction / avg_binary) * 100.0

    completion_rate_binary   = float(comparison_df["success_binary_ai"].mean())
    completion_rate_windowed = float(comparison_df["success_windowed_ai"].mean())

    # Speed-bin accuracy against ground truth true_bin_mhz if present
    speed_bin_accuracy = None
    if "true_bin_mhz" in comparison_df.columns:
        measured_bins = comparison_df["measured_fmax_windowed_ai"].apply(map_freq_to_speed_bin)
        speed_bin_accuracy = float((measured_bins == comparison_df["true_bin_mhz"].astype(str)).mean())

    # Frequency measurement error (measured Fmax vs true continuous Fmax)
    meas_series = comparison_df["measured_fmax_windowed_ai"].dropna()
    true_series = comparison_df.loc[meas_series.index, "fmax_true_mhz"]
    freq_mae  = float(mean_absolute_error(true_series, meas_series))
    freq_rmse = float(np.sqrt(mean_squared_error(true_series, meas_series)))

    # Reference values from the CSV (pre-computed by document authors)
    csv_binary_avg   = None
    csv_windowed_avg = None
    if "tests_binary_csv" in comparison_df.columns:
        csv_binary_avg   = float(comparison_df["tests_binary_csv"].mean())
    if "tests_windowed_csv" in comparison_df.columns:
        csv_windowed_avg = float(comparison_df["tests_windowed_csv"].mean())

    # Search types breakdown if present
    search_type_counts = {}
    if "search_type_ai" in comparison_df.columns:
        search_type_counts = comparison_df["search_type_ai"].value_counts().to_dict()

    return {
        "n_test_units":              len(comparison_df),
        "avg_tests_binary_ai":       round(avg_binary,   4),
        "avg_tests_windowed_ai":     round(avg_windowed, 4),
        "abs_reduction_tests":       round(abs_reduction, 4),
        "pct_reduction_tests":       round(pct_reduction, 2),
        "p50_tests_binary":          p50_binary,
        "p95_tests_binary":          p95_binary,
        "max_tests_binary":          max_binary,
        "p50_tests_windowed":        p50_windowed,
        "p95_tests_windowed":        p95_windowed,
        "max_tests_windowed":        max_windowed,
        "completion_rate_binary":    round(completion_rate_binary,   4),
        "completion_rate_windowed":  round(completion_rate_windowed, 4),
        "speed_bin_accuracy":        round(speed_bin_accuracy, 4) if speed_bin_accuracy is not None else None,
        "measured_freq_mae_mhz":     round(freq_mae, 4),
        "measured_freq_rmse_mhz":    round(freq_rmse, 4),
        "search_type_counts":        search_type_counts,
        "avg_tests_binary_csv":      round(csv_binary_avg,   4) if csv_binary_avg   is not None else None,
        "avg_tests_windowed_csv":    round(csv_windowed_avg, 4) if csv_windowed_avg is not None else None,
    }


def print_regression_metrics(m: dict) -> None:
    label = m.get("label", "")
    print(f"\n{'='*55}")
    print(f"Regression Metrics{(' — ' + label) if label else ''}")
    print(f"{'='*55}")
    print(f"  N samples    : {m['n']}")
    print(f"  MAE          : {m['mae_mhz']:.2f} MHz")
    print(f"  RMSE         : {m['rmse_mhz']:.2f} MHz")
    print(f"  R²           : {m['r2']:.4f}")
    print(f"  Mean error   : {m['mean_error_mhz']:.2f} MHz  (bias)")
    print(f"  Std error    : {m['std_error_mhz']:.2f} MHz")
    print(f"  P5  error    : {m['p5_error_mhz']:.2f} MHz")
    print(f"  P95 error    : {m['p95_error_mhz']:.2f} MHz")


def print_search_metrics(m: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Search Step Comparison (on {m['n_test_units']} test units)")
    print(f"{'='*60}")
    print(f"  Conventional binary search (our impl)  : {m['avg_tests_binary_ai']:.4f} avg tests")
    print(f"  AI-windowed binary search  (our impl)  : {m['avg_tests_windowed_ai']:.4f} avg tests")
    print(f"  Absolute test-step reduction           : {m['abs_reduction_tests']:.4f} tests")
    print(f"  Percentage test-step reduction         : {m['pct_reduction_tests']:.2f}%")
    print(f"  Percentiles (windowed)                 : P50={m['p50_tests_windowed']:.1f}, P95={m['p95_tests_windowed']:.1f}, Max={m['max_tests_windowed']}")
    print(f"  Search completion rate (windowed)      : {m['completion_rate_windowed']*100:.1f}%")
    if m.get("speed_bin_accuracy") is not None:
        print(f"  Speed-bin classification accuracy      : {m['speed_bin_accuracy']*100:.2f}%")
    print(f"  Measured frequency MAE                 : {m['measured_freq_mae_mhz']:.2f} MHz")
    print(f"  Measured frequency RMSE                : {m['measured_freq_rmse_mhz']:.2f} MHz")
    if m.get("search_type_counts"):
        print(f"  Search types breakdown                 : {m['search_type_counts']}")
    if m.get("avg_tests_binary_csv") is not None:
        print(f"\n  --- Reference (CSV pre-computed) ---")
        print(f"  Binary search avg (CSV)    : {m['avg_tests_binary_csv']:.4f}")
        print(f"  Windowed search avg (CSV)  : {m['avg_tests_windowed_csv']:.4f}")
        csv_pct = (m['avg_tests_binary_csv'] - m['avg_tests_windowed_csv']) / m['avg_tests_binary_csv'] * 100
        print(f"  Reduction from CSV values  : {csv_pct:.2f}%")
    print(f"\n  Document reference values:")
    print(f"    Binary search     : 6.45 avg tests")
    print(f"    Windowed search   : 3.89 avg tests")
    print(f"    Reduction         : ~39.7%")
