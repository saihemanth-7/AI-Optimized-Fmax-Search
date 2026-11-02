"""
evaluate_model.py
=================
Full evaluation pipeline:
1. Load the trained model, preprocessor, and calibration info.
2. Predict on the held-out test set.
3. Run both search algorithms on the test set (using Algorithm 1 with independent 4 MHz noise).
4. Compute and save all metrics.
5. Generate and save all required plots.

All results come from actual computed values — no hard-coded numbers.
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import (
    load_dataset, get_splits, get_X_y,
    FEATURE_COLS, TARGET_COL, UNIT_ID_COL,
)
from src.preprocessing import load_preprocessor, transform
from src.train_model import load_model, load_calib_info
from src.fmax_search import run_search_comparison, FREQ_GRID, DEFAULT_NOISE_SIGMA
from src.metrics import (
    compute_regression_metrics, compute_search_metrics,
    print_regression_metrics, print_search_metrics,
)

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_actual_vs_predicted(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             out_dir: Path) -> None:
    """Figure 1: Actual vs Predicted Fmax scatter."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="#2196F3", label="Test samples")
    lims = [min(y_true.min(), y_pred.min()) - 20,
            max(y_true.max(), y_pred.max()) + 20]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Fmax (MHz)", fontsize=12)
    ax.set_ylabel("Predicted Fmax (MHz)", fontsize=12)
    ax.set_title("Actual vs Predicted Fmax — Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    _save_fig(fig, out_dir / "fig1_actual_vs_predicted.png")


def plot_error_distribution(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            out_dir: Path) -> None:
    """Figure 2: Prediction error distribution."""
    errors = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(errors, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color="orange", linewidth=1.5, linestyle="-",
               label=f"Mean = {errors.mean():.1f} MHz")
    ax.set_xlabel("Prediction Error (True - Predicted, MHz)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Box plot / stats
    ax2 = axes[1]
    ax2.boxplot(errors, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#90CAF9", color="#1565C0"),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="#1565C0"),
                capprops=dict(color="#1565C0"))
    ax2.set_ylabel("Prediction Error (MHz)", fontsize=11)
    ax2.set_title("Error Box Plot", fontsize=12, fontweight="bold")
    ax2.set_xticks([])
    stats_text = (
        f"MAE  = {np.abs(errors).mean():.2f} MHz\n"
        f"RMSE = {np.sqrt((errors**2).mean()):.2f} MHz\n"
        f"Std  = {errors.std():.2f} MHz\n"
        f"P5   = {np.percentile(errors,5):.1f} MHz\n"
        f"P95  = {np.percentile(errors,95):.1f} MHz"
    )
    ax2.text(1.15, errors.mean(), stats_text, transform=ax2.get_yaxis_transform(),
             fontsize=9, va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Fmax Prediction Error Analysis — Test Set",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir / "fig2_error_distribution.png")


def plot_feature_importance(model, out_dir: Path) -> None:
    """Figure 3: Feature importance from GBR."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_features)))
    bars = ax.barh(sorted_features[::-1], sorted_importance[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature Importance (MDI)", fontsize=11)
    ax.set_title("Feature Importance — GradientBoostingRegressor", fontsize=13, fontweight="bold")
    ax.set_xlim(0, sorted_importance.max() * 1.15)

    for bar, val in zip(bars, sorted_importance[::-1]):
        ax.text(bar.get_width() + sorted_importance.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir / "fig3_feature_importance.png")


def plot_search_comparison(search_metrics: dict, out_dir: Path) -> None:
    """Figure 4: Average test steps — binary vs windowed."""
    methods = ["Conventional\nBinary Search", "AI-Windowed\nBinary Search"]
    avg_tests = [
        search_metrics["avg_tests_binary_ai"],
        search_metrics["avg_tests_windowed_ai"],
    ]
    colors = ["#EF5350", "#42A5F5"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, avg_tests, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, avg_tests):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    pct = search_metrics["pct_reduction_tests"]
    ax.set_ylabel("Average Number of Test Steps", fontsize=11)
    ax.set_title(
        f"Search Step Comparison\n({pct:.1f}% reduction with AI-Windowed Search)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, max(avg_tests) * 1.25)
    ax.grid(axis="y", alpha=0.4)

    # Document reference annotations
    ax.axhline(6.45, color="#EF5350", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axhline(3.89, color="#42A5F5", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(1.5, 6.45 + 0.08, "Doc: 6.45", fontsize=8, color="#EF5350", alpha=0.8)
    ax.text(1.5, 3.89 + 0.08, "Doc: 3.89", fontsize=8, color="#42A5F5", alpha=0.8)

    fig.tight_layout()
    _save_fig(fig, out_dir / "fig4_search_comparison.png")


def plot_step_reduction(comparison_df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 5: Per-unit test-step reduction histogram."""
    reduction = (comparison_df["tests_binary_ai"] -
                 comparison_df["tests_windowed_ai"])

    fig, ax = plt.subplots(figsize=(8, 5))
    unique_vals, counts = np.unique(reduction, return_counts=True)
    colors = ["#F44336" if v < 0 else "#4CAF50" if v > 0 else "#9E9E9E"
              for v in unique_vals]
    ax.bar(unique_vals, counts, color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("Test Steps Saved (Binary - Windowed)", fontsize=11)
    ax.set_ylabel("Number of Units", fontsize=11)
    ax.set_title("Per-Unit Test-Step Reduction\n(Positive = AI saves tests)",
                 fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    avg_reduction = reduction.mean()
    ax.axvline(avg_reduction, color="navy", linestyle="-", linewidth=2,
               label=f"Mean saving = {avg_reduction:.2f} steps")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    _save_fig(fig, out_dir / "fig5_step_reduction.png")


def plot_predicted_vs_measured(comparison_df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 6: Predicted Fmax vs Verified (windowed-search) Fmax."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(comparison_df["fmax_true_mhz"],
               comparison_df["measured_fmax_windowed_ai"],
               alpha=0.25, s=8, color="#FF7043", label="Windowed-search result")
    lims = [1550, 2400]
    ax.plot(lims, lims, "b--", linewidth=1.5, label="Perfect match")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True Fmax (MHz)", fontsize=12)
    ax.set_ylabel("Windowed Search Measured Fmax (MHz)", fontsize=12)
    ax.set_title("True vs Windowed-Search Measured Fmax\n(Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    _save_fig(fig, out_dir / "fig6_predicted_vs_measured.png")


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(csv_path: str | Path,
                   model_dir: str | Path,
                   output_dir: str | Path,
                   noise_sigma: float = DEFAULT_NOISE_SIGMA,
                   random_seed: int = 42) -> dict:
    """
    Full evaluation pipeline. Returns all metrics as a dict.
    Uses Algorithm 1 with independent test-repeatability noise (sigma = noise_sigma).
    """
    csv_path   = Path(csv_path)
    model_dir  = Path(model_dir)
    output_dir = Path(output_dir)
    fig_dir    = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artefacts ---
    print("Loading model, preprocessor, and calibration info...")
    model        = load_model(model_dir / "gbr_model.joblib")
    preprocessor = load_preprocessor(model_dir / "preprocessor.joblib")
    calib_info   = load_calib_info(model_dir / "calib_info.json")
    window_half  = calib_info["window_half_mhz"]

    print(f"  Window half-width: +/-{window_half:.2f} MHz (= 1.5 x {calib_info['sigma_calib']:.2f})")
    print(f"  Test repeatability noise: sigma = {noise_sigma:.1f} MHz, seed = {random_seed}")

    # --- Load data ---
    df = load_dataset(csv_path)
    _, _, test_df = get_splits(df)
    X_test, y_test = get_X_y(test_df)
    X_test_s = transform(preprocessor, X_test)

    print(f"\nEvaluating on {len(test_df)} test samples...")

    # --- Predictions ---
    y_pred = model.predict(X_test_s)

    # --- Regression metrics ---
    reg_metrics = compute_regression_metrics(y_test.values, y_pred, label="Test Set")
    print_regression_metrics(reg_metrics)

    # --- Search comparison (Algorithm 1 with independent test noise) ---
    print("\nRunning search comparison on test set (Algorithm 1 with stride-doubling & fallback)...")
    comparison_df = run_search_comparison(
        test_df=test_df,
        model_predictions=y_pred,
        window_half_mhz=window_half,
        grid=FREQ_GRID,
        noise_sigma=noise_sigma,
        random_seed=random_seed
    )
    search_metrics = compute_search_metrics(comparison_df)
    print_search_metrics(search_metrics)

    # --- Document comparison ---
    doc_ref = {
        "rmse_mhz_doc":           26.9,
        "avg_binary_tests_doc":   6.45,
        "avg_windowed_tests_doc": 3.89,
        "pct_reduction_doc":      39.7,
    }
    print(f"\n--- Document Reference Comparison ---")
    print(f"  RMSE:               ours = {reg_metrics['rmse_mhz']:.2f} MHz, doc = {doc_ref['rmse_mhz_doc']} MHz")
    print(f"  Avg binary tests:   ours = {search_metrics['avg_tests_binary_ai']:.4f}, doc = {doc_ref['avg_binary_tests_doc']}")
    print(f"  Avg windowed tests: ours = {search_metrics['avg_tests_windowed_ai']:.4f}, doc = {doc_ref['avg_windowed_tests_doc']}")
    print(f"  % Reduction:        ours = {search_metrics['pct_reduction_tests']:.2f}%, doc = {doc_ref['pct_reduction_doc']}%")

    # --- Save CSVs ---
    comparison_df.to_csv(output_dir / "search_comparison.csv", index=False)
    print(f"\nSearch comparison saved: outputs/search_comparison.csv")

    eval_rows = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        eval_rows.append({
            "unit_id":       row.get(UNIT_ID_COL, i),
            "fmax_true_mhz": row[TARGET_COL],
            "fmax_pred_mhz": y_pred[i],
            "error_mhz":     row[TARGET_COL] - y_pred[i],
        })
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    print(f"Evaluation results saved: outputs/evaluation_results.csv")

    # --- Save metrics JSON ---
    all_metrics = {
        "regression": reg_metrics,
        "search":     search_metrics,
        "doc_reference": doc_ref,
        "calib_info": calib_info,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved: outputs/metrics.json")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_actual_vs_predicted(y_test.values, y_pred, fig_dir)
    plot_error_distribution(y_test.values, y_pred, fig_dir)
    plot_feature_importance(model, fig_dir)
    plot_search_comparison(search_metrics, fig_dir)
    plot_step_reduction(comparison_df, fig_dir)
    plot_predicted_vs_measured(comparison_df, fig_dir)
    print("All plots saved to outputs/figures/")

    return all_metrics


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    csv  = base / "synthetic_fmax_dataset.csv"
    if not csv.exists():
        csv = base / "data" / "synthetic_fmax_dataset.csv"

    run_evaluation(
        csv_path   = csv,
        model_dir  = base / "models",
        output_dir = base / "outputs",
    )
