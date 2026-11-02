"""
train_model.py
==============
Trains a Gradient Boosting Regressor (sklearn.ensemble.GradientBoostingRegressor)
to predict Fmax from pre-test measurements.

Calibration
-----------
After training, the model is run on the CALIBRATION set to compute the residual
standard deviation (sigma_calib).  This value is used by the windowed search:

    window_half = 1.5 * sigma_calib

The factor 1.5 is confirmed by the dataset: the pre-computed window widths are
consistently ±39.2 MHz, and 1.5 × calibration_std ≈ 39.24 MHz.

No test-set information is used during calibration or training.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import (
    load_dataset, get_splits, get_X_y,
    FEATURE_COLS, TARGET_COL,
)
from src.preprocessing import fit_preprocessor, transform, save_preprocessor

# ---------------------------------------------------------------------------
# Model hyper-parameters
# Consistent with typical gradient-boosting configurations for regression;
# chosen to be reproducible and well-regularised.
# ---------------------------------------------------------------------------
GBR_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42,
    loss="squared_error",
)

# Calibration multiplier (from document and dataset verification)
CALIB_MULTIPLIER = 1.5


def train_gbr(X_train_scaled: np.ndarray, y_train: np.ndarray) -> GradientBoostingRegressor:
    """Fit and return a GradientBoostingRegressor."""
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(X_train_scaled, y_train)
    return model


def calibrate(model: GradientBoostingRegressor,
              X_calib_scaled: np.ndarray,
              y_calib: np.ndarray) -> dict:
    """
    Compute calibration statistics from the calibration set.

    Returns a dict with:
      - sigma_calib   : residual standard deviation (MHz)
      - window_half   : half-width of the prediction window (MHz)
      - mean_error    : mean signed residual (bias estimate, MHz)
      - calib_mae     : MAE on calibration set (MHz)
    """
    preds = model.predict(X_calib_scaled)
    residuals = y_calib - preds  # true - predicted

    sigma_calib = float(np.std(residuals))
    mean_error  = float(np.mean(residuals))
    calib_mae   = float(mean_absolute_error(y_calib, preds))
    calib_rmse  = float(np.sqrt(mean_squared_error(y_calib, preds)))
    window_half = CALIB_MULTIPLIER * sigma_calib

    calib_info = {
        "sigma_calib":       sigma_calib,
        "window_half_mhz":   window_half,
        "calib_multiplier":  CALIB_MULTIPLIER,
        "mean_error_mhz":    mean_error,
        "calib_mae_mhz":     calib_mae,
        "calib_rmse_mhz":    calib_rmse,
    }
    return calib_info


def evaluate_on_split(model: GradientBoostingRegressor,
                      X_scaled: np.ndarray,
                      y: np.ndarray,
                      split_name: str = "split") -> dict:
    """Evaluate the model on a given split and return metrics dict."""
    preds = model.predict(X_scaled)
    mae   = mean_absolute_error(y, preds)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    r2    = r2_score(y, preds)
    return {
        "split":   split_name,
        "mae_mhz": round(mae, 4),
        "rmse_mhz": round(rmse, 4),
        "r2":       round(r2, 6),
        "n":        len(y),
    }


def save_model(model: GradientBoostingRegressor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str | Path) -> GradientBoostingRegressor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def save_calib_info(calib_info: dict, path: str | Path) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib_info, f, indent=2)
    print(f"Calibration info saved to: {path}")


def load_calib_info(path: str | Path) -> dict:
    import json
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration info not found at {path}. Run training first.")
    with open(path) as f:
        return json.load(f)


def run_training(csv_path: str | Path,
                 model_dir: str | Path = "models") -> dict:
    """
    Full training pipeline:
    1. Load dataset and respect existing splits.
    2. Fit preprocessor on training data only.
    3. Train GBR on training data.
    4. Calibrate on calibration data.
    5. Evaluate on train, calib, and test sets.
    6. Save model, preprocessor, calibration info.

    Returns a dict of all evaluation results.
    """
    csv_path = Path(csv_path)
    model_dir = Path(model_dir)

    # --- Load ---
    df = load_dataset(csv_path)
    train_df, calib_df, test_df = get_splits(df)

    print(f"\nSplit sizes: train={len(train_df)}, calib={len(calib_df)}, test={len(test_df)}")

    X_train, y_train = get_X_y(train_df)
    X_calib, y_calib = get_X_y(calib_df)
    X_test,  y_test  = get_X_y(test_df)

    # --- Preprocess (fit on train only) ---
    print("Fitting preprocessor on training data...")
    preprocessor = fit_preprocessor(X_train)

    X_train_s = transform(preprocessor, X_train)
    X_calib_s = transform(preprocessor, X_calib)
    X_test_s  = transform(preprocessor, X_test)

    # --- Train ---
    print(f"Training GradientBoostingRegressor with params: {GBR_PARAMS}")
    model = train_gbr(X_train_s, y_train.values)

    # --- Calibrate ---
    print("Calibrating on calibration set...")
    calib_info = calibrate(model, X_calib_s, y_calib.values)
    print(f"  sigma_calib    = {calib_info['sigma_calib']:.2f} MHz")
    print(f"  window_half    = {calib_info['window_half_mhz']:.2f} MHz (= 1.5 × sigma_calib)")
    print(f"  mean_error     = {calib_info['mean_error_mhz']:.2f} MHz (bias)")
    print(f"  calib_mae      = {calib_info['calib_mae_mhz']:.2f} MHz")
    print(f"  calib_rmse     = {calib_info['calib_rmse_mhz']:.2f} MHz")

    # --- Evaluate ---
    results = {}
    for split_name, X_s, y in [
        ("train", X_train_s, y_train),
        ("calibration", X_calib_s, y_calib),
        ("test", X_test_s, y_test),
    ]:
        metrics = evaluate_on_split(model, X_s, y.values, split_name)
        results[split_name] = metrics
        print(f"\n{split_name.upper()} metrics:")
        print(f"  MAE  = {metrics['mae_mhz']:.2f} MHz")
        print(f"  RMSE = {metrics['rmse_mhz']:.2f} MHz")
        print(f"  R²   = {metrics['r2']:.4f}")

    # --- Feature importance ---
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    results["feature_importances"] = importances
    results["calib_info"] = calib_info
    results["gbr_params"] = GBR_PARAMS

    # --- Save ---
    save_model(model, model_dir / "gbr_model.joblib")
    save_preprocessor(preprocessor, model_dir / "preprocessor.joblib")
    save_calib_info(calib_info, model_dir / "calib_info.json")

    return results


if __name__ == "__main__":
    from pathlib import Path
    csv = Path(__file__).parent.parent / "synthetic_fmax_dataset.csv"
    if not csv.exists():
        csv = Path(__file__).parent.parent / "data" / "synthetic_fmax_dataset.csv"

    results = run_training(csv, model_dir=Path(__file__).parent.parent / "models")
    print("\nTraining complete.")

