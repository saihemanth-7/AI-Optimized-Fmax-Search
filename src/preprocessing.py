"""
preprocessing.py
================
Leakage-safe preprocessing pipeline.

StandardScaler is fitted ONLY on training data, then applied to calibration
and test sets.  No test-set information leaks into the scaler parameters.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data_loader import FEATURE_COLS


def build_preprocessor() -> Pipeline:
    """
    Build a sklearn Pipeline with:
      1. SimpleImputer (median strategy) — handles any unexpected NaNs
      2. StandardScaler                  — zero mean, unit variance

    The pipeline is fitted ONLY on training data.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def fit_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Fit the preprocessing pipeline on training features only.

    Parameters
    ----------
    X_train : pd.DataFrame — training feature matrix

    Returns
    -------
    Fitted Pipeline object.
    """
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train[FEATURE_COLS])
    return preprocessor


def transform(preprocessor: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Apply a fitted preprocessor to a feature matrix.

    Parameters
    ----------
    preprocessor : fitted Pipeline
    X : pd.DataFrame

    Returns
    -------
    np.ndarray of scaled features.
    """
    return preprocessor.transform(X[FEATURE_COLS])


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to: {path}")


def load_preprocessor(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run training first.")
    return joblib.load(path)


if __name__ == "__main__":
    from src.data_loader import load_dataset, get_splits, get_X_y
    from pathlib import Path

    csv = Path(__file__).parent.parent / "synthetic_fmax_dataset.csv"
    if not csv.exists():
        csv = Path(__file__).parent.parent / "data" / "synthetic_fmax_dataset.csv"

    df = load_dataset(csv)
    train_df, calib_df, test_df = get_splits(df)
    X_train, y_train = get_X_y(train_df)
    X_calib, y_calib = get_X_y(calib_df)

    prep = fit_preprocessor(X_train)
    X_train_s = transform(prep, X_train)
    X_calib_s = transform(prep, X_calib)

    print(f"X_train scaled shape: {X_train_s.shape}")
    print(f"X_calib scaled shape: {X_calib_s.shape}")
    print(f"Feature means (train): {X_train_s.mean(axis=0).round(4)}")

