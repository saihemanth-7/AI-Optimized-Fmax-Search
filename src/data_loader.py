"""
data_loader.py
==============
Loads the synthetic_fmax_dataset.csv and returns clean, split-aware DataFrames.

No target-leaking columns are ever passed to model training.
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# All column names discovered from the actual CSV file.
UNIT_ID_COL = "unit_id"
SPLIT_COL = "split"

# Pre-test feature columns — only measurements available before Fmax prediction.
# ring-oscillator speed measurements (normalised / scaled)
RO_COLS = [f"ro_{i:02d}" for i in range(1, 11)]  # ro_01 … ro_10

# Additional pre-test electrical measurements
EXTRA_FEATURE_COLS = [
    "iddq_norm",   # normalised IDDQ leakage current (pre-test, die-level)
    "vmin_norm",   # normalised minimum operating voltage (pre-test)
    "die_x",       # die X coordinate on wafer (pre-test metadata)
    "die_y",       # die Y coordinate on wafer (pre-test metadata)
    "radius_sq",   # squared wafer radius (derived from die_x, die_y; pre-test)
]

# Final feature list used for model training
FEATURE_COLS = RO_COLS + EXTRA_FEATURE_COLS

# Prediction target
TARGET_COL = "fmax_true_mhz"

# Columns that are POST-test or derived from predictions — NEVER used as features.
EXCLUDED_COLS = [
    "fmax_pred_mhz",           # prediction already made by document authors
    "window_low_mhz",          # derived from prediction
    "window_high_mhz",         # derived from prediction
    "tests_binary",            # test outcome
    "tests_windowed",          # test outcome
    "true_bin_mhz",            # coarse bin — post-test label
    "fmax_measured_windowed_mhz",  # measured result after windowed search
]

# Expected split labels
SPLIT_TRAIN = "train"
SPLIT_CALIB = "calibration"
SPLIT_TEST  = "test"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV and return it unchanged.

    Parameters
    ----------
    csv_path : str or Path
        Path to synthetic_fmax_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Full dataset, all columns retained.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Please ensure 'synthetic_fmax_dataset.csv' is in the project folder or data/ subfolder."
        )
    df = pd.read_csv(csv_path)
    return df


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train / calibration / test subsets using the
    'split' column already present in the CSV.  No rows are randomly re-assigned.

    Returns
    -------
    train_df, calib_df, test_df
    """
    if SPLIT_COL not in df.columns:
        raise KeyError(f"Expected column '{SPLIT_COL}' not found in dataset.")

    train_df = df[df[SPLIT_COL] == SPLIT_TRAIN].copy()
    calib_df = df[df[SPLIT_COL] == SPLIT_CALIB].copy()
    test_df  = df[df[SPLIT_COL] == SPLIT_TEST].copy()

    # Sanity check
    total = len(train_df) + len(calib_df) + len(test_df)
    if total != len(df):
        unknown = df[~df[SPLIT_COL].isin([SPLIT_TRAIN, SPLIT_CALIB, SPLIT_TEST])]
        print(
            f"[WARNING] {len(unknown)} rows have unrecognised split labels: "
            f"{unknown[SPLIT_COL].unique()}"
        )

    return train_df, calib_df, test_df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from a split DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A split subset (train, calib, or test).

    Returns
    -------
    X : pd.DataFrame  — shape (n, len(FEATURE_COLS))
    y : pd.Series     — fmax_true_mhz values
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Feature column(s) missing from dataset: {missing}")
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


if __name__ == "__main__":
    import sys
    csv_path = Path(__file__).parent.parent / "synthetic_fmax_dataset.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data" / "synthetic_fmax_dataset.csv"

    df = load_dataset(csv_path)
    train_df, calib_df, test_df = get_splits(df)

    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Train:       {len(train_df):>5} rows")
    print(f"  Calibration: {len(calib_df):>5} rows")
    print(f"  Test:        {len(test_df):>5} rows")
    print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Target: {TARGET_COL}")

