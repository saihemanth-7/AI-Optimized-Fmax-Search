"""
data_validation.py
==================
Validates the loaded dataset for schema correctness, missing values,
split integrity, and data-leakage risks before training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import (
    FEATURE_COLS, TARGET_COL, SPLIT_COL, EXCLUDED_COLS, UNIT_ID_COL,
    SPLIT_TRAIN, SPLIT_CALIB, SPLIT_TEST,
)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Check that all required columns exist and have expected dtypes.
    Returns a list of warning/error strings (empty = all good).
    """
    issues = []

    required = [SPLIT_COL, TARGET_COL] + FEATURE_COLS
    for col in required:
        if col not in df.columns:
            issues.append(f"MISSING COLUMN: '{col}' not found in dataset.")

    # Check numeric features
    for col in FEATURE_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"TYPE ERROR: Feature '{col}' is not numeric (dtype={df[col].dtype}).")

    if TARGET_COL in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        issues.append(f"TYPE ERROR: Target '{TARGET_COL}' is not numeric.")

    return issues


def validate_missing(df: pd.DataFrame) -> list[str]:
    """Check for missing values in features and target."""
    issues = []
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                issues.append(f"MISSING VALUES: '{col}' has {n_missing} NaN(s).")
    return issues


def validate_splits(df: pd.DataFrame) -> list[str]:
    """Validate that the split column contains expected labels and reasonable counts."""
    issues = []
    if SPLIT_COL not in df.columns:
        return [f"MISSING COLUMN: '{SPLIT_COL}' not found."]

    counts = df[SPLIT_COL].value_counts()
    for label in [SPLIT_TRAIN, SPLIT_CALIB, SPLIT_TEST]:
        if label not in counts:
            issues.append(f"SPLIT ERROR: Label '{label}' not found in '{SPLIT_COL}' column.")
        else:
            issues.append(f"INFO: Split '{label}' -> {counts[label]} rows.")

    unknown = set(df[SPLIT_COL].unique()) - {SPLIT_TRAIN, SPLIT_CALIB, SPLIT_TEST}
    if unknown:
        issues.append(f"SPLIT WARNING: Unknown split labels found: {unknown}.")

    return issues


def validate_no_leakage(df: pd.DataFrame) -> list[str]:
    """
    Verify that excluded (post-test / derived) columns are not included in FEATURE_COLS.
    Also checks that UNIT_ID_COL is excluded.
    """
    issues = []
    for col in EXCLUDED_COLS:
        if col in FEATURE_COLS:
            issues.append(f"LEAKAGE RISK: '{col}' is in EXCLUDED_COLS but also in FEATURE_COLS!")
    if UNIT_ID_COL in FEATURE_COLS:
        issues.append(f"LEAKAGE RISK: '{UNIT_ID_COL}' (unit identifier) is in FEATURE_COLS!")
    if TARGET_COL in FEATURE_COLS:
        issues.append(f"LEAKAGE RISK: Target '{TARGET_COL}' is in FEATURE_COLS!")
    if not issues:
        issues.append("INFO: No data-leakage issues detected in feature selection.")
    return issues


def validate_target_distribution(df: pd.DataFrame) -> dict:
    """Return descriptive statistics for the target column."""
    if TARGET_COL not in df.columns:
        return {}
    return df[TARGET_COL].describe().to_dict()


def run_all_validations(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Run all validations and print results.
    Returns True if no critical errors found, False otherwise.
    """
    all_issues = []
    errors = []

    checks = [
        ("Schema",        validate_schema(df)),
        ("Missing values", validate_missing(df)),
        ("Split labels",   validate_splits(df)),
        ("Leakage check",  validate_no_leakage(df)),
    ]

    for name, issues in checks:
        if verbose:
            print(f"\n--- {name} ---")
        for issue in issues:
            if verbose:
                print(f"  {issue}")
            all_issues.append(issue)
            if "ERROR" in issue or "LEAKAGE RISK" in issue or "MISSING COLUMN" in issue:
                errors.append(issue)

    if verbose:
        print(f"\n--- Target Distribution ({TARGET_COL}) ---")
        stats = validate_target_distribution(df)
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}")

    if verbose:
        print(f"\n{'='*50}")
        if errors:
            print(f"VALIDATION FAILED: {len(errors)} critical issue(s) found.")
            for e in errors:
                print(f"  [FAIL] {e}")
        else:
            print(f"VALIDATION PASSED: Dataset is ready for training. [OK]")

    return len(errors) == 0


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from pathlib import Path

    csv_path = Path(__file__).parent.parent / "synthetic_fmax_dataset.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data" / "synthetic_fmax_dataset.csv"

    df = load_dataset(csv_path)
    ok = run_all_validations(df, verbose=True)
    import sys
    sys.exit(0 if ok else 1)
