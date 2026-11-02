"""
tests/test_data_validation.py
=============================
Tests for dataset loading, schema validation, split integrity, and leakage checks.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_dataset, get_splits, get_X_y,
    FEATURE_COLS, TARGET_COL, SPLIT_COL, EXCLUDED_COLS,
    UNIT_ID_COL, RO_COLS, EXTRA_FEATURE_COLS,
    SPLIT_TRAIN, SPLIT_CALIB, SPLIT_TEST,
)
from src.data_validation import (
    validate_schema, validate_missing, validate_splits,
    validate_no_leakage, run_all_validations,
)


def _find_csv():
    root = Path(__file__).parent.parent
    for p in [root / "synthetic_fmax_dataset.csv",
              root / "data" / "synthetic_fmax_dataset.csv"]:
        if p.exists():
            return p
    pytest.skip("Dataset CSV not found — skipping integration tests.")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    def test_csv_exists(self):
        """Dataset file must exist."""
        csv = _find_csv()
        assert csv.exists()

    def test_loads_as_dataframe(self):
        """load_dataset() must return a non-empty DataFrame."""
        df = load_dataset(_find_csv())
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_row_count(self):
        """Dataset must have exactly 6000 rows."""
        df = load_dataset(_find_csv())
        assert len(df) == 6000, f"Expected 6000 rows, got {len(df)}"

    def test_column_count(self):
        """Dataset must have exactly 25 columns."""
        df = load_dataset(_find_csv())
        assert len(df.columns) == 25, f"Expected 25 columns, got {len(df.columns)}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    @pytest.fixture(scope="class")
    @classmethod
    def df(cls):
        return load_dataset(_find_csv())

    def test_feature_columns_present(self, df):
        """All FEATURE_COLS must exist in dataset."""
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        assert missing == [], f"Missing feature columns: {missing}"

    def test_target_column_present(self, df):
        assert TARGET_COL in df.columns

    def test_split_column_present(self, df):
        assert SPLIT_COL in df.columns

    def test_ro_columns_count(self):
        """Must have exactly 10 ring-oscillator columns."""
        assert len(RO_COLS) == 10

    def test_feature_columns_numeric(self, df):
        for col in FEATURE_COLS:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"Feature '{col}' is not numeric (dtype={df[col].dtype})"

    def test_target_numeric(self, df):
        assert pd.api.types.is_numeric_dtype(df[TARGET_COL])

    def test_no_missing_values_in_features(self, df):
        for col in FEATURE_COLS:
            assert df[col].isnull().sum() == 0, f"'{col}' has missing values"

    def test_no_missing_values_in_target(self, df):
        assert df[TARGET_COL].isnull().sum() == 0


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------

class TestSplits:
    @pytest.fixture(scope="class")
    @classmethod
    def splits(cls):
        df = load_dataset(_find_csv())
        return get_splits(df)

    def test_split_sizes(self, splits):
        train_df, calib_df, test_df = splits
        assert len(train_df) == 1500, f"Expected 1500 train, got {len(train_df)}"
        assert len(calib_df) == 500,  f"Expected 500 calib, got {len(calib_df)}"
        assert len(test_df)  == 4000, f"Expected 4000 test, got {len(test_df)}"

    def test_no_overlap_between_splits(self, splits):
        train_df, calib_df, test_df = splits
        train_ids = set(train_df[UNIT_ID_COL])
        calib_ids = set(calib_df[UNIT_ID_COL])
        test_ids  = set(test_df[UNIT_ID_COL])
        assert len(train_ids & calib_ids) == 0, "Train/Calib overlap!"
        assert len(train_ids & test_ids)  == 0, "Train/Test overlap!"
        assert len(calib_ids & test_ids)  == 0, "Calib/Test overlap!"

    def test_total_rows_preserved(self, splits):
        train_df, calib_df, test_df = splits
        assert len(train_df) + len(calib_df) + len(test_df) == 6000


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------

class TestLeakage:
    def test_excluded_cols_not_in_features(self):
        """No excluded (post-test / derived) column should be in FEATURE_COLS."""
        for col in EXCLUDED_COLS:
            assert col not in FEATURE_COLS, \
                f"LEAKAGE: '{col}' is in EXCLUDED_COLS but also in FEATURE_COLS!"

    def test_unit_id_not_in_features(self):
        assert UNIT_ID_COL not in FEATURE_COLS

    def test_target_not_in_features(self):
        assert TARGET_COL not in FEATURE_COLS

    def test_fmax_pred_not_in_features(self):
        assert "fmax_pred_mhz" not in FEATURE_COLS

    def test_window_cols_not_in_features(self):
        assert "window_low_mhz"  not in FEATURE_COLS
        assert "window_high_mhz" not in FEATURE_COLS

    def test_post_test_cols_not_in_features(self):
        assert "tests_binary"             not in FEATURE_COLS
        assert "tests_windowed"           not in FEATURE_COLS
        assert "fmax_measured_windowed_mhz" not in FEATURE_COLS
        assert "true_bin_mhz"             not in FEATURE_COLS

    def test_validation_passes(self):
        df = load_dataset(_find_csv())
        issues = validate_no_leakage(df)
        errors = [i for i in issues if "LEAKAGE RISK" in i]
        assert errors == [], f"Leakage detected: {errors}"

    def test_full_validation_passes(self):
        df = load_dataset(_find_csv())
        ok = run_all_validations(df, verbose=False)
        assert ok, "Data validation failed — check dataset schema."

