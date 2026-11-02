"""
prediction_service.py
======================
Service for loading trained models and generating Fmax predictions
using the existing preprocessing pipeline and Gradient Boosting Regressor.

Strictly preserves original model and preprocessing artifacts.
Loads artifacts ONCE at module import / server initialization.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

# Resolve project root (parent directory of web_app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.data_loader import FEATURE_COLS
from src.preprocessing import load_preprocessor, transform
from src.train_model import load_model, load_calib_info
from src.fmax_search import snap_to_grid, FREQ_GRID, FREQ_MIN, FREQ_MAX, FREQ_STEP


class PredictionService:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.model_dir = self.project_root / "models"
        
        # Load artifacts ONCE
        self.model = load_model(self.model_dir / "gbr_model.joblib")
        self.preprocessor = load_preprocessor(self.model_dir / "preprocessor.joblib")
        self.calib_info = load_calib_info(self.model_dir / "calib_info.json")
        
        self.sigma_calib = float(self.calib_info["sigma_calib"])
        self.window_half = float(self.calib_info["window_half_mhz"])
        self.calib_multiplier = float(self.calib_info.get("calib_multiplier", 1.5))

    def validate_features(self, data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that all 15 required features are provided and are valid finite numbers.
        Returns: (is_valid, missing_keys, invalid_keys)
        """
        missing = []
        invalid = []
        for feat in FEATURE_COLS:
            if feat not in data or data[feat] is None or data[feat] == "":
                missing.append(feat)
            else:
                try:
                    val = float(data[feat])
                    if not np.isfinite(val):
                        invalid.append(feat)
                except (ValueError, TypeError):
                    invalid.append(feat)
        return (len(missing) == 0 and len(invalid) == 0), missing, invalid

    def predict_fmax(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run preprocessing and predict Fmax using the original trained pipeline.
        Calculates recommended search window based on original calibration info.
        """
        is_valid, missing, invalid = self.validate_features(feature_data)
        if not is_valid:
            err_msg = []
            if missing:
                err_msg.append(f"Missing required features: {', '.join(missing)}")
            if invalid:
                err_msg.append(f"Invalid non-numeric values in features: {', '.join(invalid)}")
            raise ValueError("; ".join(err_msg))

        # Build feature DataFrame with exact original feature order
        ordered_data = {feat: [float(feature_data[feat])] for feat in FEATURE_COLS}
        feature_df = pd.DataFrame(ordered_data)[FEATURE_COLS]

        # Apply original fitted preprocessor
        X_scaled = transform(self.preprocessor, feature_df)

        # Run original model inference
        pred_mhz = float(self.model.predict(X_scaled)[0])

        # Compute original search window with grid snapping
        raw_low = pred_mhz - self.window_half
        raw_high = pred_mhz + self.window_half

        grid_min = int(FREQ_GRID[0])
        grid_max = int(FREQ_GRID[-1])

        clamped_low = max(raw_low, float(grid_min))
        clamped_high = min(raw_high, float(grid_max))

        win_low = snap_to_grid(clamped_low, FREQ_GRID, "floor")
        win_high = snap_to_grid(clamped_high, FREQ_GRID, "ceil")

        if win_low >= win_high:
            win_low = snap_to_grid(pred_mhz, FREQ_GRID, "floor")
            win_high = snap_to_grid(pred_mhz, FREQ_GRID, "ceil")
            if win_low == win_high:
                win_high = min(win_high + FREQ_STEP, grid_max)

        candidate_pts = int((win_high - win_low) / FREQ_STEP) + 1
        grid_center = snap_to_grid(pred_mhz, FREQ_GRID, "nearest")
        space_reduction = round((1.0 - (win_high - win_low) / (FREQ_MAX - FREQ_MIN)) * 100.0, 1)

        return {
            "predicted_fmax_mhz": round(pred_mhz, 1),
            "pred_fmax_mhz": round(pred_mhz, 1),
            "grid_center_mhz": round(grid_center, 1),
            "window_low_mhz": int(win_low),
            "window_high_mhz": int(win_high),
            "window_width_mhz": int(win_high - win_low),
            "candidate_points": candidate_pts,
            "space_reduction_pct": space_reduction,
            "raw_low_mhz": round(raw_low, 2),
            "raw_high_mhz": round(raw_high, 2),
            "sigma_calib_mhz": round(self.sigma_calib, 2),
            "window_half_mhz": round(self.window_half, 2),
            "confidence_factor": f"{self.calib_multiplier:.1f}x sigma_calib",
            "model_info": {
                "algorithm": "GradientBoostingRegressor (n_estimators=300, max_depth=4)",
                "target": "fmax_true_mhz",
                "features_count": len(FEATURE_COLS)
            },
            "disclaimer": (
                "This value is a MACHINE LEARNING PREDICTION used to narrow the search window. "
                "It is not a verified physical measurement. Physical verification on ATE hardware "
                "or simulated search against ground truth is required to determine measured Fmax."
            )
        }

    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and predict Fmax and search window for a batch of chips (up to 1,000 records).

        Parameters
        ----------
        df : pd.DataFrame containing chip_id (or unit_id) and the 15 feature columns.

        Returns
        -------
        Tuple of (result_df, summary_dict)
        """
        if df is None or len(df) == 0:
            raise ValueError("CSV file is empty. Please provide a valid CSV with at least one record.")

        if len(df) > 1000:
            raise ValueError(
                f"Uploaded CSV contains {len(df):,} records, which exceeds the maximum limit of 1,000 records."
            )

        # Determine chip_id column
        id_col = None
        if "chip_id" in df.columns:
            id_col = "chip_id"
        elif "unit_id" in df.columns:
            id_col = "unit_id"
        else:
            raise ValueError(
                "Missing chip identifier column. The CSV must include 'chip_id' (or 'unit_id')."
            )

        # Check for missing feature columns
        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {', '.join(missing_cols)}")

        # Check for empty/missing chip_id
        null_ids = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
        if null_ids.any():
            bad_idx = df.index[null_ids].tolist()[:5]
            row_nums = [i + 2 for i in bad_idx]
            raise ValueError(f"Missing chip identifier in row(s): {', '.join(f'Row {r}' for r in row_nums)}.")

        # Validate numeric feature values and check for NaNs/non-numeric
        invalid_rows = []
        for feat in FEATURE_COLS:
            col_series = df[feat]
            numeric_series = pd.to_numeric(col_series, errors="coerce")
            bad_mask = numeric_series.isna() | ~np.isfinite(numeric_series)
            if bad_mask.any():
                bad_indices = df.index[bad_mask].tolist()
                for idx in bad_indices[:5]:
                    invalid_rows.append(f"Row {idx + 2} ({feat}='{df.at[idx, feat]}')")

        if invalid_rows:
            unique_samples = list(dict.fromkeys(invalid_rows))[:5]
            raise ValueError(
                f"Invalid or missing feature values detected: {'; '.join(unique_samples)}. "
                "All 15 feature columns must contain valid finite numbers."
            )

        # Extract feature matrix and convert to float
        X_df = df[FEATURE_COLS].astype(float)

        # Transform using existing fitted preprocessor
        X_scaled = transform(self.preprocessor, X_df)

        # Run model inference using existing trained model
        preds = self.model.predict(X_scaled)

        grid_min = int(FREQ_GRID[0])
        grid_max = int(FREQ_GRID[-1])

        win_lows = []
        win_highs = []
        pred_fmax_list = []

        for p in preds:
            p_val = float(p)
            pred_fmax_list.append(round(p_val, 1))

            raw_low = p_val - self.window_half
            raw_high = p_val + self.window_half

            clamped_low = max(raw_low, float(grid_min))
            clamped_high = min(raw_high, float(grid_max))

            win_low = snap_to_grid(clamped_low, FREQ_GRID, "floor")
            win_high = snap_to_grid(clamped_high, FREQ_GRID, "ceil")

            if win_low >= win_high:
                win_low = snap_to_grid(p_val, FREQ_GRID, "floor")
                win_high = snap_to_grid(p_val, FREQ_GRID, "ceil")
                if win_low == win_high:
                    win_high = min(win_high + FREQ_STEP, grid_max)

            win_lows.append(int(win_low))
            win_highs.append(int(win_high))

        result_df = pd.DataFrame({
            "chip_id": df[id_col].astype(str).values,
            "predicted_fmax_mhz": pred_fmax_list,
            "search_window_lower_mhz": win_lows,
            "search_window_upper_mhz": win_highs,
        })

        summary = {
            "total_records": len(result_df),
            "min_pred_fmax": round(float(np.min(pred_fmax_list)), 1),
            "max_pred_fmax": round(float(np.max(pred_fmax_list)), 1),
            "avg_pred_fmax": round(float(np.mean(pred_fmax_list)), 1),
        }

        return result_df, summary


# Global singleton instance loaded once
prediction_service = PredictionService()

