"""
predict.py
==========
Command-line interface for predicting Fmax for a new chip
given its pre-test measurements.

Usage examples:
    python predict.py --ro 0.1 -0.2 0.3 -0.1 0.4 -0.3 0.2 -0.1 0.3 -0.2 \
                      --iddq 0.5 --vmin -0.3 --die_x 0.4 --die_y 0.1 --radius_sq 0.17

    python predict.py --csv_row path/to/row.csv

The prediction is NOT a verified physical measurement.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import FEATURE_COLS, RO_COLS, EXTRA_FEATURE_COLS
from src.preprocessing import load_preprocessor, transform
from src.train_model import load_model, load_calib_info
from src.fmax_search import windowed_binary_search, snap_to_grid, FREQ_GRID


def build_feature_df(ro_values: list[float],
                     iddq_norm: float,
                     vmin_norm: float,
                     die_x: float,
                     die_y: float,
                     radius_sq: float) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from raw measurements.
    """
    if len(ro_values) != 10:
        raise ValueError(f"Expected 10 ring-oscillator values (ro_01..ro_10), got {len(ro_values)}.")

    row = {col: val for col, val in zip(RO_COLS, ro_values)}
    row["iddq_norm"]  = iddq_norm
    row["vmin_norm"]  = vmin_norm
    row["die_x"]      = die_x
    row["die_y"]      = die_y
    row["radius_sq"]  = radius_sq

    return pd.DataFrame([row])[FEATURE_COLS]


def predict_fmax(feature_df: pd.DataFrame,
                 model_dir: str | Path = "models") -> dict:
    """
    Load the trained model and return prediction + window information.

    Parameters
    ----------
    feature_df : single-row DataFrame with FEATURE_COLS columns
    model_dir  : directory containing model artefacts

    Returns
    -------
    dict with keys:
        predicted_fmax_mhz, window_low_mhz, window_high_mhz,
        window_half_mhz, sigma_calib_mhz, caution_note
    """
    model_dir = Path(model_dir)
    model        = load_model(model_dir / "gbr_model.joblib")
    preprocessor = load_preprocessor(model_dir / "preprocessor.joblib")
    calib_info   = load_calib_info(model_dir / "calib_info.json")

    X_scaled = transform(preprocessor, feature_df)
    pred_mhz = float(model.predict(X_scaled)[0])

    window_half = calib_info["window_half_mhz"]
    sigma_calib = calib_info["sigma_calib"]

    raw_low  = pred_mhz - window_half
    raw_high = pred_mhz + window_half
    grid_min = int(FREQ_GRID[0])
    grid_max = int(FREQ_GRID[-1])
    raw_low  = max(raw_low,  float(grid_min))
    raw_high = min(raw_high, float(grid_max))
    win_low  = snap_to_grid(raw_low,  FREQ_GRID, "floor")
    win_high = snap_to_grid(raw_high, FREQ_GRID, "ceil")

    return {
        "predicted_fmax_mhz": round(pred_mhz, 1),
        "window_low_mhz":     int(win_low),
        "window_high_mhz":    int(win_high),
        "window_half_mhz":    round(window_half, 2),
        "sigma_calib_mhz":    round(sigma_calib, 2),
        "caution_note": (
            "[WARNING] This is a MODEL PREDICTION, not a verified physical measurement. "
            "The actual Fmax must be determined by running a binary search test "
            "within the recommended frequency window."
        ),
    }


def print_prediction(result: dict) -> None:
    print("\n" + "="*60)
    print("  FMAX PREDICTION RESULT")
    print("="*60)
    print(f"  Predicted Fmax     : {result['predicted_fmax_mhz']:.1f} MHz")
    print(f"  Search window      : [{result['window_low_mhz']} - {result['window_high_mhz']}] MHz")
    print(f"  Window half-width  : +/-{result['window_half_mhz']:.1f} MHz")
    print(f"  sigma_calib        : {result['sigma_calib_mhz']:.1f} MHz")
    print(f"\n  [WARNING] This is a MODEL PREDICTION, not a verified physical measurement. "
          f"The actual Fmax must be determined by running a binary search test "
          f"within the recommended frequency window.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Predict Fmax for a semiconductor chip from pre-test measurements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Supply raw measurements:
  python predict.py --ro 0.1 -0.2 0.3 -0.1 0.4 -0.3 0.2 -0.1 0.3 -0.2 \\
                    --iddq 0.5 --vmin -0.3 --die_x 0.4 --die_y 0.1 --radius_sq 0.17

  # Use a pre-built CSV row:
  python predict.py --csv_row path/to/measurement.csv

  # Specify a custom model directory:
  python predict.py --ro 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \\
                    --iddq 0.0 --vmin 0.0 --die_x 0.5 --die_y 0.5 --radius_sq 0.5 \\
                    --model_dir models/
        """
    )
    parser.add_argument("--ro", nargs=10, type=float, metavar=("RO_01", "..."),
                        help="10 ring-oscillator values (ro_01 through ro_10)")
    parser.add_argument("--iddq",      type=float, help="Normalised IDDQ leakage current")
    parser.add_argument("--vmin",      type=float, help="Normalised minimum operating voltage")
    parser.add_argument("--die_x",     type=float, help="Die X coordinate on wafer")
    parser.add_argument("--die_y",     type=float, help="Die Y coordinate on wafer")
    parser.add_argument("--radius_sq", type=float, help="Squared wafer radius (die_x²+die_y²)")
    parser.add_argument("--csv_row",   type=str,   help="Path to a CSV file with a single measurement row")
    parser.add_argument("--model_dir", type=str,   default="models",
                        help="Directory containing model artefacts (default: models/)")
    parser.add_argument("--json",      action="store_true",
                        help="Output result as JSON instead of formatted text")

    args = parser.parse_args()

    # Build feature DataFrame
    if args.csv_row:
        feature_df = pd.read_csv(args.csv_row)[FEATURE_COLS]
        if len(feature_df) == 0:
            parser.error("CSV file is empty or has no recognised feature columns.")
    elif args.ro is not None:
        required = ["iddq", "vmin", "die_x", "die_y", "radius_sq"]
        missing  = [r for r in required if getattr(args, r) is None]
        if missing:
            parser.error(f"Missing required arguments: {', '.join('--'+m for m in missing)}")
        feature_df = build_feature_df(
            ro_values  = args.ro,
            iddq_norm  = args.iddq,
            vmin_norm  = args.vmin,
            die_x      = args.die_x,
            die_y      = args.die_y,
            radius_sq  = args.radius_sq,
        )
    else:
        parser.error("Provide either --ro ... --iddq ... --vmin ... --die_x --die_y --radius_sq OR --csv_row.")

    result = predict_fmax(feature_df, model_dir=args.model_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main()
