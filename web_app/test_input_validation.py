"""
web_app/test_input_validation.py
Verification script for HTML input attributes and feature precision validation.
"""

import sys
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.app import app

client = app.test_client()

def test_html_input_attributes():
    print("\n--- Verifying HTML Input Attributes on /prediction ---")
    resp = client.get("/prediction")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    features = [f"ro_{i:02d}" for i in range(1, 11)] + [
        "iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"
    ]

    for feat in features:
        # Search for <input ... id="feat" ...>
        # Match tag containing id="feat"
        pattern = re.compile(rf'<input[^>]*id="{feat}"[^>]*>', re.IGNORECASE)
        match = pattern.search(html)
        assert match is not None, f"Input tag for {feat} not found in /prediction HTML"
        tag = match.group(0)
        assert 'step="any"' in tag, f"Input {feat} is missing step=\"any\"! Tag: {tag}"
        assert 'type="number"' in tag, f"Input {feat} is not type=\"number\"! Tag: {tag}"
        print(f"[PASS] {feat:10s} -> type=\"number\", step=\"any\"")


def test_dataset_precision_and_prediction():
    print("\n--- Testing Model Prediction with Real Dataset Values (4 Decimal Precision) ---")
    df = pd.read_csv("synthetic_fmax_dataset.csv")
    features = [f"ro_{i:02d}" for i in range(1, 11)] + [
        "iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"
    ]

    # Sample rows from train, calib, and test splits
    for split in ["train", "calibration", "test"]:
        subset = df[df["split"] == split].sample(3, random_state=42)
        for idx, row in subset.iterrows():
            payload = {col: float(row[col]) for col in features}
            resp = client.post("/api/predict", json=payload)
            assert resp.status_code == 200, f"Failed for {row['unit_id']}: {resp.get_json()}"
            data = resp.get_json()["data"]
            assert data["predicted_fmax_mhz"] > 1600.0 and data["predicted_fmax_mhz"] < 2400.0
            print(f"[PASS] Split={split:5s} Unit={row['unit_id']}: RO_02={payload['ro_02']:7.4f} -> Pred={data['predicted_fmax_mhz']} MHz")

    # Specifically test user's example case: RO_02 = 0.4475
    print("\n--- Specifically Testing User's Example: RO_02 = 0.4475 ---")
    first_row = df.iloc[0]
    custom_payload = {col: float(first_row[col]) for col in features}
    custom_payload["ro_02"] = 0.4475
    resp = client.post("/api/predict", json=custom_payload)
    assert resp.status_code == 200
    res = resp.get_json()["data"]
    print(f"[PASS] Explicit test with RO_02=0.4475 succeeded -> Pred={res['predicted_fmax_mhz']} MHz, Window=[{res['window_low_mhz']}, {res['window_high_mhz']}]")


def test_all_10_ro_boundary_and_precision():
    print("\n--- Testing All 10 RO Fields with Min, Max, and Multi-Decimal Values ---")
    df = pd.read_csv("synthetic_fmax_dataset.csv")
    features = [f"ro_{i:02d}" for i in range(1, 11)] + [
        "iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"
    ]

    base_payload = {col: float(df.iloc[0][col]) for col in features}

    for i in range(1, 11):
        ro_col = f"ro_{i:02d}"
        min_val = float(df[ro_col].min())
        max_val = float(df[ro_col].max())

        # Test min
        p_min = dict(base_payload)
        p_min[ro_col] = min_val
        r_min = client.post("/api/predict", json=p_min)
        assert r_min.status_code == 200

        # Test max
        p_max = dict(base_payload)
        p_max[ro_col] = max_val
        r_max = client.post("/api/predict", json=p_max)
        assert r_max.status_code == 200

        # Test 4 decimal precision
        p_dec = dict(base_payload)
        p_dec[ro_col] = 0.4475
        r_dec = client.post("/api/predict", json=p_dec)
        assert r_dec.status_code == 200

        print(f"[PASS] {ro_col}: tested min ({min_val:.4f}), max ({max_val:.4f}), and 0.4475 successfully")


if __name__ == "__main__":
    test_html_input_attributes()
    test_dataset_precision_and_prediction()
    test_all_10_ro_boundary_and_precision()
    print("\n=======================================================")
    print("ALL INPUT ATTRIBUTE & PRECISION VALIDATIONS PASSED [OK]")
    print("=======================================================")
