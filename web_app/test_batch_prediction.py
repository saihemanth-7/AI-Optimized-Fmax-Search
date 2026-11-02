"""
web_app/test_batch_prediction.py
================================
Unit and integration tests for the CSV Batch Fmax Prediction feature.
Tests file upload, validation, prediction accuracy, error handling,
and boundary conditions (up to 1,000 records).
"""

import io
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.app import app
from src.data_loader import FEATURE_COLS

client = app.test_client()


def _get_sample_data(num_records: int = 10, use_unit_id: bool = False) -> pd.DataFrame:
    """Helper to load real features from synthetic_fmax_dataset.csv."""
    dataset_path = PROJECT_ROOT / "synthetic_fmax_dataset.csv"
    raw_df = pd.read_csv(dataset_path)
    
    # If num_records > len(raw_df), repeat
    if num_records > len(raw_df):
        repeats = int(np.ceil(num_records / len(raw_df)))
        raw_df = pd.concat([raw_df] * repeats, ignore_index=True)
    
    subset = raw_df.head(num_records).copy()
    id_col = "unit_id" if use_unit_id else "chip_id"
    subset[id_col] = [f"CHIP_{i:04d}" for i in range(1, num_records + 1)]
    
    cols = [id_col] + FEATURE_COLS
    return subset[cols].copy()


def test_batch_prediction_page_get():
    """Verify GET /batch-prediction renders 200 OK with proper HTML structure."""
    resp = client.get("/batch-prediction")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Batch Fmax Prediction" in html
    assert "batchCsvInput" in html
    assert "btnRunBatch" in html
    assert "btnDownloadResultsCsv" in html
    assert "batchResultsTable" in html
    print("[PASS] GET /batch-prediction -> 200 OK")


def test_sample_batch_csv_download():
    """Verify GET /api/sample-batch-csv returns a downloadable CSV template."""
    resp = client.get("/api/sample-batch-csv")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    csv_text = resp.get_data(as_text=True)
    df = pd.read_csv(io.StringIO(csv_text))
    assert len(df) == 10
    assert "chip_id" in df.columns
    for feat in FEATURE_COLS:
        assert feat in df.columns
    print("[PASS] GET /api/sample-batch-csv -> valid CSV template")


def test_batch_predict_valid_csv():
    """Verify batch prediction with 10 valid chip records."""
    df = _get_sample_data(10)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "chips.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res = resp.get_json()
    assert res["status"] == "success"
    assert res["count"] == 10
    assert len(res["data"]) == 10

    required_keys = {"chip_id", "predicted_fmax_mhz", "search_window_lower_mhz", "search_window_upper_mhz"}
    for item in res["data"]:
        assert required_keys.issubset(item.keys())
        assert item["chip_id"].startswith("CHIP_")
        assert 1600.0 <= item["predicted_fmax_mhz"] <= 2400.0
        assert 1600 <= item["search_window_lower_mhz"] <= item["search_window_upper_mhz"] <= 2400
        assert item["search_window_lower_mhz"] <= item["predicted_fmax_mhz"] <= item["search_window_upper_mhz"]

    print(f"[PASS] /api/batch-predict with 10 valid records -> 200 OK (Avg: {res['summary']['avg_pred_fmax']} MHz)")


def test_batch_predict_unit_id_alias():
    """Verify batch prediction accepts 'unit_id' column as chip identifier and maps it to 'chip_id'."""
    df = _get_sample_data(5, use_unit_id=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "unit_ids.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res = resp.get_json()
    assert res["count"] == 5
    assert res["data"][0]["chip_id"] == "CHIP_0001"
    print("[PASS] 'unit_id' column accepted and mapped to 'chip_id' in output")


def test_batch_predict_up_to_1000_records():
    """Verify batch prediction processes exactly 1,000 records successfully."""
    df = _get_sample_data(1000)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "1000_chips.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res = resp.get_json()
    assert res["count"] == 1000
    assert len(res["data"]) == 1000
    print("[PASS] /api/batch-predict succeeded for exactly 1,000 records")


def test_batch_predict_rejects_over_1000_records():
    """Verify batch prediction rejects uploads exceeding 1,000 records (e.g. 1,001)."""
    df = _get_sample_data(1001)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "1001_chips.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res = resp.get_json()
    assert "exceeds the maximum limit of 1,000 records" in res["message"]
    print("[PASS] > 1,000 records rejected with clear error message")


def test_batch_predict_rejects_empty_csv():
    """Verify batch prediction rejects empty CSV files."""
    data = {"file": (io.BytesIO(b""), "empty.csv")}
    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "empty" in resp.get_json()["message"].lower()
    print("[PASS] Empty CSV rejected with HTTP 400")


def test_batch_predict_rejects_missing_feature_columns():
    """Verify batch prediction rejects CSV missing required feature columns."""
    df = _get_sample_data(5)
    df = df.drop(columns=["ro_05", "iddq_norm"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "missing_cols.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res = resp.get_json()
    assert "Missing required feature columns" in res["message"]
    assert "ro_05" in res["message"]
    assert "iddq_norm" in res["message"]
    print("[PASS] Missing required feature columns rejected with HTTP 400")


def test_batch_predict_rejects_missing_chip_id_column():
    """Verify batch prediction rejects CSV missing chip identifier column."""
    df = _get_sample_data(5)
    df = df.drop(columns=["chip_id"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "no_id.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res = resp.get_json()
    assert "Missing chip identifier column" in res["message"]
    print("[PASS] Missing chip identifier column rejected with HTTP 400")


def test_batch_predict_rejects_invalid_feature_values():
    """Verify batch prediction does NOT silently skip invalid feature values and reports row."""
    df = _get_sample_data(5)
    # Inject non-numeric value in row 3 (ro_02)
    df.at[2, "ro_02"] = "INVALID_CORRUPTED"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "invalid_vals.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res = resp.get_json()
    assert "Invalid or missing feature values detected" in res["message"]
    assert "Row 4" in res["message"]  # 1-based header row + index 2 = Row 4 in CSV
    print("[PASS] Non-numeric feature value rejected with specific row error message")


def test_batch_predict_rejects_blank_chip_id():
    """Verify batch prediction rejects blank/null chip_id."""
    df = _get_sample_data(5)
    df.at[1, "chip_id"] = "   "  # blank chip ID in row 2
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    data = {"file": (io.BytesIO(csv_bytes), "blank_id.csv")}

    resp = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res = resp.get_json()
    assert "Missing chip identifier" in res["message"]
    assert "Row 3" in res["message"]
    print("[PASS] Blank chip_id rejected with specific row error message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

