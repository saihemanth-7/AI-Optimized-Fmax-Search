"""
test_phase5_integration.py
Comprehensive verification script for Phase 5 integration testing.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.app import app
from web_app.services.results_service import results_service
from web_app.services.prediction_service import prediction_service
from web_app.services.search_service import search_service

client = app.test_client()

def test_1_flask_backend():
    print("\n--- Test 1: Flask Backend Endpoints ---")
    
    # 1.1 /api/health
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"/api/health failed with {resp.status_code}"
    health = resp.get_json()
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    assert health["preprocessor_loaded"] is True
    assert health["calib_loaded"] is True
    print(f"[PASS] /api/health -> healthy, model_loaded: {health['model_loaded']}")

    # 1.2 /api/metrics
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    metrics = resp.get_json()
    assert metrics["status"] == "success"
    assert "model" in metrics or "model_metrics" in metrics
    assert "search_comparison" in metrics
    print(f"[PASS] /api/metrics -> R2: {metrics['model']['test_r2']:.4f}, step reduction: {metrics['search_comparison']['step_reduction_pct']}%")

    # 1.3 /api/sample-units
    resp = client.get("/api/sample-units")
    assert resp.status_code == 200
    samples_data = resp.get_json()
    assert samples_data["status"] == "success"
    samples = samples_data["samples"]
    assert len(samples) == 3
    uids = [s["unit_id"] for s in samples]
    assert uids == ["U02001", "U02024", "U02008"]
    print(f"[PASS] /api/sample-units -> 3 curated units loaded: {uids}")

    # 1.4 /api/predict
    sample_u02001 = samples[0]
    resp = client.post("/api/predict", json=sample_u02001["features"])
    assert resp.status_code == 200
    pred = resp.get_json()
    assert pred["status"] == "success"
    p_data = pred["data"]
    assert "predicted_fmax_mhz" in p_data
    assert "pred_fmax_mhz" in p_data
    assert "window_low_mhz" in p_data
    assert "window_high_mhz" in p_data
    assert "sigma_calib_mhz" in p_data
    assert "window_half_mhz" in p_data
    assert p_data["window_low_mhz"] <= p_data["predicted_fmax_mhz"] <= p_data["window_high_mhz"]
    print(f"[PASS] /api/predict -> pred: {p_data['predicted_fmax_mhz']} MHz, window: [{p_data['window_low_mhz']}, {p_data['window_high_mhz']}] MHz")

    # 1.5 /api/search
    search_payload = {
        "pred_fmax_mhz": p_data["predicted_fmax_mhz"],
        "window_half_mhz": p_data["window_half_mhz"],
        "fmax_true_mhz": sample_u02001["fmax_true_mhz"],
        "unit_id": "U02001",
        "is_synthetic": True,
        "noise_sigma": 4.0,
        "random_seed": 42
    }
    resp = client.post("/api/search", json=search_payload)
    assert resp.status_code == 200
    search_res = resp.get_json()["data"]
    assert "conventional_search" in search_res
    assert "ai_windowed_search" in search_res
    assert "comparison" in search_res
    conv = search_res["conventional_search"]
    ai = search_res["ai_windowed_search"]
    print(f"[PASS] /api/search -> Conv tests: {conv['n_tests']}, AI tests: {ai['n_tests']}, Saved: {search_res['comparison']['tests_saved']}")

    # 1.6 /api/results
    resp = client.get("/api/results?page=1&limit=10")
    assert resp.status_code == 200
    res_data = resp.get_json()
    assert res_data["total_records"] == 4000
    assert len(res_data["records"]) == 10
    first = res_data["records"][0]
    assert "conv_tests" in first and "ai_tests" in first
    print(f"[PASS] /api/results -> 4000 total records, page 1 returned {len(res_data['records'])} rows")


def test_2_and_3_prediction_and_samples():
    print("\n--- Test 2 & 3: Sample Chip Loading & Prediction Validation ---")
    samples = results_service.get_sample_units()
    assert len(samples) == 3, f"Expected 3 sample units, got {len(samples)}"

    target_units = ["U02001", "U02024", "U02008"]
    loaded_ids = [s["unit_id"] for s in samples]
    assert loaded_ids == target_units, f"Expected {target_units}, got {loaded_ids}"

    for sample in samples:
        uid = sample["unit_id"]
        features = sample["features"]
        assert len(features) == 15, f"{uid} features count is {len(features)}, expected 15"
        
        # Verify feature keys match the 15 required features
        from src.data_loader import FEATURE_COLS
        for col in FEATURE_COLS:
            assert col in features, f"Missing feature {col} in sample {uid}"
            assert isinstance(features[col], (int, float)), f"Feature {col} is not numeric in {uid}"

        # Test predict with sample features
        resp = client.post("/api/predict", json=features)
        assert resp.status_code == 200, f"Predict failed for {uid} with {resp.status_code}"
        res = resp.get_json()["data"]

        # Confirm ML model returns prediction
        assert "predicted_fmax_mhz" in res
        pred_fmax = res["predicted_fmax_mhz"]
        assert 1600.0 <= pred_fmax <= 2400.0, f"Predicted Fmax {pred_fmax} out of grid bounds"

        # Confirm calibration uncertainty is displayed
        assert "sigma_calib_mhz" in res
        assert res["sigma_calib_mhz"] == 26.53
        assert "window_half_mhz" in res
        assert res["window_half_mhz"] == 39.80

        # Confirm lower and upper search window limits are displayed
        assert "window_low_mhz" in res
        assert "window_high_mhz" in res
        assert res["window_low_mhz"] < res["window_high_mhz"]
        assert res["window_low_mhz"] >= 1600 and res["window_high_mhz"] <= 2400

        print(f"[PASS] {uid} ({sample['expected_behavior']}): Pred={pred_fmax:.1f} MHz, Window=[{res['window_low_mhz']}, {res['window_high_mhz']}], Sigma={res['sigma_calib_mhz']} MHz")


def test_4_search_comparison_modes():
    print("\n--- Test 4: Search Comparison All 3 Modes ---")
    # Direct
    s1 = search_service.compare_searches(2010.5, 39.8, 2013.4, random_seed=42)
    assert s1["ai_windowed_search"]["search_type"] == "window_direct"
    assert len(s1["ai_windowed_search"]["steps"]) > 0
    print(f"[PASS] Normal Window Search: {s1['ai_windowed_search']['n_tests']} tests, final: {s1['ai_windowed_search']['measured_fmax_mhz']} MHz")

    # Upper Extension
    s2 = search_service.compare_searches(1957.9, 39.8, 2061.2, random_seed=42)
    assert s2["ai_windowed_search"]["search_type"] == "upper_extension"
    assert any("extension" in st.get("phase", "") for st in s2["ai_windowed_search"]["steps"])
    print(f"[PASS] Upper Extension Search: {s2['ai_windowed_search']['n_tests']} tests, final: {s2['ai_windowed_search']['measured_fmax_mhz']} MHz")

    # Lower Fallback
    s3 = search_service.compare_searches(2042.1, 39.8, 1944.8, random_seed=42)
    assert s3["ai_windowed_search"]["search_type"] == "lower_fallback"
    assert any("fallback" in st.get("phase", "") for st in s3["ai_windowed_search"]["steps"])
    print(f"[PASS] Lower Fallback Search: {s3['ai_windowed_search']['n_tests']} tests, final: {s3['ai_windowed_search']['measured_fmax_mhz']} MHz")


def test_5_analytics_figures():
    print("\n--- Test 5: Analytics Graphs ---")
    figures = [
        "fig1_actual_vs_predicted.png",
        "fig2_error_distribution.png",
        "fig3_feature_importance.png",
        "fig4_search_comparison.png",
        "fig5_step_reduction.png",
        "fig6_predicted_vs_measured.png"
    ]
    for fig in figures:
        resp = client.get(f"/api/graphs/{fig}")
        assert resp.status_code == 200, f"Graph {fig} returned {resp.status_code}"
        assert resp.mimetype == "image/png"
        assert len(resp.data) > 10000, f"Graph {fig} is empty or suspiciously small ({len(resp.data)} bytes)"
        print(f"[PASS] Figure {fig}: HTTP 200, {len(resp.data)} bytes")


def test_6_results_and_downloads():
    print("\n--- Test 6: Experiment Results & Downloads ---")
    # Filters
    r_all = client.get("/api/results?search_type=all").get_json()
    r_dir = client.get("/api/results?search_type=direct_window").get_json()
    r_upp = client.get("/api/results?search_type=upper_extension").get_json()
    r_low = client.get("/api/results?search_type=lower_fallback").get_json()

    assert r_all["total_records"] == 4000
    assert r_dir["total_records"] == 3603
    assert r_upp["total_records"] == 215
    assert r_low["total_records"] == 182
    print(f"[PASS] Filters: all=4000, direct=3603, upper=215, lower=182")

    # Downloads
    dls = ["search_comparison.csv", "evaluation_results.csv", "metrics.json"]
    for d in dls:
        resp = client.get(f"/api/download/{d}")
        assert resp.status_code == 200, f"Download {d} returned {resp.status_code}"
        assert len(resp.data) > 1000
        print(f"[PASS] Download {d}: HTTP 200, {len(resp.data)} bytes")


def test_7_error_handling():
    print("\n--- Test 7: Error Handling & Validation ---")
    
    # 7.1 Empty input
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 400
    print(f"[PASS] Empty input -> HTTP 400: {resp.get_json()['message']}")

    # 7.2 Missing features
    resp = client.post("/api/predict", json={"ro_01": 2000.0})
    assert resp.status_code == 400
    assert "Missing required features" in resp.get_json()["message"]
    print(f"[PASS] Missing features -> HTTP 400: {resp.get_json()['message'][:60]}...")

    # 7.3 Invalid non-numeric values
    bad_features = {f"ro_{i:02d}": 2000.0 for i in range(1, 11)}
    bad_features.update({
        "iddq_norm": "invalid_string",
        "vmin_norm": 1.0,
        "die_x": 0.0,
        "die_y": 0.0,
        "radius_sq": 0.0
    })
    resp = client.post("/api/predict", json=bad_features)
    assert resp.status_code == 400
    assert "Invalid non-numeric values" in resp.get_json()["message"]
    print(f"[PASS] Non-numeric input -> HTTP 400: {resp.get_json()['message']}")

    # 7.4 Invalid search request (missing target Fmax)
    resp = client.post("/api/search", json={"pred_fmax_mhz": 2000.0})
    assert resp.status_code == 400
    print(f"[PASS] Search missing target Fmax -> HTTP 400: {resp.get_json()['message'][:60]}...")

    # 7.5 Path traversal in downloads
    resp = client.get("/api/download/../../etc/passwd")
    assert resp.status_code == 404
    print("[PASS] Path traversal attack rejected with HTTP 404")


if __name__ == "__main__":
    test_1_flask_backend()
    test_2_and_3_prediction_and_samples()
    test_4_search_comparison_modes()
    test_5_analytics_figures()
    test_6_results_and_downloads()
    test_7_error_handling()
    print("\n==========================================")
    print("ALL 7 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==========================================")
