"""
web_app/app.py
==============
Flask REST API and Web Application Backend for AI-Optimized Fmax Search.

Provides clean endpoints to:
- Serve saved experiment metrics (/api/metrics)
- Generate Fmax predictions from chip pre-test features (/api/predict)
- Execute and compare conventional vs AI-windowed search (/api/search)
- Provide curated test units for demonstration (/api/sample-units)
- Fetch paginated benchmark results (/api/results)
- Serve figures (/api/graphs/<filename>) and downloadable CSVs (/api/download/<filename>)
"""

import sys
import io
from pathlib import Path
import pandas as pd

# Add project root to sys.path so original src modules are imported without change
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory, abort, render_template

from web_app.services.prediction_service import prediction_service
from web_app.services.search_service import search_service
from web_app.services.results_service import results_service

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Return saved experiment benchmark metrics from original outputs/metrics.json.
    """
    try:
        metrics = results_service.get_saved_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_fmax():
    """
    Accept chip pre-test measurements, validate features, and run original model inference.
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Invalid request content-type. Expected application/json."
        }), 400

    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "Empty request body. Please provide the 15 required chip features."
        }), 400

    try:
        result = prediction_service.predict_fmax(data)
        return jsonify({
            "status": "success",
            "data": result
        }), 200
    except ValueError as ve:
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/api/search", methods=["POST"])
def search_comparison():
    """
    Execute conventional binary search and AI-windowed binary search (Algorithm 1)
    and return step-by-step traces, test counts, and comparison stats.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Expected application/json."}), 400

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Empty request payload."}), 400

    try:
        if "pred_fmax_mhz" not in data or data["pred_fmax_mhz"] is None:
            return jsonify({
                "status": "error",
                "message": "Missing required 'pred_fmax_mhz'."
            }), 400

        try:
            pred_fmax_mhz = float(data["pred_fmax_mhz"])
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Predicted Fmax must be a valid number."
            }), 400

        if pred_fmax_mhz <= 0:
            return jsonify({
                "status": "error",
                "message": "Predicted Fmax must be a positive number."
            }), 400

        window_half_mhz = float(data.get("window_half_mhz", prediction_service.window_half))
        noise_sigma = float(data.get("noise_sigma", 4.0))
        random_seed = int(data.get("random_seed", 42))
        unit_id = data.get("unit_id")
        is_synthetic = bool(data.get("is_synthetic", False))
        features = data.get("features")
        simulate_ood_drift = bool(data.get("simulate_ood_drift", False))

        # Check for ground-truth or simulated target Fmax
        if "fmax_true_mhz" in data and data["fmax_true_mhz"] is not None:
            try:
                fmax_true_mhz = float(data["fmax_true_mhz"])
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": "Actual Fmax must be a valid number."
                }), 400
        elif "simulated_true_fmax_mhz" in data and data["simulated_true_fmax_mhz"] is not None:
            try:
                fmax_true_mhz = float(data["simulated_true_fmax_mhz"])
                is_synthetic = False
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": "Actual Fmax must be a valid number."
                }), 400
        else:
            return jsonify({
                "status": "error",
                "message": (
                    "Search simulation requires an evaluation target. "
                    "Provide 'fmax_true_mhz' (from dataset) or 'simulated_true_fmax_mhz' (user-specified test target)."
                )
            }), 400

        if fmax_true_mhz <= 0:
            return jsonify({
                "status": "error",
                "message": "Actual (Ground Truth) Fmax must be a positive number."
            }), 400

        comparison = search_service.compare_searches(
            pred_fmax_mhz=pred_fmax_mhz,
            window_half_mhz=window_half_mhz,
            fmax_true_mhz=fmax_true_mhz,
            noise_sigma=noise_sigma,
            random_seed=random_seed,
            is_synthetic_unit=is_synthetic,
            unit_id=unit_id,
            features=features,
            simulate_ood_drift=simulate_ood_drift
        )

        return jsonify({
            "status": "success",
            "data": comparison
        }), 200

    except (ValueError, TypeError) as ve:
        return jsonify({"status": "error", "message": f"Invalid numerical inputs: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Search comparison failed: {str(e)}"}), 500


@app.route("/api/sample-units", methods=["GET"])
def get_sample_units():
    """
    Return curated test units representing in-window, upper-extension, and lower-fallback cases.
    """
    try:
        samples = results_service.get_sample_units()
        return jsonify({
            "status": "success",
            "count": len(samples),
            "samples": samples
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/results", methods=["GET"])
def get_results_table():
    """
    Return paginated benchmark search comparison records from outputs/search_comparison.csv.
    """
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
        search_type = request.args.get("search_type", "all")

        # Clamp pagination bounds
        page = max(1, page)
        limit = max(1, min(100, limit))

        results = results_service.get_paginated_results(page=page, limit=limit, search_type=search_type)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/graphs/<filename>", methods=["GET"])
def get_graph(filename):
    """
    Securely serve existing original figures from outputs/figures/.
    """
    file_path = results_service.validate_figure_name(filename)
    if not file_path:
        abort(404, description=f"Figure '{filename}' not found or unauthorized.")
    return send_file(file_path, mimetype="image/png")


@app.route("/api/download/<filename>", methods=["GET"])
def download_output_file(filename):
    """
    Securely download original CSV / JSON output files from outputs/.
    """
    file_path = results_service.validate_download_name(filename)
    if not file_path:
        abort(404, description=f"File '{filename}' not found or unauthorized for download.")
    
    mimetype = "text/csv" if filename.endswith(".csv") else "application/json"
    return send_file(file_path, mimetype=mimetype, as_attachment=True, download_name=filename)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status and project metadata."""
    return jsonify({
        "status": "healthy",
        "service": "AI-Optimized Fmax Search Backend",
        "model_loaded": prediction_service.model is not None,
        "preprocessor_loaded": prediction_service.preprocessor is not None,
        "calib_loaded": prediction_service.calib_info is not None,
        "sigma_calib_mhz": prediction_service.sigma_calib
    }), 200


# ---------------------------------------------------------------------------
# UI Template Views
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def view_dashboard():
    """Render Main Dashboard."""
    return render_template("index.html", active_page="dashboard")


@app.route("/prediction", methods=["GET"])
def view_prediction():
    """Render Fmax Prediction Page."""
    return render_template("prediction.html", active_page="prediction")


@app.route("/search", methods=["GET"])
def view_search():
    """Render Search Comparison Page."""
    return render_template("search.html", active_page="search")


@app.route("/analytics", methods=["GET"])
def view_analytics():
    """Render Analytics & Graphs Page."""
    return render_template("analytics.html", active_page="analytics")


@app.route("/results", methods=["GET"])
def view_results():
    """Render Experiment Results Page."""
    return render_template("results.html", active_page="results")


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """
    Accept a CSV file upload (up to 1,000 records), validate 15 features and chip_id,
    and return predictions with search window boundaries.
    """
    import io
    try:
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if not uploaded_file or uploaded_file.filename == "":
                return jsonify({"status": "error", "message": "No file selected."}), 400

            if not uploaded_file.filename.lower().endswith(".csv"):
                return jsonify({
                    "status": "error",
                    "message": "Invalid file format. Please upload a valid CSV file."
                }), 400

            try:
                content = uploaded_file.read().decode("utf-8", errors="replace")
                if not content.strip():
                    return jsonify({"status": "error", "message": "Uploaded CSV is empty."}), 400
                df = pd.read_csv(io.StringIO(content))
            except Exception as read_err:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to parse CSV file: {str(read_err)}"
                }), 400

        elif request.is_json:
            data = request.get_json()
            if not data or "records" not in data:
                return jsonify({
                    "status": "error",
                    "message": "No CSV file uploaded or 'records' payload provided."
                }), 400
            df = pd.DataFrame(data["records"])
        else:
            return jsonify({
                "status": "error",
                "message": "No CSV file provided. Please upload a CSV file with up to 1,000 records."
            }), 400

        # Run batch prediction through prediction_service
        result_df, summary = prediction_service.predict_batch(df)

        records = result_df.to_dict(orient="records")
        return jsonify({
            "status": "success",
            "count": len(records),
            "summary": summary,
            "data": records
        }), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Batch prediction failed: {str(e)}"
        }), 500


@app.route("/api/sample-batch-csv", methods=["GET"])
def get_sample_batch_csv():
    """
    Serve a sample CSV file (10 records) for users to test batch prediction.
    """
    from io import BytesIO
    import pandas as pd
    try:
        dataset_path = PROJECT_ROOT / "synthetic_fmax_dataset.csv"
        if dataset_path.exists():
            raw_df = pd.read_csv(dataset_path)
            from src.data_loader import FEATURE_COLS
            cols = ["unit_id"] + [c for c in FEATURE_COLS if c in raw_df.columns]
            sample_df = raw_df[cols].head(10).rename(columns={"unit_id": "chip_id"})
            csv_buf = sample_df.to_csv(index=False)
            mem = BytesIO(csv_buf.encode("utf-8"))
            return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="sample_batch_chips.csv")
        else:
            abort(404, description="Dataset file not found.")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/batch-prediction", methods=["GET"])
def view_batch_prediction():
    """Render Batch Prediction Page."""
    return render_template("batch_prediction.html", active_page="batch_prediction")


if __name__ == "__main__":
    print(f"Starting AI-Optimized Fmax Search Flask Backend on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)

