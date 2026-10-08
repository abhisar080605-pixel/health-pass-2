"""
HealthPass - IoT Kiosk Diabetes Risk Screening REST API

Flask backend for ESP32 HealthPass kiosk:
- POST /predict: Accepts sensor readings, evaluates risk score, logs to SQLite with RFID tag.
- POST /retrain: Re-runs GA feature selection & model retraining on updated dataset.
- GET /features: Returns GA-selected features, pruned features, and global importances.
- GET /convergence: Returns GA convergence history and chart metadata.
- GET /screenings/<rfid_tag>: Retrieves historical screenings for a given RFID/NFC tag.
- GET /health: System health and active screening model metadata.

CLINICAL FRAMING NOTICE:
HealthPass is an opportunistic RISK SCREENING tool using physiological proxies
(BMI, blood pressure, heart rate, physical activity, GSR stress, age).
There is NO blood glucose sensor. It CANNOT diagnose diabetes.
"""

import os
import sqlite3
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_file, g, render_template
from model import RiskModelPipeline, CLINICAL_DISCLAIMER, DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH

if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/healthpass.db"
else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthpass.db")

app = Flask(__name__)
pipeline = RiskModelPipeline(data_path=DEFAULT_DATA_PATH)

def get_db():
    """Establish or retrieve per-request SQLite connection."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    """Close the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database tables for kiosk logs and retraining history."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rfid_tag TEXT,
                readings_json TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_probability REAL NOT NULL,
                risk_label TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                top_features_json TEXT NOT NULL,
                disclaimer TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrain_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                baseline_accuracy REAL,
                ga_accuracy REAL,
                baseline_f1 REAL,
                ga_f1 REAL,
                feature_reduction_pct REAL,
                selected_features_json TEXT,
                pruned_features_json TEXT
            );
        """)
        conn.commit()
    finally:
        conn.close()

# Ensure database and model are initialized on load
init_db()
if not pipeline.load(DEFAULT_MODEL_PATH):
    print("[HealthPass API] No saved model found. Training initial pipeline...")
    pipeline.train(n_population=25, n_generations=20, random_state=42)
    pipeline.save(DEFAULT_MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    """Serve the interactive HealthPass Kiosk Web Dashboard with multi-path fallback."""
    # 1. Try standard render_template
    try:
        return render_template("index.html")
    except Exception:
        pass

    # 2. Check all common candidate paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "templates", "index.html"),
        os.path.join(base_dir, "index.html"),
        os.path.join(os.getcwd(), "templates", "index.html"),
        os.path.join(os.getcwd(), "index.html")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
            except Exception:
                continue

    return "<h2>HealthPass Dashboard</h2><p>Template file is being provisioned. Please check back shortly.</p>", 200, {"Content-Type": "text/html"}


@app.route("/health", methods=["GET"])
def health():
    """Health check and model status endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "HealthPass Risk Screening Engine",
        "kiosk_hardware": {
            "sensors": [
                "MAX30102 (heart_rate, spo2)",
                "MLX90614 (body_temp)",
                "HX711 (weight -> bmi)",
                "GSR (stress proxy)",
                "MPU6050 (activity proxy)",
                "RFID/NFC (patient_id)"
            ],
            "diagnostic_capability": "NONE (Physiological Proxy Risk Screening Only - No Glucose Sensor)"
        },
        "model_trained": pipeline.is_trained,
        "selected_features_count": len(pipeline.selected_features),
        "selected_features": pipeline.selected_features,
        "trained_at": pipeline.trained_at,
        "disclaimer": CLINICAL_DISCLAIMER
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON body:
    {
        "rfid_tag": "A1B2C3D4",    # Optional patient RFID identifier
        "readings": {
            "age": 48,
            "bmi": 29.4,
            "heart_rate": 78,
            "systolic_bp": 134,
            "activity_level": 2,
            "gsr_stress": 5.2,
            "spo2": 97.0,
            "body_temp": 36.6
        }
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request. JSON body expected."}), 400

    # Support either top-level sensor fields or nested inside "readings"
    rfid_tag = data.get("rfid_tag", "ANONYMOUS")
    readings = data.get("readings", data)
    if not isinstance(readings, dict):
        return jsonify({"error": "Readings must be a key-value dictionary of sensor values."}), 400

    try:
        prediction = pipeline.predict_risk(readings)
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # Log to SQLite database
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO screenings (
                    timestamp, rfid_tag, readings_json, risk_score,
                    risk_probability, risk_label, recommendation,
                    top_features_json, disclaimer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction["timestamp"],
                rfid_tag,
                json.dumps(readings),
                prediction["risk_score"],
                prediction["risk_probability"],
                prediction["risk_label"],
                prediction["clinical_recommendation"],
                json.dumps(prediction["top_contributing_features"]),
                prediction["disclaimer"]
            ))
            conn.commit()
    except Exception as db_err:
        print(f"[Warning] Failed to log screening to database: {db_err}")

    response = {
        "rfid_tag": rfid_tag,
        "risk_score": prediction["risk_score"],
        "risk_probability": prediction["risk_probability"],
        "risk_label": prediction["risk_label"],
        "clinical_recommendation": prediction["clinical_recommendation"],
        "top_contributing_features": prediction["top_contributing_features"],
        "features_used": prediction["features_used_in_prediction"],
        "features_imputed": prediction["features_imputed"],
        "explainability_method": prediction["explainability_method"],
        "screening_disclaimer": prediction["disclaimer"],
        "timestamp": prediction["timestamp"]
    }
    return jsonify(response), 200


@app.route("/features", methods=["GET"])
def get_features():
    """
    GET /features
    Returns the currently selected feature subset, pruned features, and
    global feature importances from the GA-optimized model.
    """
    if not pipeline.is_trained or pipeline.explainer is None:
        return jsonify({"error": "Model not trained yet"}), 500

    global_importances = pipeline.explainer.get_global_importances()
    metrics = pipeline.metrics

    return jsonify({
        "selected_features": pipeline.selected_features,
        "selected_count": len(pipeline.selected_features),
        "pruned_features": pipeline.pruned_features,
        "pruned_count": len(pipeline.pruned_features),
        "feature_importances": global_importances,
        "baseline_comparison": metrics.get("comparison", {}),
        "evaluation_metrics": {
            "baseline": metrics.get("baseline", {}),
            "ga_selected": metrics.get("ga_selected", {})
        },
        "disclaimer": CLINICAL_DISCLAIMER
    }), 200


@app.route("/convergence", methods=["GET"])
def get_convergence():
    """
    GET /convergence
    Returns GA convergence history per generation, plus chart file availability.
    """
    history = pipeline.selector.history_ if pipeline.selector else []
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "ga_convergence.png")
    has_chart = os.path.exists(chart_path)

    return jsonify({
        "total_generations_run": len(history),
        "convergence_history": history,
        "chart_available": has_chart,
        "chart_url": "/convergence/chart" if has_chart else None,
        "fitness_formula": "fitness = 0.8 * val_accuracy + 0.2 * (1 - selected/total)"
    }), 200


@app.route("/convergence/chart", methods=["GET"])
def get_convergence_chart():
    """Return the generated convergence image file, or generate it dynamically on-the-fly."""
    import io
    import base64
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "static", "ga_convergence.png"),
        os.path.join(base_dir, "ga_convergence.png"),
        "/tmp/ga_convergence.png"
    ]
    for p in candidates:
        if os.path.exists(p):
            return send_file(p, mimetype="image/png")

    # Generate dynamically from model selector history if available
    if pipeline.selector and pipeline.selector.history_:
        try:
            b64_str = pipeline.selector.plot_convergence()
            if b64_str:
                img_data = base64.b64decode(b64_str)
                return send_file(io.BytesIO(img_data), mimetype="image/png")
        except Exception as e:
            print(f"[Chart Generator] Dynamic chart generation error: {e}")

    return jsonify({"error": "Chart not generated yet"}), 404


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    POST /retrain
    Re-runs GA feature selection and trains a new model.
    Optional JSON payload:
    {
        "dataset_path": "data/synthetic_kiosk_data.csv",
        "n_population": 30,
        "n_generations": 25
    }
    """
    body = request.get_json(silent=True) or {}
    dataset_path = body.get("dataset_path", DEFAULT_DATA_PATH)
    n_population = int(body.get("n_population", 30))
    n_generations = int(body.get("n_generations", 25))

    try:
        metrics = pipeline.train(
            dataset_path=dataset_path,
            n_population=n_population,
            n_generations=n_generations
        )
        pipeline.save(DEFAULT_MODEL_PATH)

        # Log retrain event in DB
        with get_db() as conn:
            conn.execute("""
                INSERT INTO retrain_history (
                    timestamp, baseline_accuracy, ga_accuracy,
                    baseline_f1, ga_f1, feature_reduction_pct,
                    selected_features_json, pruned_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                metrics["baseline"]["accuracy"],
                metrics["ga_selected"]["accuracy"],
                metrics["baseline"]["f1_score"],
                metrics["ga_selected"]["f1_score"],
                metrics["comparison"]["feature_reduction_percent"],
                json.dumps(metrics["ga_selected"]["features"]),
                json.dumps(metrics["ga_selected"]["pruned_features"])
            ))
            conn.commit()

        return jsonify({
            "message": "Adaptive GA feature selection and model retraining successful.",
            "metrics": metrics,
            "trained_at": pipeline.trained_at,
            "disclaimer": CLINICAL_DISCLAIMER
        }), 200
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500


@app.route("/screenings/<rfid_tag>", methods=["GET"])
def get_patient_screenings(rfid_tag: str):
    """
    GET /screenings/<rfid_tag>
    Returns all logged kiosk screening visits for a specific RFID/NFC patient badge.
    """
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM screenings WHERE rfid_tag = ? ORDER BY id DESC LIMIT 50",
            (rfid_tag,)
        )
        rows = cursor.fetchall()

    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "rfid_tag": r["rfid_tag"],
            "readings": json.loads(r["readings_json"]),
            "risk_score": r["risk_score"],
            "risk_probability": r["risk_probability"],
            "risk_label": r["risk_label"],
            "recommendation": r["recommendation"],
            "top_contributing_features": json.loads(r["top_features_json"]),
            "disclaimer": r["disclaimer"]
        })

    return jsonify({
        "rfid_tag": rfid_tag,
        "total_screenings": len(records),
        "history": records
    }), 200


@app.route("/screenings", methods=["GET"])
def get_all_screenings():
    """
    GET /screenings?limit=20
    Returns the most recent kiosk screenings across all patients.
    """
    limit = min(int(request.args.get("limit", 20)), 100)
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM screenings ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()

    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "rfid_tag": r["rfid_tag"],
            "risk_score": r["risk_score"],
            "risk_label": r["risk_label"],
            "recommendation": r["recommendation"]
        })

    return jsonify({
        "limit": limit,
        "count": len(records),
        "screenings": records
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[HealthPass API] Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
