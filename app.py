"""
Flask Web Application for Bank Marketing Prediction
Deployed on AWS EC2 as a containerised service.
"""
import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model & feature info ────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model/lr_deployment_pipeline.joblib")
FEATURE_INFO_PATH = os.environ.get("FEATURE_INFO_PATH", "model/feature_info.json")

pipeline = joblib.load(MODEL_PATH)
with open(FEATURE_INFO_PATH, "r") as f:
    feature_info = json.load(f)

FEATURES = feature_info["features"]
CATEGORICAL_VALUES = feature_info["categorical_values"]

print(f"✅ Model loaded from {MODEL_PATH}")
print(f"   Features: {FEATURES}")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the prediction form."""
    return render_template("index.html", categorical_values=CATEGORICAL_VALUES)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, run prediction, return result."""
    try:
        # Parse form inputs
        data = {
            "age": int(request.form["age"]),
            "job": request.form["job"],
            "marital": request.form["marital"],
            "education": request.form["education"],
            "default": request.form["default"],
            "housing": request.form["housing"],
            "loan": request.form["loan"],
        }

        # Build DataFrame matching training schema
        input_df = pd.DataFrame([data])

        # Predict
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]

        result = {
            "prediction": "YES" if prediction == 1 else "NO",
            "confidence": f"{max(probabilities) * 100:.1f}%",
            "prob_no": f"{probabilities[0] * 100:.1f}%",
            "prob_yes": f"{probabilities[1] * 100:.1f}%",
        }

        return render_template(
            "result.html",
            result=result,
            input_data=data,
            categorical_values=CATEGORICAL_VALUES,
        )

    except Exception as e:
        return render_template("index.html", error=str(e), categorical_values=CATEGORICAL_VALUES)


@app.route("/health")
def health():
    """Health check endpoint for load balancers / monitoring."""
    return jsonify({"status": "healthy", "model": MODEL_PATH})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic access."""
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        return jsonify({
            "prediction": "YES" if prediction == 1 else "NO",
            "subscribed": bool(prediction),
            "probability": {"no": float(probabilities[0]), "yes": float(probabilities[1])},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
