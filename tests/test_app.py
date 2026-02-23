"""
Unit tests for the Flask prediction application.
"""
import json
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_feature_info_structure():
    """Test that feature_info.json has the correct structure."""
    info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "feature_info.json")
    if not os.path.exists(info_path):
        pytest.skip("Model not yet trained — run notebook first")

    with open(info_path) as f:
        info = json.load(f)

    assert "features" in info
    assert "categorical_values" in info

    expected_features = ["age", "job", "marital", "education", "default", "housing", "loan"]
    assert info["features"] == expected_features

    for cat in ["job", "marital", "education", "default", "housing", "loan"]:
        assert cat in info["categorical_values"]
        assert len(info["categorical_values"][cat]) > 0


def test_model_loads():
    """Test that the saved model pipeline loads and predicts."""
    import joblib
    import pandas as pd

    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "lr_deployment_pipeline.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained — run notebook first")

    pipeline = joblib.load(model_path)

    sample = pd.DataFrame([{
        "age": 30,
        "job": "admin.",
        "marital": "single",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
    }])

    prediction = pipeline.predict(sample)
    assert prediction[0] in [0, 1]

    proba = pipeline.predict_proba(sample)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_flask_app_health():
    """Test the /health endpoint."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "lr_deployment_pipeline.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained — run notebook first")

    from app import app
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


def test_flask_app_index():
    """Test that the index page loads."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "lr_deployment_pipeline.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained — run notebook first")

    from app import app
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Bank Marketing Prediction" in response.data


def test_flask_app_predict():
    """Test the /predict endpoint with valid form data."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "lr_deployment_pipeline.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained — run notebook first")

    from app import app
    client = app.test_client()
    response = client.post("/predict", data={
        "age": "35",
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
    })
    assert response.status_code == 200
    assert b"YES" in response.data or b"NO" in response.data


def test_api_predict():
    """Test the /api/predict REST endpoint."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "lr_deployment_pipeline.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained — run notebook first")

    from app import app
    client = app.test_client()
    response = client.post("/api/predict",
        data=json.dumps({
            "age": 35,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
        }),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in ["YES", "NO"]
    assert "probability" in data
