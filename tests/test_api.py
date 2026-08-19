"""Unit and integration tests for the PAL Passenger Market Segmentation FastAPI endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402
from api.model_loader import model_service  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def initialize_model():
    """Ensure model is loaded before tests run."""
    model_service.load_model()


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Test root endpoint returns 200 and system metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data
    assert "/health" in data["endpoints"].values()
    assert "/predict" in data["endpoints"].values()


def test_health_endpoint_returns_ok_and_model_version(client: TestClient):
    """Deliverable requirement: GET /health returns {'status': 'ok'} and model version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert len(data["model_version"]) > 0
    assert data["model_name"] == "PAL_Passenger_Segmenter"
    assert "model_source" in data
    assert "timestamp" in data


def test_predict_valid_inputs_returns_prediction_and_confidence(client: TestClient):
    """Deliverable requirement: POST /predict accepts JSON, returns prediction + confidence."""
    payload = {
        "PurchaseLeadTime": 15.0,
        "PAX_Count": 1,
        "AverageFare": 140.87,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in {0, 1, 2}
    assert isinstance(data["segment_name"], str)
    assert len(data["segment_name"]) > 0
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_version" in data
    assert data["features"]["PAX Count"] == 1
    assert data["features"]["AverageFare"] == 140.87


def test_predict_advance_group_booking(client: TestClient):
    """Test prediction for advance group booking."""
    payload = {
        "PurchaseLeadTime": 180.0,
        "PAX_Count": 4,
        "AverageFare": 45.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {0, 1, 2}
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_invalid_negative_fare_returns_422(client: TestClient):
    """Deliverable requirement: POST /predict returns HTTP 422 on invalid input (not 500)."""
    payload = {
        "PurchaseLeadTime": 10.0,
        "PAX_Count": 1,
        "AverageFare": -25.50,  # Invalid negative fare
    }
    response = client.post("/predict", json=payload)
    err = f"Expected HTTP 422 for negative fare, got {response.status_code}"
    assert response.status_code == 422, err


def test_predict_invalid_zero_pax_returns_422(client: TestClient):
    """PAX Count must be at least 1; 0 should return HTTP 422."""
    payload = {
        "PurchaseLeadTime": 10.0,
        "PAX_Count": 0,  # Invalid PAX count
        "AverageFare": 100.0,
    }
    response = client.post("/predict", json=payload)
    err = f"Expected HTTP 422 for 0 PAX count, got {response.status_code}"
    assert response.status_code == 422, err


def test_predict_invalid_negative_lead_time_returns_422(client: TestClient):
    """PurchaseLeadTime cannot be negative; negative value should return HTTP 422."""
    payload = {
        "PurchaseLeadTime": -5.0,  # Invalid negative lead time
        "PAX_Count": 1,
        "AverageFare": 100.0,
    }
    response = client.post("/predict", json=payload)
    err = f"Expected HTTP 422 for negative lead time, got {response.status_code}"
    assert response.status_code == 422, err


def test_predict_missing_required_field_returns_422(client: TestClient):
    """Missing required field must trigger Pydantic validation error (HTTP 422)."""
    payload = {
        "PurchaseLeadTime": 10.0,
        # Missing PAX_Count and AverageFare
    }
    response = client.post("/predict", json=payload)
    err = f"Expected HTTP 422 for missing fields, got {response.status_code}"
    assert response.status_code == 422, err


def test_predict_wrong_data_type_returns_422(client: TestClient):
    """Passing string to numeric field must return HTTP 422."""
    payload = {
        "PurchaseLeadTime": "not-a-number",
        "PAX_Count": 1,
        "AverageFare": 100.0,
    }
    response = client.post("/predict", json=payload)
    err = f"Expected HTTP 422 for string type, got {response.status_code}"
    assert response.status_code == 422, err


def test_predict_batch_returns_list_of_predictions(client: TestClient):
    """Test batch prediction endpoint with multiple passenger records."""
    payload = {
        "passengers": [
            {"PurchaseLeadTime": 10.0, "PAX_Count": 1, "AverageFare": 150.0},
            {"PurchaseLeadTime": 200.0, "PAX_Count": 5, "AverageFare": 40.0},
            {"PurchaseLeadTime": 30.0, "PAX_Count": 2, "AverageFare": 55.0},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_predictions"] == 3
    assert len(data["predictions"]) == 3
    for pred in data["predictions"]:
        assert pred["prediction"] in {0, 1, 2}
        assert 0.0 <= pred["confidence"] <= 1.0
