"""
Basic tests for the FastAPI inference service.
Run with:  uv run pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded (simulates startup failure)."""
    # Import here so the lifespan does not try to connect to MLflow
    with patch("src.api.model", None):
        from src.api import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture()
def client_with_model():
    """TestClient with a mocked model that returns a fixed prediction."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.42]])

    with patch("src.api.model", mock_model):
        from src.api import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
def test_health_no_model(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_with_model(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# Predict endpoint — happy path
# ---------------------------------------------------------------------------
def test_predict_returns_prediction(client_with_model):
    payload = {
        "sequence": [[float(i) / 10 for i in range(8)]] * 24  # 24 timesteps × 8 features
    }
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], list)
    assert len(body["prediction"]) >= 1
    assert "inference_time_ms" in body


# ---------------------------------------------------------------------------
# Predict endpoint — error cases
# ---------------------------------------------------------------------------
def test_predict_no_model(client_no_model):
    payload = {"sequence": [[0.1] * 8] * 24}
    response = client_no_model.post("/predict", json=payload)
    assert response.status_code == 503


def test_predict_invalid_payload(client_with_model):
    # Missing 'sequence' key entirely
    response = client_with_model.post("/predict", json={"data": []})
    assert response.status_code == 422


def test_predict_wrong_shape(client_with_model):
    # 1-D list instead of 2-D
    response = client_with_model.post("/predict", json={"sequence": [0.1, 0.2, 0.3]})
    assert response.status_code == 422
