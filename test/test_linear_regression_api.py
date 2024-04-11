import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
from fastapi.testclient import TestClient
import pytest
from src.linear_regression_app.linear_regression_api import app


@pytest.fixture(scope="module")
def test_client():
    # Si nécessaire, effectuez une configuration initiale ici
    client = TestClient(app)
    return client

def test_linear_regression_predict(test_client):
    # Simulez une requête POST vers votre endpoint /predict
    response = test_client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], list)

