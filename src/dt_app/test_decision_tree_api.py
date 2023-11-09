import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from dt_app.decision_tree_api import app
import joblib


# define the client
client = TestClient(app)

@pytest.fixture(scope="module")
def test_client():
    # Si nécessaire, effectuez une configuration initiale ici
    return client

def test_decision_tree_predict(test_client):
    # Simulez une requête POST vers votre endpoint /predict
    response = test_client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))  # Assurez-vous que la prédiction est retournée en liste
