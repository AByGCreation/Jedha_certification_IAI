import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path so src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.main import app

client = TestClient(app)

def test_predict_returns_200():
    """Test que l'endpoint /predict retourne 200"""
    response = client.post("/predict", json={
        "transaction_id": "tx_test_001",
        "amount": 150.0,
        "merchant_id": "MCH_123"
    })
    assert response.status_code == 200

def test_predict_returns_fraud_score():
    """Test que la réponse contient un fraud_score"""
    response = client.post("/predict", json={
        "transaction_id": "tx_test_002",
        "amount": 9999.0,
        "merchant_id": "MCH_SUSPECT"
    })
    data = response.json()
    assert "fraud_score" in data
    assert 0 <= data["fraud_score"] <= 1

def test_predict_returns_decision():
    """Test que la réponse contient une décision"""
    response = client.post("/predict", json={
        "transaction_id": "tx_test_003",
        "amount": 50.0,
        "merchant_id": "MCH_OK"
    })
    data = response.json()
    assert "decision" in data
    assert data["decision"] in ["APPROVE", "BLOCK", "REVIEW"]