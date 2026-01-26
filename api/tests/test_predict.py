from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_schema_and_values():
    payload = {"features": {"DAYS_BIRTH": -12000, "EXT_SOURCE_2": 0.2}}
    r = client.post("/v1/predict?threshold=0.5", json=payload)

    assert r.status_code == 200
    data = r.json()

    assert "approved" in data
    assert "probability_default" in data
    assert "threshold" in data

    assert 0.0 <= data["probability_default"] <= 1.0
    assert data["threshold"] == 0.5
