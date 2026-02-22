from fastapi.testclient import TestClient
from api.app.main import app
import pandas as pd
from pathlib import Path
import json
from api.app.model_service import predict_from_id

client = TestClient(app)

API_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def test_predict_schema_and_values():
    df = pd.read_csv(DATA_DIR / "application_test.csv")
    sk_id = int(df["SK_ID_CURR"].iloc[0])

    r = client.post("/predict", json={"sk_id_curr": sk_id})
    print(r.status_code, r.text)

    assert r.status_code == 200
    data = r.json()

    assert "approved" in data
    assert "probability" in data
    assert "threshold" in data

    assert 0.0 <= data["probability"] <= 1.0
    if data["approved"]:
        assert data["probability"] < data["threshold"]
    else:
        assert data["probability"] >= data["threshold"]

def test_predict_from_application_test():
    with open(API_DIR / "models" / "export_data.json") as f:
        threshold = float(json.load(f)["threshold"])

    df = pd.read_csv(DATA_DIR / "application_test.csv")
    sk_id = int(df["SK_ID_CURR"].iloc[0])

    approved, proba_default, returned_threshold = predict_from_id(sk_id)

    assert returned_threshold == threshold
    assert approved == (proba_default < threshold)
    assert 0.0 <= proba_default <= 1.0