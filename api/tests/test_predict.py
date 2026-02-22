from fastapi.testclient import TestClient
from api.app.main import app
import pandas as pd
from pathlib import Path
import joblib
import json
from api.app.model_service import predict_from_features, get_model

client = TestClient(app)

API_DIR = Path(__file__).resolve().parents[1]   # -> api
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"  # -> data/raw
MODEL_PATH = API_DIR / "models" / "model.pkl"

def test_predict_schema_and_values():
    payload = {"features": {"DAYS_BIRTH": -12000, "EXT_SOURCE_2": 0.2}}
    r = client.post("/predict", json=payload)
    print(r.status_code, r.text)

    assert r.status_code == 200
    data = r.json()

    assert "approved" in data
    assert "probability" in data
    assert "threshold" in data

    assert 0.0 <= data["probability"] <= 1.0
    print("Data:", data)
    if data["approved"]:
        assert data["probability"] < data["threshold"]
    else:
        assert data["probability"] >= data["threshold"]

# tester la réponse de l'api pour une ligne de application_train.csv
def test_predict_from_application_train():
    # lecture du seuil
    with open(API_DIR / "models" / "export_data.json") as f:
        data = json.load(f)
    threshold = float(data["threshold"])

    # première ligne de application.csv
    df = pd.read_csv(DATA_DIR / "application_train.csv", nrows=1)
    row = df.iloc[0]
    features = row.where(pd.notnull(row), None).to_dict()

    # appel de la fonction predict_from_features
    approved, proba_default, returned_threshold = predict_from_features(features)

    # tests sur threshold et approved
    assert returned_threshold == threshold
    assert approved == (proba_default < threshold)
    assert 0.0 <= proba_default <= 1.0
    