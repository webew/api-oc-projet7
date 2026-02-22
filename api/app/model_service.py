import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple
import json

API_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = API_DIR / "models" / "model.pkl"
FEATURES_STORE_PATH = API_DIR / "models" / "features_store.parquet"

with open(API_DIR / "models" / "export_data.json", "r") as f:
    THRESHOLD = float(json.load(f)["threshold"])

_model = None
_features_store = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def get_features_store():
    global _features_store
    if _features_store is None:
        _features_store = pd.read_parquet(FEATURES_STORE_PATH)
    return _features_store

def predict_from_id(sk_id_curr: int) -> Tuple[bool, float, float]:
    model = get_model()
    store = get_features_store()

    row = store[store["SK_ID_CURR"] == sk_id_curr]
    if row.empty:
        raise ValueError(f"SK_ID_CURR {sk_id_curr} introuvable dans le features store")

    cols = list(model.feature_names_in_)
    X = row[cols]

    proba_default = float(model.predict_proba(X)[0][1])
    approved = proba_default < THRESHOLD
    return approved, proba_default, THRESHOLD