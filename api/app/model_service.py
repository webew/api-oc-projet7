import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

# Dossier api/ (stable, même si on lance pytest depuis ailleurs)
API_DIR = Path(__file__).resolve().parents[1]  # .../api
DEFAULT_MODEL_PATH = API_DIR / "models" / "model.pkl"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_from_features(features: Dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    model = get_model()

    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        X = pd.DataFrame([{c: features.get(c) for c in cols}], columns=cols)
    else:
        X = pd.DataFrame([features])

    if hasattr(model, "predict_proba"):
        proba_default = float(model.predict_proba(X)[0][1])
    else:
        pred = model.predict(X)[0]
        proba_default = float(pred)

    approved = proba_default < float(threshold)
    return approved, proba_default
