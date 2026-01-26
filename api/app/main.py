from fastapi import FastAPI, Query
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="OC Projet 7 - API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    # Dummy predictor (placeholder) : proba fixe, juste pour valider le pipeline API+tests+CI.
    proba_default = 0.42
    approved = proba_default < threshold

    return PredictResponse(
        approved=approved,
        probability_default=proba_default,
        threshold=threshold,
        top_features=[],
    )
