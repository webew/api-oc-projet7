from fastapi import FastAPI, Query, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.model_service import predict_from_features

app = FastAPI(title="OC Projet 7 - API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    try:
        approved, proba_default = predict_from_features(payload.features, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictResponse(
        approved=approved,
        probability_default=proba_default,
        threshold=threshold,
        top_features=[],
    )
    


