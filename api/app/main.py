from fastapi import FastAPI, HTTPException
from api.app.schemas import PredictRequest, PredictResponse
from api.app.model_service import predict_from_id

app = FastAPI(title="OC Projet 7 - API", version="0.1.0")

# health check de l'api
@app.get("/health")
def health():
    return {"status": "ok"}

# prédiction à partir de l'identifiant d'un client
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        approved, proba_default, threshold = predict_from_id(payload.sk_id_curr)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictResponse(
        approved=approved,
        probability=proba_default,
        threshold=threshold
    )