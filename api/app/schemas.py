from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    approved: bool
    probability_default: float
    threshold: float
    top_features: Optional[List[dict]] = None
