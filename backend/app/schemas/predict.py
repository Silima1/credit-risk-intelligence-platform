from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class PredictRequest(BaseModel):
    # para um pedido “por ID” (do test.csv)
    application_id: Optional[Any] = None
    # ou input direto como features
    features: Optional[Dict[str, Any]] = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    risk_probability: float
    prediction: int
    recommendation: str
    used_threshold: float