from pydantic import BaseModel
from typing import Dict

class ScoreDriftResponse(BaseModel):
    psi: float
    ks: float
    drift_detected: bool
    severity: str

class FeatureDriftItem(BaseModel):
    psi: float
    ks: float
    drift_detected: bool
    severity: str

class DataDriftResponse(BaseModel):
    results: Dict[str, FeatureDriftItem]