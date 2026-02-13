# backend/app/schemas/dashboard.py

from pydantic import BaseModel

class DashboardResponse(BaseModel):
    total_applications: int
    high_risk: int
    medium_risk: int
    low_risk: int
    avg_risk_score: float
    std_risk_score: float
    score_drift_detected: bool
    score_drift_severity: str
    score_drift_psi: float
    score_drift_ks: float