from pydantic import BaseModel
from typing import List, Dict, Any, Literal


class ReportDriver(BaseModel):
    title: str
    direction: Literal["increases", "decreases"]
    strength: Literal["small", "medium", "large"]
    shap_value: float
    narrative: str


class ReportResponse(BaseModel):
    application_id: str
    risk_probability: float
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    threshold: float

    avg_population_score: float
    std_population_score: float
    percentile_vs_population: float

    narrative: str
    key_drivers: List[ReportDriver]
    what_to_improve: List[str]

    charts: Dict[str, Any]