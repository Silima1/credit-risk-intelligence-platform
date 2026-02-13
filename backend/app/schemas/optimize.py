from pydantic import BaseModel, Field
from typing import Dict

class CostMatrix(BaseModel):
    FP: float = 100
    FN: float = 500
    TP: float = -200
    TN: float = -50

class OptimizeThresholdRequest(BaseModel):
    cost_matrix: CostMatrix

class OptimizeThresholdResponse(BaseModel):
    optimal_threshold: float = Field(ge=0.0, le=1.0)
    expected_cost: float