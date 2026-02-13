from pydantic import BaseModel
from typing import List

class LocalExplanationItem(BaseModel):
    feature: str
    display_feature: str
    shap_value: float
    abs_shap: float
    impact_text: str

class LocalExplanationResponse(BaseModel):
    items: List[LocalExplanationItem]
    reason_codes: List[str]

class GlobalImportanceItem(BaseModel):
    feature: str
    display_feature: str
    mean_abs_shap: float
    std_shap: float

class GlobalImportanceResponse(BaseModel):
    items: List[GlobalImportanceItem]