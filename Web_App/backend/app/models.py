# backend/app/models.py
from pydantic import BaseModel
from typing import List, Dict, Any

class SeqPredictReq(BaseModel):
    country: str
    sequence_features: List[List[float]]  # 3 rows x 10 features

class ManualPredictReq(BaseModel):
    country: str
    features: Dict[str, float]  # FEATURE_CORE

class RecommendReq(BaseModel):
    country: str
    feature_selection: List[Dict[str, Any]]  # [{"feature", "min_pct", "max_pct"}]
    fixed_features: Dict[str, float]         # {"Population": ..., "GDP": ...}
    co2_target: float
