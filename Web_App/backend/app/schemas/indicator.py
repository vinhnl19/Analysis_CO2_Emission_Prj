from typing import List, Any, Optional
from pydantic import BaseModel

class GetIndicatorFromCountryReq(BaseModel):
    country_code_list: List[str]
    fromYear: int
    toYear: int
class IndicatorDataOut(BaseModel):
    value: Any
    unit: str
    description: str
    iconMapping: str
    calculationNote: str
class MetricShareItem(BaseModel):
    key: str
    label: str
    value: Optional[float]
    unit: str
