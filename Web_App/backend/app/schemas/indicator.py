from typing import List, Any
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