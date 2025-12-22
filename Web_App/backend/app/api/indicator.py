from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.indicator import IndicatorDataOut, GetIndicatorFromCountryReq
from app.services.indicator_service import get_indicator_data

router = APIRouter(prefix="/indicator", tags=["Indicator"])

@router.post("/getfromcountry", response_model=List[IndicatorDataOut])
def get_indicator_from_country(req: GetIndicatorFromCountryReq):
    try:
        result = get_indicator_data(country_codes=req.country_code_list, fromYear=req.fromYear, toYear=req.toYear)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))