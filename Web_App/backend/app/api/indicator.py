from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.indicator import IndicatorDataOut, GetIndicatorFromCountryReq, MetricShareItem
from app.services.indicator_service import get_indicator_data, get_gdp_allocation, get_distribution_energy, get_distribution_land_area

router = APIRouter(prefix="/indicator", tags=["Indicator"])

@router.post("/getfromcountry", response_model=List[IndicatorDataOut])
def get_indicator_from_country(req: GetIndicatorFromCountryReq):
    try:
        result = get_indicator_data(country_codes=req.country_code_list, fromYear=req.fromYear, toYear=req.toYear)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/get-gdp-allocation", response_model=List[MetricShareItem])
def get_gdp_allocation_api(req: GetIndicatorFromCountryReq):
    try:
        result = get_gdp_allocation(country_codes=req.country_code_list, fromYear=req.fromYear, toYear=req.toYear)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
@router.post("/get-distribution-energy", response_model=List[MetricShareItem])
def get_distribution_energy_api(req: GetIndicatorFromCountryReq):
    try:
        result = get_distribution_energy(country_codes=req.country_code_list, fromYear=req.fromYear, toYear=req.toYear)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
@router.post("/get-distribution-land-area", response_model=List[MetricShareItem])
def get_distribution_land_area_api(req: GetIndicatorFromCountryReq):
    try:
        result = get_distribution_land_area(country_codes=req.country_code_list, fromYear=req.fromYear, toYear=req.toYear)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))