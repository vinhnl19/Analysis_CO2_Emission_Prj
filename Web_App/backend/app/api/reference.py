from fastapi import APIRouter, HTTPException
from app.schemas.reference import RangeYearOut
from app.services.reference_service import get_range_year_from_data

router = APIRouter(prefix="/reference", tags=["Reference"])

@router.get("/rangeyear", response_model=RangeYearOut)
def get_range_year():
    try:
        result = get_range_year_from_data()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))