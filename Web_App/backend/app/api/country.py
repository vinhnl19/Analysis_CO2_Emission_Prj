from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.country import CountryOut
from app.services.country_service import get_country_list

router = APIRouter(prefix="/country", tags=["Country"])

@router.get("/getall", response_model=List[CountryOut])
def get_all_country():
    try:
        result = get_country_list()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))