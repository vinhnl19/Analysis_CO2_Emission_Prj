from fastapi import APIRouter
from .country import router as country_router
from .indicator import router as indicator_router

api_router = APIRouter()

api_router.include_router(country_router)
api_router.include_router(indicator_router)