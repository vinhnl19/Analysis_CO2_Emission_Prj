from typing import List
from pydantic import BaseModel
class CountryOut(BaseModel):
    country_code: str
    country_name: str
    continent: str

    class Config:
        from_attributes = True


