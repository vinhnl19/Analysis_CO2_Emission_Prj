from pydantic import BaseModel

class RangeYearOut(BaseModel):
    minYear: int
    maxYear: int