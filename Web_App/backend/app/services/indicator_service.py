from app.core.resources import df
from typing import List

def get_indicator_data(country_codes: List[str], fromYear: int, toYear: int):
    indicators_meta = [
        {
            "name": "Co2_MtCO2",
            "unit": "MtCO2",
            "description": "Total CO2 Emissions",
            "iconMapping": "co2global"
        },
        {
            "name": "Co2_Capita_tCO2",
            "unit": "tCO2",
            "description": "CO2 per Capita",
            "iconMapping": "co2capita"
        },
        {
            "name": "Population",
            "unit": "dân",
            "description": "Population",
            "iconMapping": "population"
        },
        {
            "name": "GDP",
            "unit": "$",
            "description": "GDP",
            "iconMapping": "gdp"
        },
        {
            "name": "Government_Expenditure_on_Education",
            "unit": "$",
            "description": "Government Expenditure on Education",
            "iconMapping": "education"
        },
        {
            "name": "Energy_MWh",
            "unit": "MWh",
            "description": "Total Energy",
            "iconMapping": "energy"
        },
        {
            "name": "Global_Climate_Risk_Index",
            "unit": "",
            "description": "CRI",
            "iconMapping": "cri"
        },
        {
            "name": "Area_ha",
            "unit": "ha",
            "description": "Area",
            "iconMapping": "area"
        },
        {
            "name": "HDI",
            "unit": "",
            "description": "HDI",
            "iconMapping": "hdi"
        },
    ]
    df_filtered = df[(df['ISO_Code'].isin(country_codes)) & (df['Year'] >= fromYear) & (df['Year'] <= toYear)]
    result = []
    for indicator in indicators_meta:
        if (indicator["name"] in df_filtered.columns):
            value = df_filtered[indicator["name"]].sum()
        else:
            value = None
        
        result.append({
            "value": value,
            "unit": indicator["unit"],
            "description": indicator["description"],
            "iconMapping": indicator["iconMapping"]
        })
    return result