from app.core.resources import df
from typing import List

def get_indicator_data(country_codes: List[str], fromYear: int, toYear: int):
    indicators_meta = [
        {
            "name": "Co2_MtCO2",
            "unit": "MtCO2",
            "description": "Total CO2 Emissions"
        },
        {
            "name": "Co2_Capita_tCO2",
            "unit": "tCO2",
            "description": "CO2 per Capita",
        },
        {
            "name": "Population",
            "unit": "Tỷ dân",
            "description": "Population"
        },
        {
            "name": "GDP",
            "unit": "$",
            "description": "GDP"
        },
        {
            "name": "Government_Expenditure_on_Education",
            "unit": "$",
            "description": "Government Expenditure on Education"
        },
        {
            "name": "Energy_MWh",
            "unit": "MWh",
            "description": "Total Energy"
        },
        {
            "name": "Global_Climate_Risk_Index",
            "unit": "",
            "description": "CRI"
        },
        {
            "name": "Area_ha",
            "unit": "ha",
            "description": "Area"
        },
        {
            "name": "HDI",
            "unit": "",
            "description": "HDI"
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
            "description": indicator["description"]
        })
    return result