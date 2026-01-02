from app.core.resources import df
from typing import List
import pandas as pd
from app.utils.format_number import format_number

def get_indicator_data(country_codes: List[str], fromYear: int, toYear: int):
    indicators_meta = [
        # Group A
        {
            "name": "Co2_MtCO2",
            "unit": "MtCO2",
            "description": "Total CO2 Emissions",
            "iconMapping": "co2global",
            "agg_type": "sum"
        },
        {
            "name": "Co2_Capita_tCO2",
            "unit": "tCO2",
            "description": "CO2 per Capita",
            "iconMapping": "co2capita",
            "agg_type": "weighted_latest"
        },
        {
            "name": "Population",
            "unit": "people",
            "description": "Population",
            "iconMapping": "population",
            "agg_type": "latest_sum"
        },
        {
            "name": "GDP",
            "unit": "$",
            "description": "GDP",
            "iconMapping": "gdp",
            "agg_type": "latest_sum"
        },
        {
            "name": "Government_Expenditure_on_Education",
            "unit": '% GDP',
            "description": "Government Expenditure on Education",
            "iconMapping": "education",
            "agg_type": "mean_year_country"
        },
        {
            "name": "Energy_MWh",
            "unit": "MWh",
            "description": "Total Energy",
            "iconMapping": "energy",
            "agg_type": "sum"
        },
        {
            "name": "Global_Climate_Risk_Index",
            "unit": "",
            "description": "CRI",
            "iconMapping": "cri",
            "agg_type": "mean_year_country"
        },
        {
            "name": "Area_ha",
            "unit": "ha",
            "description": "Area",
            "iconMapping": "area",
            "agg_type": "latest_sum"
        },
        {
            "name": "HDI",
            "unit": "",
            "description": "HDI",
            "iconMapping": "hdi",
            "agg_type": "mean_year_country"
        },
    ]
    df_filtered = df[(df['ISO_Code'].isin(country_codes)) & (df['Year'] >= fromYear) & (df['Year'] <= toYear)]
    if df_filtered.empty:
        return [
            {
                "value": None,
                "unit": meta["unit"],
                "description": meta["description"],
                "iconMapping": meta["iconMapping"],
                "calculationNote": "No data available for the selected filters."
            }
            for meta in indicators_meta
        ]
    lastest_year = df_filtered["Year"].max()
    df_lastest = df_filtered[df_filtered["Year"] == lastest_year]

    result = []

    for meta in indicators_meta:
        name = meta["name"]
        agg_type = meta["agg_type"]
        if name not in df_filtered.columns:
            value = None
            calculationNote_text = "No data available for the selected filters."
        elif agg_type == "sum":
            value = 0.0
            calculationNote_lines = []
            calculationNote_lines.append(
                f"'{meta["description"]}' is calculated as the sum of all values "
                f"across selected countries and years ({fromYear:.0f}-{toYear:.0f})"
            )
            calculationNote_lines.append("")
            
            for country, df_country in df_filtered.groupby("Country"):
                calculationNote_lines.append(f"{country}:")
                for _, row in df_country.iterrows():
                    val = row[name]
                    if pd.isna(val):
                        continue
                    value += val
                    calculationNote_lines.append(
                        f"- {int(row['Year'])}: {format_number(val, 2)} {meta['unit']}"
                    )
                calculationNote_lines.append("")
            calculationNote_lines.append(
                f"Total = {format_number(value, 2)} {meta['unit']}"
            )
            calculationNote_text = "\n".join(calculationNote_lines)

        elif agg_type == "latest_sum":
            value = 0.0
            calculationNote_lines = []

            calculationNote_lines.append(
                f"'{meta['description']}' is calculated as the sum of the latest available values "
                f"for all selected country in year {lastest_year:.0f}"
            )
            calculationNote_lines.append("")

            for _, row in df_lastest.iterrows():
                val = row[name]
                value += val
                calculationNote_lines.append(
                    f"- {row['Country']}: {format_number(val,2)} {meta['unit']}"
                )
            
            calculationNote_lines.append("")
            calculationNote_lines.append(
                f"Total ({lastest_year:.0f}): {format_number(value, 2)} {meta['unit']}"
            )
            calculationNote_text = "\n".join(calculationNote_lines)
        elif agg_type == "weighted_latest":
            # if 'Population' in df_lastest.columns:
            #     value = (
            #         (df_lastest[name] * df_lastest["Population"]).sum() 
            #         / (df_lastest["Population"].sum())
            #     )
            # else: 
            #     value = df_lastest[name].mean()
            calculationNote_lines = []

            calculationNote_lines.append(
                f"'{meta['description']}' is calculated as a population-weighted average "
                f"across selected countries for year {lastest_year:.0f}."
            )
            calculationNote_lines.append(
                "Each country's value is weighted by its population:"
            )
            calculationNote_lines.append(
                "Weighted average = Sum(value × population) / Sum(population)"
            )
            calculationNote_lines.append("")

            weighted_sum = 0.0
            population_sum = 0.0

            for _, row in df_lastest.iterrows():
                val = row[name]
                pop = row["Population"]

                weighted_sum += val * pop
                population_sum += pop

                calculationNote_lines.append(
                    f"- {row['Country']}: "
                    f"{format_number(val,2)} {meta['unit']} × {int(pop):,} people"
                )

            value = weighted_sum / population_sum

            calculationNote_lines.append("")
            calculationNote_lines.append(
                f"Final weighted value ({lastest_year:.0f}) = {format_number(value, 4)} {meta['unit']}"
            )

            calculationNote_text = "\n".join(calculationNote_lines)
        elif agg_type == "mean_year_country":
            # value = (
            #     df_filtered
            #         .groupby("ISO_Code")[name]
            #         .mean() # mean theo year
            #         .mean() # mean theo country
            # )
            calculationNote_lines = []

            calculationNote_lines.append(
                f"'{meta['description']}' is calculated using a two-step averaging process "
                f"across selected countries and years ({fromYear:.0f}-{toYear:.0f})."
            )
            calculationNote_lines.append("")
            calculationNote_lines.append("Step 1: For each country, calculate the average across years.")
            calculationNote_lines.append("Step 2: Take the average of those country-level averages.")
            calculationNote_lines.append("")

            country_means = []

            for country, df_country in df_filtered.groupby("Country"):
                yearly_values = df_country.sort_values("Year")[name].tolist()
                country_mean = sum(yearly_values) / len(yearly_values)
                country_means.append(country_mean)

                years_str = ", ".join(
                    f"{int(y)}: {v:.3f}{meta['unit']}"
                    for y, v in zip(df_country["Year"], yearly_values)
                )

                calculationNote_lines.append(
                    f"- {country}: mean({years_str}) = {country_mean:.3f}{meta['unit']}"
                )

            value = sum(country_means) / len(country_means)

            calculationNote_lines.append("")
            calculationNote_lines.append(
                f"Final value = mean of {len(country_means)} country averages "
                f"= {format_number(value, 3)}{meta['unit']}"
            )

            calculationNote_text = "\n".join(calculationNote_lines)
        else:
            value = None
            calculationNote_text = "No data available for the selected filters."
        result.append({
            "value": value,
            "unit": meta["unit"],
            "description": meta["description"],
            "iconMapping": meta["iconMapping"],
            "calculationNote": calculationNote_text
        })
    return result

def get_gdp_allocation(country_codes: List[str], fromYear: int, toYear: int):
    indicators = [
        {
            "name": "Industry_on_GDP",
            "label": "Industry",
            "unit": "%"
        },
        {
            "name": "Government_Expenditure_on_Education",
            "label": "Education",
            "unit": "%"
        }
    ]
    df_filtered = df[
        (df["ISO_Code"].isin(country_codes)) &
        (df["Year"] >= fromYear) &
        (df["Year"] <= toYear)
    ]
    if df_filtered.empty:
        return [
                {
                    "key": meta["name"],
                    "label": meta["label"],
                    "value": None,
                    "unit": meta["unit"]
                }
                for meta in indicators
            ] + [
                {
                    "key": "Other",
                    "label": "Other",
                    "value": None,
                    "unit": "%"
                }
            ]
    
    result = []
    otherPercentValue = 100
    for meta in indicators:
        name = meta["name"]

        if name not in df_filtered.columns:
            value = None
        else:
            country_means = (
                df_filtered.groupby("ISO_Code")[name]
                .mean() # mean theo year
            )

            value = country_means.mean() # mean theo country

        otherPercentValue -= value

        result.append({
            "key": name,
            "label": meta["label"],
            "value": round(value, 2) if value is not None else None,
            "unit": meta["unit"]
        })
    
    result.append( {
        "key": "Other",
        "label": "Other",
        "value": round(otherPercentValue, 2) if value is not None else None,
        "unit": "%"
    })
    
    return result

def get_distribution_energy(country_codes: List[str], fromYear: int, toYear: int):
    indicators = [
        {
            "name": "Renewable_Energy_MWh",
            "label": "Renewable Energy",
            "unit": "MWh"
        }
    ]
    TOTAL_ENERGY_COLS_NAME = "Energy_MWh"
    df_filtered = df[
        (df["ISO_Code"].isin(country_codes)) &
        (df["Year"] >= fromYear) &
        (df["Year"] <= toYear)
    ]
    if df_filtered.empty:
        return [
                {
                    "key": meta["name"],
                    "label": meta["label"],
                    "value": None,
                    "unit": meta["unit"]
                }
                for meta in indicators
            ] + [
                {
                    "key": "Other_Energy",
                    "label": "Other Energy",
                    "value": None,
                    "unit": "MWh"
                }
            ]
    
    result = []
    otherValue = df_filtered[TOTAL_ENERGY_COLS_NAME].sum()
    for meta in indicators:
        name = meta["name"]

        if name not in df_filtered.columns:
            value = None
        else:
            value = df_filtered[name].sum()

        otherValue -= value

        result.append({
            "key": name,
            "label": meta["label"],
            "value": round(value, 2) if value is not None else None,
            "unit": meta["unit"]
        })
    
    result.append( {
        "key": "Other_Energy",
        "label": "Other Energy",
        "value": round(otherValue, 2) if value is not None else None,
        "unit": "MWh"
    })
    
    return result
def get_distribution_land_area(country_codes: List[str], fromYear: int, toYear: int):
    indicators = [
        {
            "name": "Forest_Area_ha",
            "label": "Forest Area",
            "unit": "hecta"
        }
    ]
    TOTAL_LAND_AREA = "Area_ha"
    df_filtered = df[
        (df["ISO_Code"].isin(country_codes)) &
        (df["Year"] == toYear) 
    ] # Lấy năm mới nhất
    if df_filtered.empty:
        return [
                {
                    "key": meta["name"],
                    "label": meta["label"],
                    "value": None,
                    "unit": meta["unit"]
                }
                for meta in indicators
            ] + [
                {
                    "key": "Other_Area",
                    "label": "Other Area",
                    "value": None,
                    "unit": "hecta"
                }
            ]
    
    result = []
    otherValue = df_filtered[TOTAL_LAND_AREA].sum()
    for meta in indicators:
        name = meta["name"]

        if name not in df_filtered.columns:
            value = None
        else:
            value = df_filtered[name].sum()

        otherValue -= value

        result.append({
            "key": name,
            "label": meta["label"],
            "value": round(value, 2) if value is not None else None,
            "unit": meta["unit"]
        })
    
    result.append( {
        "key": "Other_Area",
        "label": "Other Area",
        "value": round(otherValue, 2) if value is not None else None,
        "unit": "hecta"
    })
    
    return result