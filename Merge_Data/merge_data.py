import pandas as pd
import numpy as np
from dataclasses import dataclass
import re
import os
import shutil

@dataclass
class DataFrames_Obj:
    iso: pd.DataFrame = None
    co2: pd.DataFrame = None
    hdr: pd.DataFrame = None
    industry_gdp: pd.DataFrame = None
    treecover_loss: pd.DataFrame = None
    carbon_regulation: pd.DataFrame = None
    forest_area_percent: pd.DataFrame = None
    forest_area_sqkm: pd.DataFrame = None
    population: pd.DataFrame = None
    car_sales: pd.DataFrame = None
    climate: pd.DataFrame = None
    world_cri: pd.DataFrame = None
    fossil_fuel_share: pd.DataFrame = None
    edu_share: pd.DataFrame = None
    env_share: pd.DataFrame = None
    deforestation: pd.DataFrame = None
    gdp: pd.DataFrame = None
    renewable_energy: pd.DataFrame = None

def load_dataframes() -> DataFrames_Obj:
    """
    Đọc tất cả dataframe feature thu thập được bằng file csv

    Returns:
    Một object chứa tất cả dataframe, với atrribute_name là tên của đặc trưng tương ứng:
        + iso: dữ liệu iso_code 
        + co2: dữ liệu phát thải co2
        + hdr: dữ liệu ...
        +....
    """


    # iso_code_file_path
    iso_code_file_path = "iso_alpha3_codes.csv"

    #An
    folder_an_path = "An_Data"
    final_hdr_data_file_path = folder_an_path + "/hdr_data_final.csv"
    final_industry_gdp_data_file_path = folder_an_path + "/industry_on_GDP.csv"
    final_treecover_loss_data_file_path = folder_an_path + "/treecover_loss_ha_final.csv"
    #Long
    folder_long_path = "Long_Data"
    final_carbon_regulations_file_path = folder_long_path + "/carbon_regulations.csv"
    final_forest_area_percent_file_path = folder_long_path + "/forest_area_percent.csv"
    final_forest_area_sqkm_file_path = folder_long_path + "/forest_area_sqkm.csv"
    final_population_file_path = folder_long_path + "/population.csv"
    #Tai
    folder_tai_path = "Tai_Data"
    final_car_sales_file_path = folder_tai_path + "/car-sales-clean.csv"
    final_climate_file_path = folder_tai_path + "/Global_Climate_Physical_Risk_Index_clean.csv"
    final_cri_file_path = folder_tai_path + "/world_CRI_clean.csv"
    #Linh
    folder_linh_path = "Linh_Data"
    final_fossil_fuel_share_file_path = folder_linh_path + "/fossil_fuel_share.csv"
    final_education_share_file_path = folder_linh_path + "/share-of-education-in-government-expenditure.csv"
    final_env_protect_share_file_path = folder_linh_path + "/share-of-environmentprotect-in-government-expenditure.csv"
    #Ha
    folder_ha_path = "Ha_Data"
    final_deforestation_file_path = folder_ha_path + "/deforestation_mapped_iso_code.csv"
    final_gdp_file_path = folder_ha_path + "/gdp.csv"
    final_renewable_energy_file_path = folder_ha_path + "/renewable_energy.csv"
    #Vinh
    folder_vinh_path = "Vinh_Data"
    final_co2_data_file_path = folder_vinh_path +  "/final_co2_data.csv"

    # read dataframe
    df_iso = pd.read_csv(iso_code_file_path, engine='python')

    df_hdr = pd.read_csv(final_hdr_data_file_path, engine='python', sep=";")
    df_industry_gdp = pd.read_csv(final_industry_gdp_data_file_path, engine='python')
    df_treecover_loss = pd.read_csv(final_treecover_loss_data_file_path, engine='python')

    df_carbon_regulations = pd.read_csv(final_carbon_regulations_file_path, engine='python')
    df_forest_area_percent = pd.read_csv(final_forest_area_percent_file_path, engine='python')
    df_forest_area_sqkm = pd.read_csv(final_forest_area_sqkm_file_path, engine='python')
    df_population = pd.read_csv(final_population_file_path, engine='python')

    df_car_sales = pd.read_csv(final_car_sales_file_path, engine='python')
    df_climate = pd.read_csv(final_climate_file_path, engine='python')
    df_world_cri = pd.read_csv(final_cri_file_path, engine='python')

    df_fossil_fuel_share = pd.read_csv(final_fossil_fuel_share_file_path, engine='python')
    df_edu_share = pd.read_csv(final_education_share_file_path, engine='python')
    df_env_share = pd.read_csv(final_env_protect_share_file_path, engine='python')

    df_deforestation = pd.read_csv(final_deforestation_file_path, engine='python')
    df_gdp = pd.read_csv(final_gdp_file_path, engine='python', index_col=0)
    df_renewable_energy = pd.read_csv(final_renewable_energy_file_path, engine='python', index_col=0)


    df_co2 = pd.read_csv(final_co2_data_file_path, engine='python')
    return DataFrames_Obj(
        iso = df_iso,
        co2 = df_co2,
        hdr = df_hdr,
        industry_gdp = df_industry_gdp,
        treecover_loss = df_treecover_loss,
        carbon_regulation = df_carbon_regulations,
        forest_area_percent = df_forest_area_percent,
        forest_area_sqkm = df_forest_area_sqkm,
        population = df_population,
        car_sales = df_car_sales,
        climate = df_climate,
        world_cri = df_world_cri,
        fossil_fuel_share = df_fossil_fuel_share,
        edu_share = df_edu_share,
        env_share = df_env_share,
        deforestation = df_deforestation,
        gdp = df_gdp,
        renewable_energy = df_renewable_energy
    )

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        col_clean = re.sub(r'[^a-zA-Z0-9_]', '', col.strip().lower())

        if col_clean in ["name", "country","country_name", "countryname"]:
            rename_map[col] = "country"
        elif col_clean in ["code", "iso_code", "iso_alpha3", "country_code", "countrycode"]:
            rename_map[col] = "iso_code"
        elif col_clean in ["year"]:
            rename_map[col] = "year"
        else:
            rename_map[col] = f"ft_{col_clean}"
    
    df = df.rename(columns=rename_map)

    col_order = ["year", "iso_code", "country"]
    ordered_col = [c for c in col_order if c in df.columns]
    remaining_col = [c for c in df.columns if c not in ordered_col]
    df = df[ordered_col + remaining_col]

    return df

def normalize_columns_all(df_obj: DataFrames_Obj) -> DataFrames_Obj:
    for name_df, df in vars(df_obj).items():
        if isinstance(df, pd.DataFrame):
            vars(df_obj)[name_df] = normalize_columns(df)
    return df_obj

def filter_by_valid_iso(df_obj: DataFrames_Obj) -> DataFrames_Obj:
    valid_iso_codes: set

    if "iso_code" in df_obj.iso.columns:
        valid_iso_codes = set(df_obj.iso["iso_code"].dropna().unique())

    for name_df, df in vars(df_obj).items():
        if (isinstance(df, pd.DataFrame) and name_df != "iso"):
            if "iso_code" in df.columns:
                filter_df = df[df["iso_code"].notna() & df["iso_code"].isin(valid_iso_codes)]
                vars(df_obj)[name_df] = filter_df
    return df_obj

def merge_all_features(df_obj: DataFrames_Obj, base_df_name: str = "co2", exclude_df_name: list = ['iso']) -> pd.DataFrame:
    """
    Merge tất cả các DataFrame của các feature trong df_obj vào base_df theo kiểu outer join

    Parameter:
    - df_obj: Object chứa các dataframe
    - base_df_name: tên của dataframe là base, default là 'co2'
    - exclude_df_name: danh sách tên các dataframe không merge, default ['iso']
    """

    if (exclude_df_name is None):
        exclude_df_name = []
    exclude_df_name.append(base_df_name) # không merge base vào base

    merged_df = getattr(df_obj, base_df_name)

    for df_name, df in vars(df_obj).items():
        if (isinstance(df, pd.DataFrame) and df_name not in exclude_df_name):
            #Xác định feature_col trong df
            feature_cols = [c for c in df.columns if c.startswith("ft_")]
            if not feature_cols:
                continue # bỏ qua việc merge nếu không có cột feature
            
            merged_df = pd.merge(
                merged_df,
                df[['iso_code', 'year'] + feature_cols],
                on = ['iso_code', 'year'],
                how = 'outer'
            )

            if 'country' in df.columns:
                keys = ['iso_code','year']
                country_series = df.set_index(keys)['country']
                merged_df = merged_df.set_index(keys)
                merged_df['country'] = merged_df['country'].combine_first(country_series)
                merged_df = merged_df.reset_index()



    key_cols = ['country', 'iso_code', 'year']
    feature_cols_all = [c for c in merged_df.columns if c not in key_cols]
    merged_df = merged_df[key_cols + feature_cols_all]

    return merged_df

def merge_feature_with_empty_base(df_obj: DataFrames_Obj, exclude_df_name: list = ['iso']) -> pd.DataFrame:
    if exclude_df_name is None:
        exclude_df_name = []
    exclude_df_name.append('iso')
    
    start_year = 2000
    end_year = 2025
    years = list(range(start_year, end_year + 1))
    df_iso = df_obj.iso[['country','iso_code']].drop_duplicates()
    df_base = pd.DataFrame([(iso, country, year)
                            for iso, country in zip(df_iso['iso_code'], df_iso['country'])
                            for year in years], columns=['iso_code', 'country', 'year'])
    merged_df = df_base.copy()

    for df_name, df in vars(df_obj).items():
        if (isinstance(df, pd.DataFrame) and df_name not in exclude_df_name):
            feature_cols = [c for c in df.columns if c not in ['iso_code', 'country', 'year']]
            if not feature_cols:
                continue

            merged_df = pd.merge(
                merged_df,
                df[['iso_code', 'year'] + feature_cols],
                on=['iso_code', 'year'],
                how='left'
            )
    key_cols = ['country', 'iso_code', 'year']
    feature_cols_all = [c for c in merged_df.columns if c not in key_cols]
    merged_df = merged_df[key_cols + feature_cols_all]

    return merged_df

def check_merge(df_final: pd.DataFrame, df_obj: DataFrames_Obj, output_dir="merge_check_results"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    df_final_keys = df_final[['iso_code', 'year']].drop_duplicates()

    for name_df, df in vars(df_obj).items():
        if not isinstance(df, pd.DataFrame) or name_df == 'iso':
            continue

        if not {'iso_code', 'year'}.issubset(df.columns):
            continue

        feature_cols = [c for c in df.columns if c not in ['iso_code', 'country', 'year']]
        if not feature_cols:
            continue

        df_not_in_final = df.merge(df_final_keys, on=['iso_code', 'year'], how='left', indicator=True)
        df_not_in_final = df_not_in_final[df_not_in_final['_merge'] == 'left_only'].drop(columns=['_merge'])

        if not df_not_in_final.empty:
            file_path = os.path.join(output_dir, f"{name_df}_missing_in_final.csv")
            df_not_in_final.to_csv(file_path, index=False)

        df_compare = pd.merge(
            df[['iso_code', 'year'] + feature_cols],
            df_final[['iso_code', 'year'] + feature_cols],
            on=['iso_code', 'year'],
            how='inner',
            suffixes=('_orig', '_final')
        )

        mismatches = []
        for c in feature_cols:
            cond = (
                (df_compare[f"{c}_orig"].notna() | df_compare[f"{c}_final"].notna()) & 
                (df_compare[f"{c}_orig"] != df_compare[f"{c}_final"])
            )
            diff = df_compare[cond].copy()
            if not diff.empty:
                diff['_feature'] = c
                mismatches.append(diff[['iso_code', 'year', f"{c}_orig", f"{c}_final", '_feature']])

        if mismatches:
            df_mismatch = pd.concat(mismatches, ignore_index=True)
            file_path = os.path.join(output_dir, f"{name_df}_value_mismatch.csv")
            df_mismatch.to_csv(file_path, index=False)

def main():
    df_obj = load_dataframes()
    df_obj = normalize_columns_all(df_obj)
    df_obj = filter_by_valid_iso(df_obj)

    # final_df = merge_all_features(df_obj=df_obj, base_df_name="co2", exclude_df_name=["iso"])
    # final_df.to_csv("temp.csv", index=False)

    final_df_merged_with_base = merge_feature_with_empty_base(df_obj=df_obj, exclude_df_name=['iso'])
    final_df_merged_with_base.to_csv('final.csv')
    
    check_merge(final_df_merged_with_base, df_obj)

if __name__ == "__main__":
    main()
    
