# # load once
# MODEL_PATH = os.path.join(BASE_DIR, "model/best_model_gru3_final.keras")
# SCALER_PATH = os.path.join(BASE_DIR, "model/scaler_quantile.save")
# LE_PATH = os.path.join(BASE_DIR, "model/labelencoder_country.save")
# XGB_MODEL_PATH = os.path.join(BASE_DIR, "model/Model_XGBoost_Final.joblib")
# XGB_LE_PATH = os.path.join(BASE_DIR, "model/Encoder_Country_Final.joblib")
# DF_PATH = os.path.join(BASE_DIR, "data/filled_data.csv")

# backend/app/engine.py
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from .es_optimizer import es_optimize_changes    # bạn đã có file này

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ==== PATH MODEL ====
MODEL_GRU = os.path.join(BASE_DIR, "model/best_model_gru3_final.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model/scaler_quantile.save")
LE_PATH = os.path.join(BASE_DIR, "model/labelencoder_country.save")

XGB_MODEL = os.path.join(BASE_DIR, "model/Model_XGBoost_Final.joblib")
XGB_LE = os.path.join(BASE_DIR, "model/Encoder_Country_Final.joblib")

DF_PATH = os.path.join(BASE_DIR, "data/filled_data.csv")

# ==== LOAD ONCE ====
print("Loading models...")
model_gru = load_model(MODEL_GRU)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)
model_xgb = joblib.load(XGB_MODEL)
le_xgb = joblib.load(XGB_LE)
df = pd.read_csv(DF_PATH)
print("Models loaded.")

FEATURE = [
    'Co2_MtCO2',
    'Population',
    'GDP',
    'Industry_on_GDP',
    'Government_Expenditure_on_Education',
    'Global_Climate_Risk_Index',
    'HDI',
    'Renewable_Energy_Percent',
    'Deforest_Percent',
    'Energy_Capita_kWh'
]

FEATURE_CORE = [
    'Population',
    'GDP',
    'Industry_on_GDP',
    'Government_Expenditure_on_Education',
    'Global_Climate_Risk_Index',
    'HDI',
    'Renewable_Energy_Percent',
    'Deforest_Percent',
    'Energy_Capita_kWh'
]


# ==========================================================
# ===============     1) PREDICT SEQUENCE     ===============
# ==========================================================
def predict_sequence(country_name, sequence_features):
    seq_df = pd.DataFrame(sequence_features, columns=FEATURE)

    seq_df_log = np.log1p(seq_df)
    seq_scaled = scaler.transform(seq_df_log)
    X_new = np.expand_dims(seq_scaled, axis=0)

    country_code = 0
    if country_name in le.classes_:
        country_code = int(le.transform([country_name])[0])

    X_country = np.array([[country_code]], dtype='int32')

    y_pred_scaled = model_gru.predict([X_new, X_country], verbose=0)
    y_pred_real = scaler.inverse_transform(
        np.concatenate([y_pred_scaled, np.zeros((1, len(FEATURE)-1))], axis=1)
    )[0, 0]

    return float(np.expm1(y_pred_real))


# ==========================================================
# ===============     2) MANUAL PREDICT       ===============
# ==========================================================
def predict_manual(country, feat_dict):
    x = [feat_dict[f] for f in FEATURE_CORE]
    df_in = pd.DataFrame([x], columns=FEATURE_CORE)

    if country in le_xgb.classes_:
        df_in['Country_Encoded'] = int(le_xgb.transform([country])[0])
    else:
        df_in['Country_Encoded'] = -1

    pred = model_xgb.predict(df_in)[0]
    return float(pred)


# ==========================================================
# ================== 3) RECOMMEND (ES) =======================
# ==========================================================
def recommend_changes_es(country, feature_selection, fixed_features, co2_target):

    def predict_fn(indiv_changes, fixed_inner, country_name):
        last_row = df[df['Country'] == country_name].sort_values("Year").iloc[-1]

        x_val = {}
        x_full = []

        for f in FEATURE_CORE:
            if f in indiv_changes:
                orig = last_row[f]
                pct = indiv_changes[f] / 100.0
                new_val = orig * (1 + pct)
            else:
                new_val = fixed_inner[f]

            x_val[f] = float(new_val)
            x_full.append(new_val)

        df_in = pd.DataFrame([x_full], columns=FEATURE_CORE)
        if country_name in le_xgb.classes_:
            df_in['Country_Encoded'] = int(le_xgb.transform([country_name])[0])
        else:
            df_in['Country_Encoded'] = -1

        pred = model_xgb.predict(df_in)[0]
        return float(pred), x_val

    best_change, best_fit, best_pred, best_x = es_optimize_changes(
        feature_selection,
        fixed_features,
        predict_fn,
        co2_target,
        country
    )

    return {
        "best_change": best_change,
        "best_fitness": float(best_fit),
        "best_predicted_co2": float(best_pred),
        "best_x": best_x
    }
# ==========================================================
# ================== 4) GET COUNTRY LIST =======================
# ==========================================================
def get_country_list():
    country_df = (
        df[['Country', 'ISO_Code', 'Continent']]
        .dropna(subset=['Country', 'ISO_Code', 'Continent'])
        .drop_duplicates(subset=['Country'])
        .sort_values('Country')
    )

    return [
        {
            "country_name": row['Country'],
            "country_code": row['ISO_Code'],
            "continent": row['Continent']
        }
        for _, row in country_df.iterrows()
    ]

# ==========================================================
# ================== 5) GET DATA FOR CARD DASHBOARD =======================
# ==========================================================
def get_data_cards(vountry_codes: list, fromYear, toYear):
    country_df = (
        df[['Country', 'ISO_Code', 'Continent']]
        .dropna(subset=['Country', 'ISO_Code', 'Continent'])
        .drop_duplicates(subset=['Country'])
        .sort_values('Country')
    )

    return [
        {
            "country_name": row['Country'],
            "country_code": row['ISO_Code'],
            "continent": row['Continent']
        }
        for _, row in country_df.iterrows()
    ]