import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# backend/
BASE_DIR = os.path.abspath(
    os.path.join(__file__, "../../..")
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_GRU = os.path.join(MODELS_DIR, "best_model_gru3_final.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_quantile.save")
LE_PATH = os.path.join(MODELS_DIR, "labelencoder_country.save")

XGB_MODEL = os.path.join(MODELS_DIR, "Model_XGBoost_Final.joblib")
XGB_LE = os.path.join(MODELS_DIR, "Encoder_Country_Final.joblib")

DATA_PATH = os.path.join(DATA_DIR, "filled_data.csv")

print("Loading resources...")

model_gru = load_model(MODEL_GRU)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)

model_xgb = joblib.load(XGB_MODEL)
le_xgb = joblib.load(XGB_LE)

df = pd.read_csv(DATA_PATH)

print("Resources loaded.")
