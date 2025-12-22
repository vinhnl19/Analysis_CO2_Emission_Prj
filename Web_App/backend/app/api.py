# backend/app/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import SeqPredictReq, ManualPredictReq, RecommendReq, GetDataCardReq
from .engine import predict_sequence, predict_manual, recommend_changes_es, get_country_list

app = FastAPI(title="CO2 Advisor API", version="1.0")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # thay domain khi deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "CO2 Advisor API is running!"}


# ========================================================
# ==================   API ROUTES   ======================
# ========================================================

@app.post("/predict/sequence")
def api_predict_sequence(req: SeqPredictReq):
    try:
        result = predict_sequence(req.country, req.sequence_features)
        return {"predicted_co2": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict/manual")
def api_predict_manual(req: ManualPredictReq):
    try:
        result = predict_manual(req.country, req.features)
        return {"predicted_co2": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/recommend")
def api_recommend(req: RecommendReq):
    try:
        result = recommend_changes_es(
            req.country,
            req.feature_selection,
            req.fixed_features,
            req.co2_target
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
    

@app.get("/get-country-list")
def api_get_country_list():
    try:
        result = get_country_list()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
    