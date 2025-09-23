from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
import time

# Record the app start time
app_start_time = time.time()

# =====================
# 1. Logging Setup
# =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================
# 2. Load Models
# =====================
try:
    yield_model = joblib.load("yield_predictor.pkl")
    crop_model = joblib.load("crop_recommendation_topk_model.pkl")
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

# =====================
# 3. Initialize FastAPI
# =====================
app = FastAPI(title="Smart Agri API", description="Predict crop yield and recommend crops.")

# =====================
# 4. Enable CORS
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# =====================
# 5. Request Schemas with validation
# =====================
class YieldRequest(BaseModel):
    N: float = Field(..., ge=0, le=500)
    P: float = Field(..., ge=0, le=500)
    K: float = Field(..., ge=0, le=500)
    temperature: float = Field(..., ge=-10, le=50)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=3, le=10)
    rainfall: float = Field(..., ge=0)
    Soil_OC: float = Field(..., ge=0)
    Fertilizer_kg_ha: float = Field(..., ge=0)
    Pest_Index: float = Field(..., ge=0)
    Irrigation_mm: float = Field(..., ge=0)

class CropRequest(BaseModel):
    N: float = Field(..., ge=0, le=500)
    P: float = Field(..., ge=0, le=500)
    K: float = Field(..., ge=0, le=500)
    temperature: float = Field(..., ge=-10, le=50)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=3, le=10)
    rainfall: float = Field(..., ge=0)

# =====================
# 6. Health Check Endpoint
# =====================
@app.get("/health", summary="Health Check")
def health_check():
    uptime_seconds = int(time.time() - app_start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"
    return {
        "status": "ok",
        "version": "1.0",
        "uptime": uptime_str
    }

# =====================
# 7. /predict_yield Endpoint
# =====================
@app.post("/predict_yield", summary="Predict Crop Yield", description="Returns predicted crop yield in t/ha.")
def predict_yield(req: YieldRequest):
    try:
        data = req.dict()
        logging.info(f"Yield request received: {data}")

        # Derived features
        Nutrient_Balance_Index = (data["N"] + data["P"] + data["K"]) / 3
        Stress_Index = data["temperature"] * (1 - data["humidity"] / 100)
        Rainfall_N_Interaction = data["rainfall"] * data["N"]

        # Model input
        input_array = [[
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"], data["ph"],
            data["rainfall"], data["Soil_OC"], data["Fertilizer_kg_ha"],
            data["Pest_Index"], data["Irrigation_mm"],
            Nutrient_Balance_Index, Stress_Index, Rainfall_N_Interaction
        ]]

        prediction = yield_model.predict(input_array)[0]
        logging.info(f"Predicted yield: {prediction}")

        return {"predicted_yield_t_ha": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error predicting yield: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# =====================
# 8. /recommend_crop Endpoint
# =====================
@app.post("/recommend_crop", summary="Recommend Crops", description="Returns top-k recommended crops with probabilities.")
def recommend_crop(req: CropRequest, k: int = 3):
    try:
        data = req.dict()
        logging.info(f"Crop recommendation request received: {data}, top-k={k}")

        input_df = pd.DataFrame([data])
        probs = crop_model.predict_proba(input_df)[0]
        classes = crop_model.classes_
        top_k_idx = np.argsort(probs)[::-1][:k]
        top_k_crops = [
            {"crop": classes[i], "probability": round(float(probs[i]*100), 2)}
            for i in top_k_idx
        ]

        logging.info(f"Top-{k} crops: {top_k_crops}")
        return {"top_crops": top_k_crops}
    except Exception as e:
        logging.error(f"Error in crop recommendation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
