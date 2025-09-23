from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import threading

# Record app start time
app_start_time = time.time()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models
try:
    yield_model = joblib.load("yield_predictor.pkl")
    crop_model = joblib.load("crop_recommendation_topk_model.pkl")
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

app = FastAPI(title="Smart Agri API", description="Predict crop yield and recommend crops.")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# API stats
api_stats = {
    "total_yield_requests": 0,
    "total_crop_requests": 0,
    "yield_avg_response_time": 0,
    "crop_avg_response_time": 0
}

# Base URL will be detected dynamically
base_url = None
base_url_lock = threading.Lock()

# Request Schemas
class YieldRequest(BaseModel):
    N: float = Field(..., ge=0, le=500, description="Nitrogen content in kg/ha")
    P: float = Field(..., ge=0, le=500, description="Phosphorus content in kg/ha")
    K: float = Field(..., ge=0, le=500, description="Potassium content in kg/ha")
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    ph: float = Field(..., ge=3, le=10, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")
    Soil_OC: float = Field(..., ge=0, description="Soil organic carbon (%)")
    Fertilizer_kg_ha: float = Field(..., ge=0, description="Fertilizer applied in kg/ha")
    Pest_Index: float = Field(..., ge=0, description="Pest index")
    Irrigation_mm: float = Field(..., ge=0, description="Irrigation applied in mm")

class CropRequest(BaseModel):
    N: float = Field(..., ge=0, le=500, description="Nitrogen content in kg/ha")
    P: float = Field(..., ge=0, le=500, description="Phosphorus content in kg/ha")
    K: float = Field(..., ge=0, le=500, description="Potassium content in kg/ha")
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    ph: float = Field(..., ge=3, le=10, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")

# Middleware to detect base URL dynamically
@app.middleware("http")
async def detect_base_url(request: Request, call_next):
    global base_url
    if base_url is None:
        with base_url_lock:
            if base_url is None:
                base_url = f"{request.url.scheme}://{request.url.netloc}"
                logging.info(f"Detected base URL: {base_url}")
    response = await call_next(request)
    return response

@app.get("/health")
def health_check():
    uptime_seconds = int(time.time() - app_start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"
    return {
        "status": "ok",
        "version": "1.0",
        "uptime": uptime_str,
        "api_stats": api_stats
    }

@app.post("/predict_yield", summary="Predict Crop Yield", description="Returns predicted crop yield in t/ha.")
def predict_yield(req: YieldRequest):
    start_time = time.time()
    try:
        data = req.dict()
        logging.info(f"Yield request received: {data}")

        # Handle missing values
        for k, v in data.items():
            if v is None:
                data[k] = 0.0

        # Derived features
        Nutrient_Balance_Index = (data["N"] + data["P"] + data["K"]) / 3
        Stress_Index = data["temperature"] * (1 - data["humidity"] / 100)
        Rainfall_N_Interaction = data["rainfall"] * data["N"]

        input_array = [[
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"], data["ph"],
            data["rainfall"], data["Soil_OC"], data["Fertilizer_kg_ha"],
            data["Pest_Index"], data["Irrigation_mm"],
            Nutrient_Balance_Index, Stress_Index, Rainfall_N_Interaction
        ]]

        prediction = yield_model.predict(input_array)[0]
        timestamp = datetime.now().isoformat()

        api_stats["total_yield_requests"] += 1
        elapsed = time.time() - start_time
        api_stats["yield_avg_response_time"] = (
            ((api_stats["yield_avg_response_time"] * (api_stats["total_yield_requests"] - 1)) + elapsed)
            / api_stats["total_yield_requests"]
        )

        return {"predicted_yield_t_ha": round(prediction, 2), "timestamp": timestamp}

    except Exception as e:
        logging.error(f"Error predicting yield: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend_crop", summary="Recommend Crops", description="Returns top-k recommended crops with probabilities.")
def recommend_crop(req: CropRequest, k: int = 3):
    start_time = time.time()
    try:
        data = req.dict()
        logging.info(f"Crop recommendation request received: {data}, top-k={k}")

        for col, val in data.items():
            if val is None:
                data[col] = 0.0

        input_df = pd.DataFrame([data])
        probs = crop_model.predict_proba(input_df)[0]
        classes = crop_model.classes_
        top_k_idx = np.argsort(probs)[::-1][:k]
        top_k_crops = [
            {"crop": classes[i], "probability": round(float(probs[i]*100), 2)}
            for i in top_k_idx
        ]

        api_stats["total_crop_requests"] += 1
        elapsed = time.time() - start_time
        api_stats["crop_avg_response_time"] = (
            ((api_stats["crop_avg_response_time"] * (api_stats["total_crop_requests"] - 1)) + elapsed)
            / api_stats["total_crop_requests"]
        )

        logging.info(f"Top-{k} crops: {top_k_crops}")
        return {"top_crops": top_k_crops, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logging.error(f"Error in crop recommendation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Keep-alive job
def keep_alive():
    if base_url:
        try:
            url = f"{base_url}/health"
            response = requests.get(url, timeout=5)
            logging.info(f"Keep-alive ping status: {response.status_code}")
        except Exception as e:
            logging.warning(f"Keep-alive ping failed: {e}")
    else:
        logging.info("Base URL not detected yet. Skipping ping.")

scheduler = BackgroundScheduler()
scheduler.add_job(keep_alive, "interval", minutes=10)
scheduler.start()

# Initial manual ping after short delay to avoid Render sleeping
def initial_ping():
    time.sleep(10)  # short delay to let app start
    keep_alive()

threading.Thread(target=initial_ping, daemon=True).start()