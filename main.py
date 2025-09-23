from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
from collections import deque

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
    "crop_avg_response_time": 0,
    "last_request_time": None,
    "per_endpoint": {
        "/predict_yield": 0,
        "/recommend_crop": 0
    },
    "peak_load": {
        "last_hour": 0,
        "last_day": 0
    }
}

# Cache recent requests
cache = {}
cache_size = 50
recent_requests = deque(maxlen=cache_size)

# Error logs for /errors endpoint
error_logs = deque(maxlen=100)

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

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌾 Smart Agri API</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gradient-to-r from-green-600 to-green-400 text-white min-h-screen flex flex-col items-center justify-center p-6">
        <div class="text-center max-w-2xl">
            <h1 class="text-4xl md:text-6xl font-bold mb-4">🌾 Smart Agri API</h1>
            <p class="text-lg md:text-xl mb-6">Your one-stop solution for crop yield prediction and smart recommendations.</p>
            
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                <a href="/docs" class="px-4 py-3 bg-green-800 rounded-lg shadow hover:bg-green-900 transition">Swagger Docs</a>
                <a href="/redoc" class="px-4 py-3 bg-green-800 rounded-lg shadow hover:bg-green-900 transition">ReDoc Docs</a>
                <a href="/health" class="px-4 py-3 bg-green-800 rounded-lg shadow hover:bg-green-900 transition">Health Check</a>
                <a href="/metrics" class="px-4 py-3 bg-green-800 rounded-lg shadow hover:bg-green-900 transition">Metrics</a>
                <a href="/errors" class="px-4 py-3 bg-green-800 rounded-lg shadow hover:bg-green-900 transition">Error Logs</a>
            </div>
        </div>

        <footer class="mt-10 text-sm opacity-80 text-center">
            &copy; 2025 Smart Agri API 🌱 | Built with <span class="text-red-400">❤</span> using FastAPI
        </footer>
    </body>
    </html>
    """

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

        # Input anomaly detection
        anomalies = [k for k, v in data.items() if v < 0]
        if anomalies:
            raise HTTPException(status_code=422, detail=f"Invalid negative values in: {anomalies}")

        # Caching
        cache_key = tuple(data.items())
        if cache_key in cache:
            return cache[cache_key]

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

        # Update stats
        api_stats["total_yield_requests"] += 1
        api_stats["per_endpoint"]["/predict_yield"] += 1
        api_stats["last_request_time"] = timestamp
        elapsed = time.time() - start_time
        api_stats["yield_avg_response_time"] = (
            ((api_stats["yield_avg_response_time"] * (api_stats["total_yield_requests"] - 1)) + elapsed)
            / api_stats["total_yield_requests"]
        )

        result = {"predicted_yield_t_ha": round(prediction, 2), "timestamp": timestamp}
        cache[cache_key] = result
        recent_requests.append({"endpoint": "/predict_yield", "input": data, "output": result, "time": timestamp})
        return result

    except Exception as e:
        error_logs.append({"endpoint": "/predict_yield", "error": str(e), "time": datetime.now().isoformat()})
        logging.error(f"Error predicting yield: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend_crop", summary="Recommend Crops", description="Returns top-k recommended crops with probabilities.")
def recommend_crop(req: CropRequest, k: int = 3):
    start_time = time.time()
    try:
        data = req.dict()
        logging.info(f"Crop recommendation request received: {data}, top-k={k}")

        # Input anomaly detection
        anomalies = [k for k, v in data.items() if v < 0]
        if anomalies:
            raise HTTPException(status_code=422, detail=f"Invalid negative values in: {anomalies}")

        input_df = pd.DataFrame([data])
        probs = crop_model.predict_proba(input_df)[0]
        classes = crop_model.classes_
        top_k_idx = np.argsort(probs)[::-1][:k]
        top_k_crops = [
            {"crop": classes[i], "probability": round(float(probs[i]*100), 2)}
            for i in top_k_idx
        ]

        timestamp = datetime.now().isoformat()
        api_stats["total_crop_requests"] += 1
        api_stats["per_endpoint"]["/recommend_crop"] += 1
        api_stats["last_request_time"] = timestamp
        elapsed = time.time() - start_time
        api_stats["crop_avg_response_time"] = (
            ((api_stats["crop_avg_response_time"] * (api_stats["total_crop_requests"] - 1)) + elapsed)
            / api_stats["total_crop_requests"]
        )

        result = {"top_crops": top_k_crops, "timestamp": timestamp}
        recent_requests.append({"endpoint": "/recommend_crop", "input": data, "output": result, "time": timestamp})
        return result

    except Exception as e:
        error_logs.append({"endpoint": "/recommend_crop", "error": str(e), "time": datetime.now().isoformat()})
        logging.error(f"Error in crop recommendation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics", summary="API Usage Metrics", description="Detailed usage statistics for monitoring.")
def get_metrics():
    return {
        "api_stats": api_stats,
        "recent_requests": list(recent_requests),
        "cache_size": len(cache)
    }

@app.get("/errors", summary="Error Logs", description="Recent error logs (public).")
def get_errors():
    return list(error_logs)

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

# Initial manual ping after short delay
def initial_ping():
    time.sleep(10)
    keep_alive()

threading.Thread(target=initial_ping, daemon=True).start()
