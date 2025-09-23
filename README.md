# Smart Agri API

This is a FastAPI backend with for our SIH Project.

## Features
- `GET /` → Mobile-responsive landing page
- `GET /health` → Health check with uptime info
- `POST /predict_yield` → Predict crop yield based on input features
- `POST /recommend_crop` → Recommend best crop given soil and weather conditions
- `GET /errors` → Public error log dashboard
- `GET /metrics` → API usage stats (per endpoint breakdown, last request, peak load)
- `GET /docs` → Swagger interactive API documentation
- `GET /redoc` → ReDoc interactive API documentation

## Installation
```bash
git clone https://github.com/AishikTokdar/SIH2025-API
cd SIH2025-API
pip install -r requirements.txt
```
## Run
```bash
uvicorn main:app --reload
```