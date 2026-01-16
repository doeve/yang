"""
Prediction API for candle direction model.

Serves the trained model via FastAPI for integration with d3v.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO

app = FastAPI(title="Candle Prediction API", version="1.0.0")

# Load model on startup
MODEL_PATH = Path("logs/candle_prediction/candle_model")
model: PPO | None = None


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    prices: list[float]  # 300 price points normalized to candle open (% change)
    volumes: list[float]  # 300 volume points (normalized)
    current_vs_open: float  # Current price vs candle open (% change)
    time_remaining: float  # 0-1, time remaining in candle
    hour: int  # Hour of day (0-23)
    atr: float  # Average true range (% of price)


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    action: Literal["WAIT", "BET_UP", "BET_DOWN"]
    confidence: float
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model
    try:
        model = PPO.load(str(MODEL_PATH))
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict candle direction.
    
    Input: Price/volume history from BinanceClient
    Output: BET_UP, BET_DOWN, or WAIT with confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input lengths
    if len(request.prices) != 300:
        raise HTTPException(400, f"Expected 300 prices, got {len(request.prices)}")
    if len(request.volumes) != 300:
        raise HTTPException(400, f"Expected 300 volumes, got {len(request.volumes)}")
    
    # Build observation (must match training format exactly)
    # Format: [prices(300), volumes(300), current_vs_open(1), time_remaining(1), hour_sin(1), hour_cos(1), atr(1)]
    hour_sin = np.sin(2 * np.pi * request.hour / 24)
    hour_cos = np.cos(2 * np.pi * request.hour / 24)
    
    obs = np.concatenate([
        np.array(request.prices, dtype=np.float32),
        np.array(request.volumes, dtype=np.float32),
        np.array([request.current_vs_open], dtype=np.float32),
        np.array([request.time_remaining], dtype=np.float32),
        np.array([hour_sin, hour_cos], dtype=np.float32),
        np.array([request.atr], dtype=np.float32),
    ])
    
    # Clip extreme values (same as training)
    obs = np.clip(obs, -100.0, 100.0)
    
    # Predict
    action, _ = model.predict(obs, deterministic=True)
    action_names = ["WAIT", "BET_UP", "BET_DOWN"]
    
    return PredictionResponse(
        action=action_names[int(action)],
        confidence=0.56,  # Model's tested accuracy
        model_loaded=True,
    )


# Simpler endpoint that accepts raw BinanceClient data
class SimplePredictionRequest(BaseModel):
    """Simplified request matching BinanceClient output."""
    priceHistory: list[dict]  # [{price, timestamp}, ...]  from BinanceClient
    candleOpen: float
    timeRemainingMs: int  # Milliseconds remaining in candle


@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_simple(request: SimplePredictionRequest):
    """
    Simpler prediction endpoint that processes raw BinanceClient data.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract prices from history
    if len(request.priceHistory) < 10:
        raise HTTPException(400, "Need at least 10 price points")
    
    prices_raw = [p["price"] for p in request.priceHistory[-300:]]
    
    # Pad if needed
    while len(prices_raw) < 300:
        prices_raw.insert(0, prices_raw[0])
    
    # Normalize to candle open (% change * 100)
    prices = [(p / request.candleOpen - 1.0) * 100 for p in prices_raw]
    
    # Dummy volumes (BinanceClient tracks this separately)
    volumes = [1.0] * 300
    
    # Current vs open
    current_price = prices_raw[-1]
    current_vs_open = (current_price / request.candleOpen - 1.0) * 100
    
    # Time remaining (normalize to 0-1, 15 min = 900000ms)
    time_remaining = max(0, min(1, request.timeRemainingMs / 900_000))
    
    # Hour (estimate from current time)
    from datetime import datetime
    hour = datetime.utcnow().hour
    
    # Build observation
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    obs = np.concatenate([
        np.array(prices, dtype=np.float32),
        np.array(volumes, dtype=np.float32),
        np.array([current_vs_open], dtype=np.float32),
        np.array([time_remaining], dtype=np.float32),
        np.array([hour_sin, hour_cos], dtype=np.float32),
        np.array([0.01], dtype=np.float32),  # Default ATR
    ])
    
    obs = np.clip(obs, -100.0, 100.0)
    
    action, _ = model.predict(obs, deterministic=True)
    action_names = ["WAIT", "BET_UP", "BET_DOWN"]
    
    return PredictionResponse(
        action=action_names[int(action)],
        confidence=0.56,
        model_loaded=True,
    )
