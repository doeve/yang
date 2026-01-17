"""
Prediction API for Multi-Asset Candle Model.

Serves the trained SAC model via FastAPI.
Handles multi-asset observation construction (BTC + DXY + EUR/USD).
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from stable_baselines3 import SAC

from ..data.forex_collector import ForexCollector

app = FastAPI(title="Multi-Asset Prediction API", version="2.0.0")

# Configuration
MODEL_PATH = Path("logs/recurrent_multi_asset_fixed/multi_asset_model")
DATA_DIR = Path("data")

# Global state
model: SAC | None = None
forex_data: dict[str, np.ndarray] = {}  # {symbol: prices_array}


class VariablePredictionResponse(BaseModel):
    """Response with position sizing."""
    action: Literal["WAIT", "BET_UP", "BET_DOWN"]
    confidence: float  # Using position size as confidence proxy
    position_size: float  # 0.0 to 0.5 (max position)
    model_loaded: bool
    forex_freshness: str  # Info about forex data age


class SimplePredictionRequest(BaseModel):
    """Request matching d3v BinanceClient output."""
    priceHistory: list[dict]  # [{price, timestamp}, ...]
    candleOpen: float
    timeRemainingMs: int


async def update_forex_data():
    """Background task to fetch latest forex data."""
    global forex_data
    collector = ForexCollector(str(DATA_DIR))
    
    # Load from disk first
    dxy = collector.load_forex_data("DXY", "1h")
    eurusd = collector.load_forex_data("EURUSD", "1h")
    
    if dxy is None or eurusd is None:
        # Try to fetch if missing
        print("Fetching forex data...")
        enc_results = await collector.collect_all("1h")
        dxy = enc_results.get("DXY", dxy)
        eurusd = enc_results.get("EURUSD", eurusd)
    
    # Process into arrays (last 300 points)
    for symbol, df in [("DXY", dxy), ("EURUSD", eurusd)]:
        if df is not None:
            # Get last 300 points (pad if needed)
            prices = df["price"].values
            if len(prices) < 300:
                prices = np.pad(prices, (300 - len(prices), 0), mode="edge")
            else:
                prices = prices[-300:]
            
            forex_data[symbol] = prices
            print(f"✓ Loaded {symbol} data (latest: {df['timestamp'].iloc[-1]})")


# Try importing RecurrentPPO
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup."""
    global model
    
    # Load Model (Try RecurrentPPO, then SAC)
    try:
        model = None
        if RecurrentPPO:
            try:
                model = RecurrentPPO.load(str(MODEL_PATH))
                print(f"✓ RecurrentPPO loaded from {MODEL_PATH}")
            except Exception:
                 pass # Fallback to standard SAC
        
        if model is None:
            model = SAC.load(str(MODEL_PATH))
            print(f"✓ SAC model loaded from {MODEL_PATH}")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model = None
    
    # Load Forex Data
    await update_forex_data()


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "forex_loaded": list(forex_data.keys())
    }


@app.post("/predict/simple", response_model=VariablePredictionResponse)
async def predict_simple(request: SimplePredictionRequest, background_tasks: BackgroundTasks):
    """
    Predict using Multi-Asset SAC model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Process BTC Data (300)
    if len(request.priceHistory) < 10:
        raise HTTPException(400, "Insufficient price history")
    
    prices_raw = [p["price"] for p in request.priceHistory[-300:]]
    while len(prices_raw) < 300:
        prices_raw.insert(0, prices_raw[0])
    
    candle_open = request.candleOpen
    btc_prices = [(p / candle_open - 1.0) * 100 for p in prices_raw]
    
    # Dummy volumes (1.0) as we don't have full history from simple request
    btc_volumes = [1.0] * 300
    
    # 2. Process Forex Data (Features)
    obs_parts = [
        np.array(btc_prices, dtype=np.float32),
        np.array(btc_volumes, dtype=np.float32),
    ]
    
    # Calculate Distilled Features (1h and 4h returns)
    # We maintain recent history in forex_data
    for symbol in ["DXY", "EURUSD"]:
        history = forex_data.get(symbol, np.array([]))
        
        # Default to 0 if insufficient data
        ret_1h = 0.0
        ret_4h = 0.0
        
        if len(history) >= 5: # Need at least 5 hours for 4h lag
            # history is daily sequence of prices
            current = history[-1]
            if current > 0:
                # 1h return: current vs lag 1
                prev_1h = history[-2]
                if prev_1h > 0:
                    ret_1h = (current / prev_1h - 1.0) * 100
                
                # 4h return: current vs lag 4
                prev_4h = history[-5]
                if prev_4h > 0:
                    ret_4h = (current / prev_4h - 1.0) * 100
        
        obs_parts.append(np.array([ret_1h, ret_4h], dtype=np.float32))
    
    # 3. Context Features (5)
    current_vs_open = (prices_raw[-1] / candle_open - 1.0) * 100
    time_remaining = max(0, min(1, request.timeRemainingMs / 900_000))
    
    now = datetime.now(timezone.utc)
    hour_sin = np.sin(2 * np.pi * now.hour / 24)
    hour_cos = np.cos(2 * np.pi * now.hour / 24)
    
    # Calculate ATR (Volatility) to match training
    # Training: mean of abs(diff) over last 14 steps (1.4s), normalized by open
    if len(prices_raw) > 1:
        changes = np.abs(np.diff(prices_raw[-20:])) # Use last 2 seconds
        atr_val = np.mean(changes) if len(changes) > 0 else 0.0
        atr = (atr_val / (candle_open + 1e-8)) * 100
    else:
        atr = 0.0
    
    obs_parts.append(np.array([
        current_vs_open,
        time_remaining,
        hour_sin,
        hour_cos,
        atr
    ], dtype=np.float32))
    
    # Log observation shape
    # 300 + 300 + 2 + 2 + 5 = 609
    obs = np.concatenate(obs_parts)
    obs = np.clip(obs, -100.0, 100.0)
    
    # 4. Predict
    # Action: [direction (-1 to 1), position_size (0 to 1)]
    action, _ = model.predict(obs, deterministic=True)
    
    direction_val = float(action[0])
    size_val = float(action[1])
    
    # Map direction to class
    action_label = "WAIT"
    if abs(direction_val) > 0.1:  # Threshold
        if direction_val > 0:
            action_label = "BET_UP"
        else:
            action_label = "BET_DOWN"
    
    # Scale position size (model outputs 0-1, max position is 50%)
    effective_size = size_val * 0.5  # Max 50%
    
    # Refresh forex data every hour (randomly spread)
    if np.random.random() < 0.01:
        background_tasks.add_task(update_forex_data)
        
    return VariablePredictionResponse(
        action=action_label,
        confidence=size_val,  # 0.0 - 1.0 raw confidence
        position_size=effective_size, # Actual % to bet
        model_loaded=True,
        forex_freshness="live"
    )
