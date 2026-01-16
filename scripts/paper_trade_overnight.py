#!/usr/bin/env python3
"""
Overnight Paper Trading Test

Runs the ML candle prediction model against live Binance data
without trading real money. Tracks predictions and theoretical P&L.

Usage:
    # Start Yang API first:
    uvicorn src.api.prediction:app --host 0.0.0.0 --port 8001
    
    # Then run this script:
    python scripts/paper_trade_overnight.py
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
import websockets
import numpy as np


@dataclass
class Trade:
    """Record of a single prediction."""
    timestamp: str
    candle_start: datetime
    candle_end: datetime
    prediction: str  # "UP" or "DOWN"
    open_price: float
    close_price: float
    actual_direction: str  # "UP" or "DOWN"
    correct: bool
    pnl_pct: float  # Theoretical P&L percentage


@dataclass
class SessionStats:
    """Accumulated session statistics."""
    start_time: str = ""
    total_candles: int = 0
    predictions_made: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    total_pnl_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    current_streak: int = 0
    best_streak: int = 0
    trades: list = field(default_factory=list)
    
    def update(self, trade: Trade):
        self.total_candles += 1
        if trade.prediction != "WAIT":
            self.predictions_made += 1
            if trade.correct:
                self.correct_predictions += 1
                self.current_streak += 1
                self.best_streak = max(self.best_streak, self.current_streak)
            else:
                self.current_streak = 0
            
            self.total_pnl_pct += trade.pnl_pct
            self.best_trade_pct = max(self.best_trade_pct, trade.pnl_pct)
            self.worst_trade_pct = min(self.worst_trade_pct, trade.pnl_pct)
        
        if self.predictions_made > 0:
            self.accuracy = self.correct_predictions / self.predictions_made
        
        self.trades.append(asdict(trade))


class PaperTrader:
    """Paper trading system using live Binance data."""
    
    def __init__(self, ml_api_url: str = "http://localhost:8001"):
        self.ml_api_url = ml_api_url
        self.stats = SessionStats(start_time=datetime.now(timezone.utc).isoformat())
        self.price_history: list[dict] = []
        self.current_candle_start: datetime | None = None
        self.current_candle_open: float = 0.0
        self.candle_minutes: int = 15
        self.log_file = Path("logs/paper_trading.json")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    async def get_ml_prediction(self, candle_open: float, time_remaining_ms: int) -> str:
        """Get prediction from ML API."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ml_api_url}/predict/simple",
                    json={
                        "priceHistory": self.price_history[-300:],
                        "candleOpen": candle_open,
                        "timeRemainingMs": time_remaining_ms,
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("action", "WAIT")
        except Exception as e:
            print(f"ML API error: {e}")
        
        return "WAIT"
    
    def get_candle_boundaries(self, timestamp: datetime) -> tuple[datetime, datetime]:
        """Get start and end time of the current 15-minute candle."""
        # Align to 15-minute boundaries
        minute = timestamp.minute
        aligned_minute = (minute // self.candle_minutes) * self.candle_minutes
        candle_start = timestamp.replace(minute=aligned_minute, second=0, microsecond=0)
        candle_end = candle_start.replace(minute=aligned_minute + self.candle_minutes)
        return candle_start, candle_end
    
    def print_status(self):
        """Print current session status."""
        stats = self.stats
        print("\n" + "=" * 60)
        print(f"📊 PAPER TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        print(f"  Running since: {stats.start_time[:19]}")
        print(f"  Candles:       {stats.total_candles}")
        print(f"  Predictions:   {stats.predictions_made}")
        print(f"  Correct:       {stats.correct_predictions}")
        print(f"  Accuracy:      {stats.accuracy:.1%}")
        print(f"  Total P&L:     {stats.total_pnl_pct:+.3f}%")
        print(f"  Best trade:    {stats.best_trade_pct:+.3f}%")
        print(f"  Worst trade:   {stats.worst_trade_pct:+.3f}%")
        print(f"  Current streak: {stats.current_streak}")
        print(f"  Best streak:   {stats.best_streak}")
        print("=" * 60)
    
    def save_stats(self):
        """Save stats to file."""
        with open(self.log_file, "w") as f:
            json.dump(asdict(self.stats), f, indent=2, default=str)
    
    async def run(self):
        """Main paper trading loop."""
        print("🚀 Starting Paper Trading Session")
        print(f"   ML API: {self.ml_api_url}")
        print(f"   Log file: {self.log_file}")
        print("   Press Ctrl+C to stop\n")
        
        last_candle_start: datetime | None = None
        candle_open_price: float = 0.0
        prediction: str = "WAIT"
        prediction_made_at: datetime | None = None
        
        uri = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
        
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    print("✅ Connected to Binance WebSocket")
                    
                    async for message in ws:
                        data = json.loads(message)
                        price = float(data["p"])
                        timestamp = datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc)
                        
                        # Add to price history
                        self.price_history.append({"price": price, "timestamp": data["T"]})
                        if len(self.price_history) > 300:
                            self.price_history = self.price_history[-300:]
                        
                        # Get candle boundaries
                        candle_start, candle_end = self.get_candle_boundaries(timestamp)
                        
                        # New candle started?
                        if last_candle_start is None or candle_start > last_candle_start:
                            # Evaluate previous candle if we made a prediction
                            if last_candle_start is not None and prediction != "WAIT":
                                actual_direction = "UP" if price > candle_open_price else "DOWN"
                                correct = (prediction == actual_direction)
                                pnl_pct = abs(price - candle_open_price) / candle_open_price * 100
                                if not correct:
                                    pnl_pct = -pnl_pct
                                
                                trade = Trade(
                                    timestamp=timestamp.isoformat(),
                                    candle_start=last_candle_start,
                                    candle_end=candle_start,
                                    prediction=prediction,
                                    open_price=candle_open_price,
                                    close_price=price,
                                    actual_direction=actual_direction,
                                    correct=correct,
                                    pnl_pct=pnl_pct,
                                )
                                self.stats.update(trade)
                                
                                emoji = "✅" if correct else "❌"
                                print(f"{emoji} Candle closed: Predicted {prediction}, Actual {actual_direction}, P&L: {pnl_pct:+.3f}%")
                                
                                self.save_stats()
                                self.print_status()
                            
                            # Start new candle
                            last_candle_start = candle_start
                            candle_open_price = price
                            prediction = "WAIT"
                            prediction_made_at = None
                            print(f"\n🕐 New candle started at {candle_start.strftime('%H:%M')}")
                        
                        # Get prediction if we haven't made one yet
                        if prediction == "WAIT" and len(self.price_history) >= 50:
                            time_remaining_ms = int((candle_end - timestamp).total_seconds() * 1000)
                            
                            # Only predict in first half of candle (gives time to act)
                            if time_remaining_ms > self.candle_minutes * 60 * 500:  # > 50% remaining
                                prediction = await self.get_ml_prediction(candle_open_price, time_remaining_ms)
                                if prediction != "WAIT":
                                    prediction_made_at = timestamp
                                    print(f"🤖 ML Prediction: {prediction} (at {timestamp.strftime('%H:%M:%S')})")
                        
            except websockets.exceptions.ConnectionClosed:
                print("⚠️ WebSocket disconnected, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paper trading with ML predictions")
    parser.add_argument("--ml-api", default="http://localhost:8001", help="ML API URL")
    args = parser.parse_args()
    
    trader = PaperTrader(ml_api_url=args.ml_api)
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping paper trading...")
        trader.print_status()
        trader.save_stats()
        print(f"\n📁 Results saved to: {trader.log_file}")


if __name__ == "__main__":
    asyncio.run(main())
