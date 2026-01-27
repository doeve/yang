#!/usr/bin/env python3
"""
Live Paper Trading for Attention-Based Market Predictor.

Uses REAL LIVE data from:
1. Binance (BTC Prices)
2. Polymarket (YES/NO Token Prices for current 15m market)

This script:
- Identifies the current active 15m BTC Up/Down market on Polymarket
- Streams live BTC prices from Binance
- Streams live YES/NO token prices from Polymarket CLOB
- Generates KAMA Spectrum and Context features
- Runs the AttentionMarketPredictor model
- Executes paper trades based on model probability (P(YES))
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import deque
from typing import Optional, Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import httpx
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False
    httpx_socks = None

import torch
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

from src.models.attention_market_predictor import (
    load_attention_market_predictor,
    AttentionMarketPredictorModel,
    AttentionMarketPredictorConfig,
)
from src.data.kama_features import create_kama_feature_builder
from src.data.enhanced_features import create_enhanced_feature_builder

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("paper_trade")
console = Console()

# Constants
BINANCE_REST = "https://api.binance.com/api/v3"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"
SOCKS5_PROXY_URL = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:1080")

class LivePolymarketStreamer:
    """
    Streams live market data from Polymarket for the current 15m BTC market.
    """
    
    def __init__(self):
        self.current_market: Optional[Dict] = None
        self.yes_price_history: deque = deque(maxlen=3000)
        self.no_price_history: deque = deque(maxlen=3000)
        self.last_fetch_time = 0
        self.market_slug = None
        self.yes_token_id = None
        self.no_token_id = None
        
        # Determine if we should use proxy
        self.use_proxy = SOCKS_AVAILABLE and "SOCKS5_PROXY" in os.environ
        if self.use_proxy:
            logger.info(f"Using SOCKS5 proxy: {SOCKS5_PROXY_URL}")

    def create_client(self) -> httpx.AsyncClient:
        """Create an HTTP client with optional proxy support"""
        if self.use_proxy:
            transport = httpx_socks.AsyncProxyTransport.from_url(SOCKS5_PROXY_URL)
            return httpx.AsyncClient(transport=transport, verify=False, timeout=10.0)
        else:
            return httpx.AsyncClient(timeout=10.0)

        
    async def find_current_market(self, client: httpx.AsyncClient) -> bool:
        """
        Find the current active 15m BTC market.
        Logic: Look for market ending at the next 15m boundary.
        """
        now = datetime.now(timezone.utc)
        
        # Calculate next 15m boundary
        minutes = now.minute
        next_boundary_min = ((minutes // 15) + 1) * 15
        if next_boundary_min == 60:
            next_boundary = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_boundary = now.replace(minute=next_boundary_min, second=0, microsecond=0)
            
        target_ts = int(next_boundary.timestamp())
        
        # Try to construct slug directly? Or search?
        # Pattern: btc-updown-15m-{timestamp}
        # Note: Polymarket slugs might use specific formatting.
        # Let's search via Gamma API if possible, or try direct slug.
        
        # We can try to guess the slug
        slug = f"btc-updown-15m-{target_ts}"
        
        try:
            resp = await client.get(f"{POLYMARKET_GAMMA}/events/slug/{slug}")
            if resp.status_code == 200:
                data = resp.json()
                markets = data.get("markets", [])
                if markets:
                    self.current_market = markets[0]
                    self.market_slug = slug
                    
                    # Extract token IDs
                    clob_tokens = self.current_market.get("clobTokenIds", [])
                    if isinstance(clob_tokens, str):
                        clob_tokens = json.loads(clob_tokens)
                        
                    if len(clob_tokens) >= 2:
                        self.yes_token_id = clob_tokens[0]
                        self.no_token_id = clob_tokens[1]
                        return True
            
            # If explicit slug fails, maybe try searching events?
            # For now, let's assume the slug pattern holds or we fail.
            # print(f"Could not find market for slug: {slug}")
            return False
            
        except Exception as e:
            logger.error(f"Error finding market: {e}")
            return False

    async def update_prices(self, client: httpx.AsyncClient):
        """Fetch latest prices for the current market."""
        if not self.yes_token_id:
            return

        # Fetch recent trades or orderbook mid-price?
        # Using prices-history for consistency with training
        try:
            # Fetch YES history (last few minutes to append)
            resp = await client.get(
                f"{POLYMARKET_CLOB}/prices-history",
                params={
                    "market": self.yes_token_id,
                    "fidelity": 1,
                    "interval": "1h" # Get last hour
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                history = data.get("history", [])
                
                # We need to act like a stream. 
                # Ideally we only add NEW data points.
                # For simplicity in this loop, we replace the history with the fetched batch
                # but in a real high-freq setup we'd use websocket.
                
                # Clear and refill for robustness (simplest for polling)
                self.yes_price_history.clear()
                self.no_price_history.clear()
                
                for point in history:
                    p = float(point['p'])
                    self.yes_price_history.append(p)
                    self.no_price_history.append(1.0 - p) # Approx NO price
            
            # Also get CURRENT price from orderbook for the very latest tick
            # (History API might be delayed)
            # await self._fetch_live_tick(client)
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")

    def get_yes_prices(self) -> np.ndarray:
        return np.array(self.yes_price_history)

    def get_no_prices(self) -> np.ndarray:
        return np.array(self.no_price_history)


class LivePaperTrader:
    def __init__(self, model_dir: str, balance: float):
        self.model_dir = model_dir
        self.balance = balance
        self.initial_balance = balance
        self.model = None
        self.kama_builder = create_kama_feature_builder()
        self.enhanced_builder = create_enhanced_feature_builder()
        
        self.poly_streamer = LivePolymarketStreamer()
        self.btc_history: deque = deque(maxlen=3000)
        self.btc_start_price = None # Open price of candle
        
        self.current_position = None # {direction: 'YES'/'NO', size: float, entry_price: float}
        self.trades = []
        self.candle_start_time = None
        self.candle_end_time = None
        
        # Stats
        self.wins = 0
        self.losses = 0

    async def load_model(self):
        console.print(f"[blue]Loading model from {self.model_dir}...[/blue]")
        self.model = load_attention_market_predictor(self.model_dir)
        console.print("[green]Model loaded![/green]")

    def start_new_candle(self):
        now = datetime.now(timezone.utc)
        minutes = now.minute
        # 15m boundary
        slot = (minutes // 15) * 15
        self.candle_start_time = now.replace(minute=slot, second=0, microsecond=0)
        self.candle_end_time = self.candle_start_time + timedelta(minutes=15)
        
        self.btc_start_price = self.btc_history[-1] if self.btc_history else 0.0
        self.current_position = None
        
        console.print(f"\n[bold cyan]NEW CANDLE: {self.candle_start_time.strftime('%H:%M')} - {self.candle_end_time.strftime('%H:%M')}[/bold cyan]")
        console.print(f"BTC Open: {self.btc_start_price}")

    async def run(self):
        await self.load_model()
        
        # Use a persistent client for Polymarket to keep connection open (maybe?)
        # Actually it's better to recreate or manage properly.
        # But for now let's use the helper to create one.
        
        # We need two clients? One for Binance (no proxy usually) and one for Poly (proxy).
        poly_client = self.poly_streamer.create_client()
        binance_client = httpx.AsyncClient(timeout=10.0)
        
        try:
            console.print("[blue]Finding current market...[/blue]")
            found = await self.poly_streamer.find_current_market(poly_client)
            if not found:
                console.print("[red]Could not find active Polymarket market. Waiting...[/red]")
            else:
                console.print(f"[green]Tracking market: {self.poly_streamer.market_slug}[/green]")
            
            # Initial fetch
            await self.poly_streamer.update_prices(poly_client)
            
            # BTC History fill (short)
            try:
                resp = await binance_client.get(f"{BINANCE_REST}/klines", params={"symbol": "BTCUSDT", "interval": "1m", "limit": 60})
                if resp.status_code == 200:
                    klines = resp.json()
                    for k in klines:
                        self.btc_history.append(float(k[4])) # Close
            except Exception as e:
                logger.error(f"Binance init error: {e}")

            # Start loop
            msg_counter = 0
            while True:
                now = datetime.now(timezone.utc)
                
                # Check candle boundary
                if self.candle_end_time and now >= self.candle_end_time:
                    self.settle_candle()
                    self.start_new_candle()
                    # Refresh Polymarket for new candle
                    await self.poly_streamer.find_current_market(poly_client)
                
                if not self.candle_start_time:
                    self.start_new_candle()

                # Update Data
                try:
                    # BTC
                    btc_resp = await binance_client.get(f"{BINANCE_REST}/ticker/price", params={"symbol": "BTCUSDT"})
                    btc_price = float(btc_resp.json()['price'])
                    self.btc_history.append(btc_price)
                    
                    # Polymarket
                    await self.poly_streamer.update_prices(poly_client)
                except Exception as e:
                    logger.error(f"Fetch error: {e}")
                    await asyncio.sleep(1)
                    continue


                # Run Inference
                # 1. Build Features
                yes_prices = self.poly_streamer.get_yes_prices()
                no_prices = self.poly_streamer.get_no_prices()
                
                if len(yes_prices) < 20: # Need some history
                    console.print("[yellow]Waiting for more price history...[/yellow]", end="\r")
                    await asyncio.sleep(2)
                    continue
                
                btc_arr = np.array(self.btc_history)
                
                total_seconds = (self.candle_end_time - self.candle_start_time).total_seconds()
                elapsed = (now - self.candle_start_time).total_seconds()
                time_remaining = max(0.0, 1.0 - (elapsed / total_seconds))
                
                # KAMA Spectrum
                kama_spectrum, context_features = self.kama_builder.compute_features(
                    yes_prices=yes_prices,
                    time_remaining=time_remaining
                )
                
                # Base Features
                base_features = self.enhanced_builder.compute_features(
                    yes_prices=yes_prices,
                    no_prices=no_prices,
                    time_remaining=time_remaining,
                    btc_prices=btc_arr,
                    btc_open=self.btc_start_price
                )
                
                # 2. Predict
                with torch.no_grad():
                    device = next(self.model.parameters()).device
                    # Batch dim
                    kama_t = torch.FloatTensor(kama_spectrum).unsqueeze(0).to(device)
                    context_t = torch.FloatTensor(context_features).unsqueeze(0).to(device)
                    base_t = torch.FloatTensor(base_features).unsqueeze(0).to(device)
                    
                    outputs = self.model(kama_t, context_t, base_t)
                    prob_yes = outputs['probability'].item()
                    attn_weights = outputs['attention_weights'].cpu().numpy()[0]

                # 3. Trade Logic
                self.execute_trade_logic(prob_yes, yes_prices[-1], time_remaining)

                # UI Update (every few seconds)
                if msg_counter % 5 == 0:
                    self.update_ui(prob_yes, yes_prices[-1], attn_weights, btc_price)
                msg_counter += 1
                
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Run loop error: {e}")
            raise

    def execute_trade_logic(self, prob_yes: float, current_price: float, time_remaining: float):
        if self.current_position:
            return # Already in a position
            
        # Simple threshold strategy driven by model confidence
        # Buy YES if Prob > 0.7 and Price < 0.6 (edge)
        # Buy NO if Prob < 0.3 and Price > 0.4
        
        # Don't trade too late
        if time_remaining < 0.1:
            return

        size = 100.0 # Fixed bet size
        
        if prob_yes > 0.70 and current_price < 0.8:
            self.current_position = {
                "side": "YES",
                "entry_price": current_price,
                "size": size,
                "entry_prop": prob_yes
            }
            console.print(f"[bold green] >>> BUY YES @ {current_price:.2f} (Prob: {prob_yes:.2f})[/bold green]")
            
        elif prob_yes < 0.30 and current_price > 0.2:
             self.current_position = {
                "side": "NO",
                "entry_price": 1.0 - current_price, # NO price
                "size": size,
                "entry_prop": prob_yes
            }
             console.print(f"[bold red] >>> BUY NO @ {1.0 - current_price:.2f} (Prob: {prob_yes:.2f})[/bold red]")

    def settle_candle(self):
        if not self.current_position:
            console.print("Candle ended. No position taken.")
            return
            
        # Determine outcome based on BTC price vs Strike
        # Wait, strictly speaking we should check Polymarket resolution.
        # But locally we can approximate with BTC price.
        btc_close = self.btc_history[-1]
        strike = self.btc_start_price # Approx
        
        outcome_yes = btc_close > strike
        
        # Calculate PnL
        side = self.current_position['side']
        entry = self.current_position['entry_price']
        size = self.current_position['size']
        
        if (side == "YES" and outcome_yes) or (side == "NO" and not outcome_yes):
            # Win
            profit = (1.0 - entry) / entry * size # Simple ROI calc
            self.balance += profit
            self.wins += 1
            console.print(f"[bold green]WIN! PnL: +${profit:.2f}[/bold green]")
        else:
            # Loss
            loss = size
            self.balance -= loss
            self.losses += 1
            console.print(f"[bold red]LOSS. PnL: -${loss:.2f}[/bold red]")
            
        self.current_position = None
        console.print(f"Current Balance: ${self.balance:.2f}")

    def update_ui(self, prob: float, price: float, attn: np.ndarray, btc: float):
        # We can make a nice rich table here
        console.clear() 
        
        table = Table(title="Attention Market Predictor - Live")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("BTC Price", f"${btc:,.2f}")
        table.add_row("Poly YES Price", f"{price:.3f}")
        table.add_row("Model P(YES)", f"{prob:.3f}")
        
        if self.current_position:
            table.add_row("Position", f"{self.current_position['side']} @ {self.current_position['entry_price']:.2f}")
        else:
            table.add_row("Position", "None")
            
        table.add_row("Balance", f"${self.balance:,.2f}")
        table.add_row("W/L", f"{self.wins}/{self.losses}")
        
        # Show top attended KAMA
        # Just simple argmax
        top_kama_idx = np.argmax(attn)
        table.add_row("Top Attention", f"KAMA Idx {top_kama_idx} ({attn[top_kama_idx]:.3f})")
        
        console.print(table)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    args = parser.parse_args()
    
    trader = LivePaperTrader(
        model_dir=args.model,
        balance=1000.0
    )
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup would go here if we kept clients in 'trader'
        pass

if __name__ == "__main__":
    asyncio.run(main())
