#!/usr/bin/env python3
"""
Attention Market Predictor - Paper Trading Script.

Refactored to match `src/paper_trade_unified.py` design patterns:
1. Robust Class-based structure (State, Config, Trader)
2. Rich UI with Layouts
3. Default SOCKS5 Proxy support
4. Unified error handling and logging
"""

import os
import asyncio
import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import httpx
import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

# Try importing SOCKS support
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False
    httpx_socks = None

# Internal imports
from src.models.attention_market_predictor import (
    load_attention_market_predictor,
    AttentionMarketPredictorModel,
)
from src.data.kama_features import create_kama_feature_builder
from src.data.enhanced_features import create_enhanced_feature_builder

# Logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("paper_trade")
console = Console()

# Constants
BINANCE_REST = "https://api.binance.com/api/v3"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"
DEFAULT_PROXY = "socks5://127.0.0.1:1080"


@dataclass
class AttentionTradingState:
    """Tracks the state of the trading session."""
    balance: float = 1000.0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    
    # Current Market
    current_candle_ts: Optional[int] = None
    market_slug: Optional[str] = None
    active_yes_id: Optional[str] = None
    active_no_id: Optional[str] = None
    
    # Prices
    btc_current: float = 0.0
    btc_open: float = 0.0
    yes_price: float = 0.5
    no_price: float = 0.5
    
    # History
    yes_price_history: List[float] = field(default_factory=list)
    no_price_history: List[float] = field(default_factory=list)
    btc_price_history: List[float] = field(default_factory=list)
    
    # Position
    position_side: Optional[str] = None  # "YES" or "NO"
    position_size: float = 0.0
    entry_price: float = 0.0
    
    # Model Output
    last_prob: float = 0.5
    last_attn: Optional[np.ndarray] = None


@dataclass
class AttentionPaperTradeConfig:
    """Configuration parameters."""
    model_path: str
    initial_balance: float = 1000.0
    
    # Strategy
    min_confidence_yes: float = 0.70
    max_confidence_no: float = 0.30
    bet_size: float = 100.0  # Fixed bet size
    
    # Network
    proxy_url: str = DEFAULT_PROXY
    refresh_rate: float = 1.0


class AttentionPaperTrader:
    """
    Paper trader for Attention Market Predictor.
    Matches the design of UnifiedPaperTrader.
    """
    
    def __init__(self, config: AttentionPaperTradeConfig):
        self.config = config
        self.state = AttentionTradingState(balance=config.initial_balance)
        
        # Feature Builders
        self.kama_builder = create_kama_feature_builder()
        self.enhanced_builder = create_enhanced_feature_builder()
        
        # Model
        self.model: Optional[AttentionMarketPredictorModel] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Clients
        self.poly_client: Optional[httpx.AsyncClient] = None
        self.binance_client: Optional[httpx.AsyncClient] = None
        
        self._running = False
        
    async def _setup_clients(self):
        """Initialize HTTP clients with Proxy support."""
        self.binance_client = httpx.AsyncClient(timeout=10.0)
        
        # Setup Proxy for Polymarket
        if SOCKS_AVAILABLE:
            console.print(f"[blue]Using SOCKS5 proxy: {self.config.proxy_url}[/blue]")
            transport = httpx_socks.AsyncProxyTransport.from_url(self.config.proxy_url)
            self.poly_client = httpx.AsyncClient(transport=transport, verify=False, timeout=10.0)
        else:
            console.print("[yellow]SOCKS5 not available. using direct connection.[/yellow]")
            self.poly_client = httpx.AsyncClient(timeout=10.0)
            
    async def load_model(self):
        """Load the PyTorch model."""
        console.print(f"[blue]Loading model from {self.config.model_path}...[/blue]")
        self.model = load_attention_market_predictor(self.config.model_path)
        # Ensure model is on correct device
        # (load function might put it on CPU by default or Auto)
        self.device = next(self.model.parameters()).device
        console.print(f"[green]Model loaded on {self.device}![/green]")

    def get_time_remaining(self) -> float:
        """Calculate fraction of 15m candle remaining."""
        if not self.state.current_candle_ts:
            return 1.0
            
        start_ts = self.state.current_candle_ts
        end_ts = start_ts + 900  # 15 mins
        now_ts = datetime.now(timezone.utc).timestamp()
        
        remaining = end_ts - now_ts
        return max(0.0, min(1.0, remaining / 900.0))

    async def fetch_active_market(self) -> bool:
        """Find the current active 15m BTC market."""
        # Polymarket 15m market slugs use the candle START timestamp
        # e.g. btc-updown-15m-{start_ts}
        now_ts = int(datetime.now(timezone.utc).timestamp())
        target_ts = (now_ts // 900) * 900
        
        # If we already have this market, verify it's still valid
        if self.state.current_candle_ts == target_ts and self.state.active_yes_id:
            return True
        
        # New candle? Settle previous if needed
        if self.state.current_candle_ts is not None and self.state.current_candle_ts != target_ts:
            self.settle_position()
        
        slug = f"btc-updown-15m-{target_ts}"
        
        try:
            resp = await self.poly_client.get(f"{POLYMARKET_GAMMA}/events/slug/{slug}")
            if resp.status_code == 200:
                data = resp.json()
                markets = data.get("markets", [])
                if markets:
                    market = markets[0]
                    clob_tokens = market.get("clobTokenIds", [])
                    if isinstance(clob_tokens, str):
                        clob_tokens = json.loads(clob_tokens)
                        
                    if len(clob_tokens) >= 2:
                        self.state.market_slug = slug
                        self.state.current_candle_ts = target_ts
                        self.state.active_yes_id = clob_tokens[0]
                        self.state.active_no_id = clob_tokens[1]
                        
                        # Reset history for new market
                        self.state.yes_price_history = []
                        self.state.no_price_history = []
                        self.state.btc_price_history = []
                        self.state.btc_open = 0.0 # Will fill on next fetch
                        
                        return True
        except Exception as e:
            logger.debug(f"Market find error: {e}")
            
        return False

    async def fetch_prices(self):
        """Fetch BTC and Polymarket prices."""
        # BTC
        try:
            resp = await self.binance_client.get(f"{BINANCE_REST}/ticker/price", params={"symbol": "BTCUSDT"})
            price = float(resp.json()['price'])
            self.state.btc_current = price
            self.state.btc_price_history.append(price)
            if self.state.btc_open == 0.0:
                self.state.btc_open = price
        except Exception as e:
            logger.error(f"BTC fetch error: {e}")
            
        # Polymarket
        if self.state.active_yes_id:
            try:
                # Use Prices History for consistency
                # We need enough history for KAMA (at least 20 points, so ~2 hours of 1m data to be safe)
                now_ts = int(datetime.now(timezone.utc).timestamp())
                start_ts = now_ts - (3600 * 2) # 2 hours ago
                
                url = f"{POLYMARKET_CLOB}/prices-history"
                params = {
                    "market": self.state.active_yes_id, 
                    "fidelity": 1,
                    "startTs": start_ts,
                    "endTs": now_ts
                }
                
                resp = await self.poly_client.get(url, params=params)
                
                if resp.status_code == 200:
                    data = resp.json()
                    history = data.get("history", [])
                    
                    if not history:
                        # Try orderbook fallback
                        await self.fetch_orderbook_price()
                    else:
                        # Extract latest
                        latest = float(history[-1]['p'])
                        self.state.yes_price = latest
                        self.state.no_price = 1.0 - latest
                        
                        # Use the fetched history directly instead of just appending
                        # This handles the "initial load" better and keeps window rolling
                        self.state.yes_price_history = [float(p['p']) for p in history]
                        self.state.no_price_history = [1.0 - p for p in self.state.yes_price_history]
                else:
                    logger.warning(f"History fetch failed: {resp.status_code} {resp.text}")
                    # Fallback
                    await self.fetch_orderbook_price()
                    
            except Exception as e:
                logger.error(f"Poly fetch error: {e}")
                
    async def fetch_orderbook_price(self):
        """Fallback to orderbook if history is empty."""
        try:
            resp = await self.poly_client.get(f"{POLYMARKET_CLOB}/book", params={"token_id": self.state.active_yes_id})
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                
                if bids and asks:
                     mid = (float(bids[0]['price']) + float(asks[0]['price'])) / 2
                     self.state.yes_price = mid
                     self.state.no_price = 1.0 - mid
                     # Stick it in history so model has *something*
                     self.state.yes_price_history.append(mid)
        except Exception as e:
            logger.error(f"OB fetch error: {e}")

    def get_model_output(self):
        """Run model inference."""
        if len(self.state.yes_price_history) < 20:
            return  # Not enough data
            
        time_rem = self.get_time_remaining()
        
        # Build Features
        kama_spectrum, context_features = self.kama_builder.compute_features(
            yes_prices=np.array(self.state.yes_price_history),
            time_remaining=time_rem
        )
        
        base_features = self.enhanced_builder.compute_features(
            yes_prices=np.array(self.state.yes_price_history),
            no_prices=np.array(self.state.no_price_history),
            time_remaining=time_rem,
            btc_prices=np.array(self.state.btc_price_history),
            btc_open=self.state.btc_open
        )
        
        # Inference
        with torch.no_grad():
            kama_t = torch.FloatTensor(kama_spectrum).unsqueeze(0).to(self.device)
            context_t = torch.FloatTensor(context_features).unsqueeze(0).to(self.device)
            base_t = torch.FloatTensor(base_features).unsqueeze(0).to(self.device)
            
            outputs = self.model(kama_t, context_t, base_t)
            
            self.state.last_prob = outputs['probability'].item()
            self.state.last_attn = outputs['attention_weights'].cpu().numpy()[0]

    def execute_trading_logic(self):
        """Execute entries based on thresholds."""
        if self.state.position_side:
            return # Hold position until settlement (binary)
            
        time_rem = self.get_time_remaining()
        if time_rem < 0.05: # Don't trade last 5%
            return
            
        prob = self.state.last_prob
        price = self.state.yes_price
        
        if prob > self.config.min_confidence_yes and price < 0.85:
            self.enter_position("YES", price)
        elif prob < self.config.max_confidence_no and price > 0.15:
            # Short YES = Buy NO
            # Polymarket 'no' price is approx 1 - yes
            self.enter_position("NO", 1.0 - price)

    def enter_position(self, side: str, price: float):
        self.state.position_side = side
        self.state.position_size = self.config.bet_size
        self.state.entry_price = price
        
        console.print(f"[bold green] >>> ENTER {side} @ {price:.3f} (Prob: {self.state.last_prob:.2f})[/bold green]")

    def settle_position(self):
        """Settle PnL at end of candle."""
        if not self.state.position_side:
             return
             
        # Determine outcome (Simulation using BTC price)
        # In real Unified trader we check resolution, but here we estimate
        won = False
        if self.state.btc_current > self.state.btc_open:
            outcome = "YES"
        else:
            outcome = "NO"
            
        if self.state.position_side == outcome:
            won = True
            
        # PnL logic
        if won:
            # Profit = (1 - entry) / entry * size
            profit = (1.0 - self.state.entry_price) / self.state.entry_price * self.state.position_size
            self.state.balance += profit
            self.state.total_pnl += profit
            self.state.wins += 1
            console.print(f"[bold green]WIN! +${profit:.2f}[/bold green]")
        else:
            loss = self.state.position_size
            self.state.balance -= loss
            self.state.total_pnl -= loss
            self.state.losses += 1
            console.print(f"[bold red]LOSS. -${loss:.2f}[/bold red]")
            
        self.state.position_side = None

    def build_display(self) -> Layout:
        """Create Rich UI Layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(Panel(Text("Attention Market Predictor - Paper Trading", justify="center", style="bold cyan")))
        
        # Main: Grid
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            self._build_market_panel(),
            self._build_stats_panel()
        )
        layout["main"].update(grid)
        
        # Footer: Attention
        layout["footer"].update(self._build_attn_panel())
        
        return layout

    def _build_market_panel(self) -> Panel:
        table = Table(box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        
        time_rem = self.get_time_remaining()
        
        table.add_row("Market", self.state.market_slug or "Searching...")
        table.add_row("Time Rem", f"{time_rem:.1%}")
        table.add_row("BTC Price", f"${self.state.btc_current:,.2f}")
        table.add_row("YES Price", f"{self.state.yes_price:.3f}")
        table.add_row("Model Prob", f"{self.state.last_prob:.3f}")
        
        return Panel(table, title="Market Status")

    def _build_stats_panel(self) -> Panel:
        table = Table(box=None)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="bold white")
        
        table.add_row("Balance", f"${self.state.balance:,.2f}")
        table.add_row("PnL", f"${self.state.total_pnl:+.2f}", style="green" if self.state.total_pnl >= 0 else "red")
        table.add_row("Win/Loss", f"{self.state.wins}/{self.state.losses}")
        
        pos_str = "None"
        if self.state.position_side:
            pos_str = f"{self.state.position_side} @ {self.state.entry_price:.2f}"
        table.add_row("Position", pos_str)
        
        return Panel(table, title="Account")
        
    def _build_attn_panel(self) -> Panel:
        if self.state.last_attn is None:
            return Panel("No attention data")
            
        # Draw a simple bar chart string
        bars = ""
        for w in self.state.last_attn:
             bars += "█" * int(w * 10) + " "
             
        return Panel(Text(f"Attention Weights: {bars}", style="yellow"), title="Model Internals")

    async def run(self):
        await self._setup_clients()
        await self.load_model()
        
        console.print("[yellow]Starting Loop...[/yellow]")
        
        with Live(self.build_display(), refresh_per_second=2) as live:
            self._running = True
            
            while self._running:
                try:
                    # 1. Market Sync
                    found = await self.fetch_active_market()
                    
                    # 2. Data Fetch
                    await self.fetch_prices()
                    
                    # 3. Model
                    self.get_model_output()
                    
                    # 4. Act
                    self.execute_trading_logic()
                    
                    # 5. UI
                    live.update(self.build_display())
                    
                    await asyncio.sleep(self.config.refresh_rate)
                    
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    await asyncio.sleep(1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model logs/ directory")
    args = parser.parse_args()
    
    config = AttentionPaperTradeConfig(
        model_path=args.model,
    )
    
    trader = AttentionPaperTrader(config)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
