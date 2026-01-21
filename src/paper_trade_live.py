"""
Live Paper Trading for DeepLOB Model.

Simulates Polymarket 15-minute crypto markets using real-time Binance data.
Each 15-minute candle is a separate "market" where we predict Up/Down/Hold.
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class LiveTradeConfig:
    """Configuration for live paper trading."""
    
    # Trading
    initial_balance: float = 1000.0
    max_position_size: float = 0.25
    fee_percent: float = 0.1
    
    # Candle settings (Polymarket-style)
    candle_minutes: int = 15
    warmup_seconds: int = 120  # Seconds into candle before first trade
    
    # Model settings
    deep_lob_model: str = "./logs/deep_lob_balanced"
    use_sac: bool = False
    aggressive: bool = True
    
    # Data settings
    history_seconds: int = 300  # 5 min history for features
    update_interval: float = 1.0  # Seconds between updates
    
    # Display
    show_live: bool = True


@dataclass
class LiveTradeState:
    """Current state of live trading."""
    
    balance: float = 1000.0
    total_pnl: float = 0.0
    
    # Current candle
    candle_start_time: Optional[datetime] = None
    candle_open_price: Optional[float] = None
    current_position: Optional[str] = None  # "yes", "no", or None
    position_size: float = 0.0
    entry_price: float = 0.0
    
    # Statistics
    trades: List[Dict[str, Any]] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    
    # Last prediction
    last_prediction: Optional[str] = None
    last_confidence: float = 0.0
    last_probs: Dict[str, float] = field(default_factory=dict)


class BinanceLiveDataFetcher:
    """Fetch real-time BTC data from Binance."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10)
        self._last_price: Optional[float] = None
    
    async def get_current_price(self) -> float:
        """Get current BTC/USDT price."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/ticker/price",
                params={"symbol": "BTCUSDT"}
            )
            response.raise_for_status()
            self._last_price = float(response.json()["price"])
            return self._last_price
        except Exception as e:
            logger.warning(f"Error fetching price: {e}")
            return self._last_price or 0.0
    
    async def get_recent_klines(
        self,
        interval: str = "1s",
        limit: int = 300,
    ) -> pd.DataFrame:
        """Get recent klines/candles."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": limit,
                }
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"
            ])
            
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["price"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["taker_buy_volume"] = df["taker_buy_volume"].astype(float)
            df["buy_pressure"] = df["taker_buy_volume"] / (df["volume"] + 1e-10)
            
            return df[["timestamp", "price", "volume", "buy_pressure"]]
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()
    
    async def close(self):
        await self.client.aclose()


class LivePaperTrader:
    """Live paper trading simulation."""
    
    def __init__(self, config: Optional[LiveTradeConfig] = None):
        self.config = config or LiveTradeConfig()
        self.state = LiveTradeState(balance=self.config.initial_balance)
        self.data_fetcher = BinanceLiveDataFetcher()
        self.bot = None
        self._running = False
    
    def load_model(self):
        """Load the DeepLOB model."""
        from src.inference.deep_lob_inference import (
            DeepLOBTwoLayerBot,
            DeepLOBInferenceConfig,
        )
        
        inference_config = DeepLOBInferenceConfig()
        self.bot = DeepLOBTwoLayerBot(config=inference_config)
        self.bot.load_models(
            deep_lob_path=self.config.deep_lob_model,
            sac_path=None,  # Rule-based for now
        )
        console.print("[green]✓ Model loaded[/green]")
    
    def get_candle_boundaries(self) -> tuple[datetime, datetime]:
        """Get current 15-min candle start and end times."""
        now = datetime.utcnow()
        
        # Round down to nearest 15 min
        minutes = (now.minute // self.config.candle_minutes) * self.config.candle_minutes
        candle_start = now.replace(minute=minutes, second=0, microsecond=0)
        candle_end = candle_start + timedelta(minutes=self.config.candle_minutes)
        
        return candle_start, candle_end
    
    def get_time_remaining(self) -> float:
        """Get fraction of time remaining in current candle (0-1)."""
        now = datetime.utcnow()
        _, candle_end = self.get_candle_boundaries()
        
        total_seconds = self.config.candle_minutes * 60
        remaining_seconds = (candle_end - now).total_seconds()
        
        return max(0.0, min(1.0, remaining_seconds / total_seconds))
    
    async def get_prediction(self, btc_data: pd.DataFrame) -> Dict[str, Any]:
        """Get prediction from model."""
        if self.bot is None or len(btc_data) < 50:
            return {"action": "hold", "size": 0, "predicted_class": "Hold"}
        
        # Get market price (simulate Polymarket YES price based on current move)
        if self.state.candle_open_price:
            current_price = btc_data["price"].iloc[-1]
            move = (current_price - self.state.candle_open_price) / self.state.candle_open_price
            # Convert to implied probability
            market_yes_price = 0.5 + move * 100  # Scale move
            market_yes_price = np.clip(market_yes_price, 0.3, 0.7)
        else:
            market_yes_price = 0.5
        
        decision = self.bot.step(
            btc_data=btc_data,
            market_yes_price=market_yes_price,
            market_spread=0.02,
            time_remaining=self.get_time_remaining(),
        )
        
        # Apply aggressive mode
        if self.config.aggressive and decision["action"] == "hold":
            prob_up = decision.get("prob_up", 0.33)
            prob_down = decision.get("prob_down", 0.33)
            diff = prob_up - prob_down
            
            if abs(diff) > 0.03:
                decision["action"] = "buy_yes" if diff > 0 else "buy_no"
                decision["size"] = min(abs(diff) * 2, self.config.max_position_size)
        
        return decision
    
    async def handle_new_candle(self):
        """Handle transition to a new candle."""
        candle_start, _ = self.get_candle_boundaries()
        
        # Settle previous position if exists
        if self.state.current_position and self.state.candle_start_time:
            await self.settle_position()
        
        # Start new candle
        self.state.candle_start_time = candle_start
        self.state.candle_open_price = await self.data_fetcher.get_current_price()
        self.state.current_position = None
        self.state.position_size = 0.0
        
        logger.info(
            "New candle started",
            start=candle_start.isoformat(),
            open_price=self.state.candle_open_price,
        )
    
    async def settle_position(self):
        """Settle current position at candle end."""
        if not self.state.current_position or not self.state.candle_open_price:
            return
        
        current_price = await self.data_fetcher.get_current_price()
        move = (current_price - self.state.candle_open_price) / self.state.candle_open_price
        
        # Determine outcome
        if move > 0.0001:
            outcome = "Up"
        elif move < -0.0001:
            outcome = "Down"
        else:
            outcome = "Hold"
        
        # Calculate PnL
        pnl = 0.0
        position_value = self.state.position_size * self.state.balance
        
        if self.state.current_position == "yes":
            # Bet on Up
            if outcome == "Up":
                pnl = position_value * (1 - self.state.entry_price - 0.002)
                self.state.wins += 1
            else:
                pnl = -position_value * self.state.entry_price
                self.state.losses += 1
        else:  # "no"
            # Bet on Down
            if outcome == "Down":
                pnl = position_value * (self.state.entry_price - 0.002)
                self.state.wins += 1
            else:
                pnl = -position_value * (1 - self.state.entry_price)
                self.state.losses += 1
        
        # Apply fees
        pnl -= position_value * self.config.fee_percent * 0.01
        
        self.state.balance += pnl
        self.state.total_pnl += pnl
        
        trade = {
            "time": self.state.candle_start_time,
            "position": self.state.current_position,
            "size": self.state.position_size,
            "outcome": outcome,
            "pnl": pnl,
            "balance": self.state.balance,
            "prediction": self.state.last_prediction,
        }
        self.state.trades.append(trade)
        
        logger.info(
            "Position settled",
            position=self.state.current_position,
            outcome=outcome,
            pnl=f"${pnl:.2f}",
            balance=f"${self.state.balance:.2f}",
        )
    
    async def execute_trade(self, decision: Dict[str, Any]):
        """Execute a trade based on decision."""
        if self.state.current_position:
            return  # Already have position
        
        action = decision.get("action", "hold")
        if action == "hold":
            return
        
        size = decision.get("size", 0.25)
        
        self.state.current_position = "yes" if action == "buy_yes" else "no"
        self.state.position_size = size
        self.state.entry_price = 0.5  # Simplified
        self.state.last_prediction = decision.get("predicted_class", "Hold")
        self.state.last_confidence = decision.get("confidence", 0.0)
        self.state.last_probs = {
            "up": decision.get("prob_up", 0.33),
            "down": decision.get("prob_down", 0.33),
            "hold": decision.get("prob_hold", 0.34),
        }
        
        logger.info(
            "Trade executed",
            action=action,
            size=size,
            prediction=self.state.last_prediction,
        )
    
    def build_display(self) -> Panel:
        """Build rich display panel."""
        now = datetime.utcnow()
        candle_start, candle_end = self.get_candle_boundaries()
        time_remaining = self.get_time_remaining()
        
        # Stats table
        stats = Table(show_header=False, box=None)
        stats.add_column("Key", style="dim")
        stats.add_column("Value", style="bold")
        
        stats.add_row("Balance", f"${self.state.balance:.2f}")
        stats.add_row("Total PnL", f"${self.state.total_pnl:+.2f}")
        stats.add_row("Trades", str(len(self.state.trades)))
        
        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        stats.add_row("Win Rate", f"{win_rate:.1%} ({self.state.wins}/{self.state.wins + self.state.losses})")
        
        stats.add_row("", "")
        stats.add_row("Candle Start", candle_start.strftime("%H:%M:%S"))
        stats.add_row("Time Remaining", f"{int(time_remaining * 15)}:{int((time_remaining * 15 * 60) % 60):02d}")
        
        if self.state.candle_open_price:
            stats.add_row("Open Price", f"${self.state.candle_open_price:,.2f}")
        
        stats.add_row("", "")
        if self.state.current_position:
            pos_color = "green" if self.state.current_position == "yes" else "red"
            stats.add_row("Position", f"[{pos_color}]{self.state.current_position.upper()}[/] ({self.state.position_size:.0%})")
        else:
            stats.add_row("Position", "[dim]None[/dim]")
        
        if self.state.last_prediction:
            pred_color = {"Up": "green", "Down": "red", "Hold": "yellow"}.get(self.state.last_prediction, "white")
            stats.add_row("Prediction", f"[{pred_color}]{self.state.last_prediction}[/]")
            stats.add_row("Probs", f"↑{self.state.last_probs.get('up', 0):.0%} ↓{self.state.last_probs.get('down', 0):.0%}")
        
        # Recent trades
        if self.state.trades:
            stats.add_row("", "")
            stats.add_row("[bold]Recent Trades[/bold]", "")
            for trade in self.state.trades[-3:]:
                pnl_color = "green" if trade["pnl"] > 0 else "red"
                stats.add_row(
                    trade["time"].strftime("%H:%M"),
                    f"[{pnl_color}]{trade['outcome']} ${trade['pnl']:+.2f}[/]"
                )
        
        return Panel(stats, title="[bold blue]Live Paper Trading[/bold blue]", border_style="blue")
    
    async def run(self):
        """Main trading loop."""
        self._running = True
        last_candle_start = None
        
        console.print("[bold blue]Starting Live Paper Trading[/bold blue]")
        console.print(f"  Model: {self.config.deep_lob_model}")
        console.print(f"  Initial Balance: ${self.config.initial_balance:.2f}")
        console.print(f"  Candle: {self.config.candle_minutes} minutes")
        console.print()
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        console.print()
        
        try:
            with Live(self.build_display(), refresh_per_second=1, console=console) as live:
                while self._running:
                    candle_start, _ = self.get_candle_boundaries()
                    
                    # Check for new candle
                    if last_candle_start != candle_start:
                        await self.handle_new_candle()
                        last_candle_start = candle_start
                    
                    # Get data and make prediction
                    btc_data = await self.data_fetcher.get_recent_klines(limit=300)
                    
                    if len(btc_data) > 50:
                        time_in_candle = 1.0 - self.get_time_remaining()
                        
                        # Only trade after warmup and if no position
                        warmup_passed = time_in_candle > (self.config.warmup_seconds / (self.config.candle_minutes * 60))
                        
                        if warmup_passed and not self.state.current_position:
                            decision = await self.get_prediction(btc_data)
                            await self.execute_trade(decision)
                    
                    # Update display
                    live.update(self.build_display())
                    
                    await asyncio.sleep(self.config.update_interval)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
        finally:
            # Final settlement
            if self.state.current_position:
                await self.settle_position()
            
            await self.data_fetcher.close()
            
            # Print summary
            console.print()
            console.print("[bold]Final Summary[/bold]")
            console.print(f"  Balance: ${self.state.balance:.2f}")
            console.print(f"  Total PnL: ${self.state.total_pnl:+.2f}")
            console.print(f"  Return: {(self.state.balance / self.config.initial_balance - 1) * 100:+.1f}%")
            console.print(f"  Trades: {len(self.state.trades)}")
            if self.state.trades:
                win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
                console.print(f"  Win Rate: {win_rate:.1%}")
    
    def stop(self):
        """Stop the trading loop."""
        self._running = False


async def run_live_paper_trading(
    deep_lob_model: str = "./logs/deep_lob_balanced",
    initial_balance: float = 1000.0,
    aggressive: bool = True,
):
    """Main entry point for live paper trading."""
    config = LiveTradeConfig(
        deep_lob_model=deep_lob_model,
        initial_balance=initial_balance,
        aggressive=aggressive,
    )
    
    trader = LivePaperTrader(config)
    trader.load_model()
    
    # Handle interrupt
    def signal_handler(sig, frame):
        trader.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    await trader.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live paper trading")
    parser.add_argument("--model", default="./logs/deep_lob_balanced")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--aggressive", action="store_true", default=True)
    
    args = parser.parse_args()
    
    asyncio.run(run_live_paper_trading(
        deep_lob_model=args.model,
        initial_balance=args.balance,
        aggressive=args.aggressive,
    ))
