"""
Paper trading script for DeepLOB + SAC two-layer system.

Fetches random intervals of data from Binance and simulates trading decisions.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from src.inference.deep_lob_inference import DeepLOBTwoLayerBot, DeepLOBInferenceConfig

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class PaperTradeConfig:
    """Configuration for paper trading."""
    
    # Model paths
    deep_lob_path: str = "./logs/deep_lob_v2"
    sac_path: str = "./logs/sac_deeplob"
    
    # Trading parameters
    initial_balance: float = 1000.0
    candle_minutes: int = 15
    
    # Data parameters
    lookback_seconds: int = 200  # Seconds of data to fetch for prediction
    data_interval: str = "1s"  # Binance kline interval
    
    # Simulation
    num_candles: int = 50
    random_intervals: bool = True  # Fetch from random historical times
    fee_percent: float = 0.1


class BinanceDataFetcher:
    """Fetch data from Binance API."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
    
    async def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1s",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch kline/candlestick data."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        response = await self.client.get(f"{self.BASE_URL}/klines", params=params)
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
    
    async def fetch_random_interval(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1s",
        lookback_seconds: int = 200,
        days_ago_max: int = 7,
    ) -> pd.DataFrame:
        """Fetch data from a random historical time."""
        # Random time in past N days
        now = datetime.utcnow()
        random_offset = random.randint(1, days_ago_max * 24 * 60)  # Random minutes
        target_time = now - timedelta(minutes=random_offset)
        
        end_time = int(target_time.timestamp() * 1000)
        start_time = int((target_time - timedelta(seconds=lookback_seconds)).timestamp() * 1000)
        
        return await self.fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=lookback_seconds + 100,
            start_time=start_time,
            end_time=end_time,
        )
    
    async def close(self):
        await self.client.aclose()


class DeepLOBPaperTrader:
    """Paper trader for DeepLOB + SAC system."""
    
    def __init__(self, config: Optional[PaperTradeConfig] = None):
        self.config = config or PaperTradeConfig()
        self.bot: Optional[DeepLOBTwoLayerBot] = None
        self.fetcher = BinanceDataFetcher()
        
        # Trading state
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.position_entry_price = 0.0
        
        # History
        self.trades: list[dict] = []
        self.decisions: list[dict] = []
    
    def load_models(self):
        """Load DeepLOB and SAC models."""
        console.print("[blue]Loading models...[/blue]")
        
        self.bot = DeepLOBTwoLayerBot()
        
        sac_path = self.config.sac_path
        if not Path(sac_path).exists() or not Path(f"{sac_path}/execution_model.zip").exists():
            sac_path = None
            console.print("[yellow]SAC model not found, using rule-based execution[/yellow]")
        
        self.bot.load_models(self.config.deep_lob_path, sac_path)
        console.print("[green]✓ Models loaded[/green]")
    
    async def simulate_candle(self, candle_num: int) -> dict:
        """Simulate one candle with random historical data."""
        # Fetch random data
        if self.config.random_intervals:
            btc_data = await self.fetcher.fetch_random_interval(
                lookback_seconds=self.config.lookback_seconds,
            )
        else:
            btc_data = await self.fetcher.fetch_klines(
                limit=self.config.lookback_seconds,
            )
        
        if len(btc_data) < 50:
            return {"error": "Insufficient data"}
        
        # Current market price (simulate Polymarket YES price)
        current_price = btc_data["price"].iloc[-1]
        prev_price = btc_data["price"].iloc[0]
        actual_return = (current_price - prev_price) / prev_price
        
        # Simulate market YES price (with some noise)
        market_yes_price = 0.5 + np.clip(np.random.normal(0, 0.1), -0.3, 0.3)
        
        # Get decision from bot
        class_probs = self.bot.predict_class_probabilities(btc_data)
        edge = 0.5 + (class_probs['up'] - class_probs['down']) * 0.5 - market_yes_price
        
        # Use aggressive mode if enabled
        if getattr(self, 'aggressive', False) and self.bot.sac_model is None:
            decision = self.bot._rule_based_decision(class_probs, market_yes_price, edge, aggressive=True)
            decision.update({
                "prob_down": class_probs["down"],
                "prob_hold": class_probs["hold"],
                "prob_up": class_probs["up"],
                "predicted_class": ["Down", "Hold", "Up"][class_probs["predicted_class"]],
            })
        else:
            decision = self.bot.step(
                btc_data=btc_data,
                market_yes_price=market_yes_price,
                market_spread=0.02,
                time_remaining=0.5,
            )

        
        # Determine actual outcome (based on future 15min move)
        # For paper trading, we'll simulate using the actual return
        if actual_return > 0.0001:
            outcome = 2  # Up
        elif actual_return < -0.0001:
            outcome = 0  # Down
        else:
            outcome = 1  # Hold
        
        # Update bias corrector with actual outcome
        predicted_class = {"Down": 0, "Hold": 1, "Up": 2}.get(decision.get("predicted_class", "Hold"), 1)
        self.bot.bias_corrector.update(predicted_class, outcome)
        
        # Execute trade
        pnl = 0.0
        if decision["action"] != "hold":
            # Calculate PnL
            position_size = decision["size"] * self.balance
            
            if decision["action"] == "buy_yes":
                # Bought YES, win if Up
                if outcome == 2:
                    pnl = position_size * (1 - market_yes_price - 0.002)
                else:
                    pnl = -position_size * market_yes_price
            else:  # buy_no
                # Bought NO, win if Down
                if outcome == 0:
                    pnl = position_size * (market_yes_price - 0.002)
                else:
                    pnl = -position_size * (1 - market_yes_price)
            
            # Apply fees
            pnl -= position_size * self.config.fee_percent * 0.01
            
            self.balance += pnl
            
            self.trades.append({
                "candle": candle_num,
                "action": decision["action"],
                "size": decision["size"],
                "pnl": pnl,
                "balance": self.balance,
            })
        
        result = {
            "candle": candle_num,
            "timestamp": btc_data["timestamp"].iloc[-1],
            "price": current_price,
            "actual_return": actual_return,
            "outcome": ["Down", "Hold", "Up"][outcome],
            **decision,
            "pnl": pnl,

            "balance": self.balance,
        }
        
        self.decisions.append(result)
        return result
    
    async def run(self, num_candles: Optional[int] = None):
        """Run paper trading simulation."""
        num_candles = num_candles or self.config.num_candles
        
        console.print(f"\n[bold blue]Starting Paper Trading[/bold blue]")
        console.print(f"  Candles: {num_candles}")
        console.print(f"  Initial balance: ${self.config.initial_balance:.2f}")
        console.print()
        
        results_table = Table(title="Paper Trading Results")
        results_table.add_column("#", style="dim")
        results_table.add_column("Time", style="cyan")
        results_table.add_column("Pred", style="yellow")
        results_table.add_column("Action", style="green")
        results_table.add_column("Size", style="blue")
        results_table.add_column("Outcome")
        results_table.add_column("PnL", style="magenta")
        results_table.add_column("Balance", style="bold")
        
        for i in range(num_candles):
            try:
                result = await self.simulate_candle(i + 1)
                
                if "error" in result:
                    console.print(f"[yellow]Candle {i+1}: {result['error']}[/yellow]")
                    continue
                
                # Add to table
                pnl_color = "green" if result["pnl"] > 0 else "red" if result["pnl"] < 0 else "dim"
                
                results_table.add_row(
                    str(i + 1),
                    str(result["timestamp"].strftime("%H:%M") if hasattr(result["timestamp"], "strftime") else "N/A"),
                    result["predicted_class"],
                    result["action"],
                    f"{result['size']:.2f}",
                    result["outcome"],
                    f"[{pnl_color}]${result['pnl']:.2f}[/{pnl_color}]",
                    f"${result['balance']:.2f}",
                )
                
                # Rate limit
                await asyncio.sleep(0.5)
                
            except Exception as e:
                console.print(f"[red]Error on candle {i+1}: {e}[/red]")
                await asyncio.sleep(1)
        
        console.print(results_table)
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print trading summary."""
        console.print("\n[bold]Trading Summary[/bold]")
        console.print(f"  Total candles: {len(self.decisions)}")
        console.print(f"  Trades: {len(self.trades)}")
        console.print(f"  Hold decisions: {len(self.decisions) - len(self.trades)}")
        
        if self.trades:
            wins = sum(1 for t in self.trades if t["pnl"] > 0)
            total_pnl = sum(t["pnl"] for t in self.trades)
            
            console.print(f"\n  Win rate: {wins/len(self.trades)*100:.1f}%")
            console.print(f"  Total PnL: ${total_pnl:.2f}")
            console.print(f"  Final balance: ${self.balance:.2f}")
            console.print(f"  Return: {(self.balance/self.config.initial_balance - 1)*100:.1f}%")
        else:
            console.print("\n  [dim]No trades executed (all Hold decisions)[/dim]")
    
    async def close(self):
        """Cleanup resources."""
        await self.fetcher.close()


async def run_paper_trade(
    num_candles: int = 50,
    deep_lob_path: str = "./logs/deep_lob_v2",
    sac_path: str = "./logs/sac_deeplob",
    initial_balance: float = 1000.0,
    aggressive: bool = True,
):
    """Main entry point for paper trading."""
    config = PaperTradeConfig(
        deep_lob_path=deep_lob_path,
        sac_path=sac_path,
        initial_balance=initial_balance,
        num_candles=num_candles,
    )
    
    trader = DeepLOBPaperTrader(config)
    trader.aggressive = aggressive
    
    try:
        trader.load_models()
        await trader.run()
    finally:
        await trader.close()



def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper trade DeepLOB + SAC")
    parser.add_argument("--candles", type=int, default=50, help="Number of candles")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial balance")
    parser.add_argument("--deep-lob", default="./logs/deep_lob_v2", help="DeepLOB model path")
    parser.add_argument("--sac", default="./logs/sac_deeplob", help="SAC model path")
    
    args = parser.parse_args()
    
    asyncio.run(run_paper_trade(
        num_candles=args.candles,
        deep_lob_path=args.deep_lob,
        sac_path=args.sac,
        initial_balance=args.balance,
    ))


if __name__ == "__main__":
    main()
