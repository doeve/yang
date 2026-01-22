#!/usr/bin/env python3
"""
Backtester for DeepLOB + SAC DSR model.

Uses real Polymarket 15-minute candle outcomes and the DSR-trained SAC model
to evaluate trading performance.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC

from src.simulation.deep_lob_execution_env import (
    DeepLOBExecutionConfig,
    DeepLOBExecutionEnv,
)
from src.inference.deep_lob_inference import (
    DeepLOBTwoLayerBot,
    DeepLOBInferenceConfig,
)

console = Console()


class DeepLOBBacktester:
    """
    Backtester that uses real Polymarket candle outcomes with DeepLOB predictions.
    
    Workflow:
    1. Load Polymarket candle data (670 candles with up_won outcomes)
    2. For each candle, fetch BTC price data from that period
    3. Run DeepLOB to get class probabilities
    4. Optionally run SAC to get trade execution decision
    5. Evaluate against real Polymarket outcome
    """
    
    def __init__(
        self,
        deep_lob_model: str = "./logs/deep_lob_balanced",
        sac_model: Optional[str] = None,
        polymarket_data: str = "./data/polymarket/btc_15min_candles.parquet",
        initial_balance: float = 10000.0,
        max_candles: int = 100,
    ):
        self.deep_lob_path = Path(deep_lob_model)
        self.sac_path = Path(sac_model) if sac_model else None
        self.polymarket_path = Path(polymarket_data)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_candles = max_candles
        
        # Models
        self.bot: Optional[DeepLOBTwoLayerBot] = None
        self.sac_model: Optional[SAC] = None
        
        # Stats
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.skipped = 0
    
    def load_models(self):
        """Load DeepLOB and optionally SAC models."""
        console.print("[blue]Loading DeepLOB model...[/blue]")
        
        config = DeepLOBInferenceConfig()
        self.bot = DeepLOBTwoLayerBot(config=config)
        self.bot.load_models(
            deep_lob_path=str(self.deep_lob_path),
            sac_path=None,  # We'll use SAC separately
        )
        console.print("[green]✓ DeepLOB loaded[/green]")
        
        if self.sac_path and self.sac_path.exists():
            console.print("[blue]Loading SAC model...[/blue]")
            self.sac_model = SAC.load(str(self.sac_path).replace('.zip', ''))
            console.print("[green]✓ SAC loaded[/green]")
    
    def load_polymarket_data(self) -> pd.DataFrame:
        """Load Polymarket candle outcomes."""
        if not self.polymarket_path.exists():
            raise FileNotFoundError(f"Polymarket data not found: {self.polymarket_path}")
        
        df = pd.read_parquet(self.polymarket_path)
        console.print(f"[green]✓ Loaded {len(df)} Polymarket candles[/green]")
        return df
    
    async def fetch_btc_data_for_candle(
        self,
        candle_time: datetime,
        client: httpx.AsyncClient,
    ) -> pd.DataFrame:
        """Fetch BTC 1s data for a specific 15-min candle."""
        # Get 20 minutes of data (5 min before + 15 min candle)
        start_time = int((candle_time.timestamp() - 300) * 1000)  # 5 min before
        end_time = int((candle_time.timestamp() + 900) * 1000)  # End of candle
        
        try:
            resp = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1s",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1200,
                }
            )
            resp.raise_for_status()
            
            klines = resp.json()
            if not klines:
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["price"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["taker_buy_volume"] = df["taker_buy_base"].astype(float)
            
            return df[["timestamp", "price", "volume", "taker_buy_volume"]]
            
        except Exception as e:
            console.print(f"[red]Error fetching BTC data: {e}[/red]")
            return pd.DataFrame()
    
    def get_sac_action(self, deeplob_probs: np.ndarray, market_price: float) -> dict:
        """Get SAC execution decision from DeepLOB probabilities."""
        if self.sac_model is None:
            # Rule-based fallback
            prob_down, prob_hold, prob_up = deeplob_probs
            confidence = max(deeplob_probs)
            
            if confidence < 0.40:
                return {"action": "hold", "size": 0.0}
            
            if prob_up > prob_down and prob_up > prob_hold:
                return {"action": "buy_yes", "size": min(confidence * 0.5, 0.25)}
            elif prob_down > prob_up and prob_down > prob_hold:
                return {"action": "buy_no", "size": min(confidence * 0.5, 0.25)}
            else:
                return {"action": "hold", "size": 0.0}
        
        # Build observation for SAC (20 dimensions - DeepLOBDynamicEnv format)
        prob_down, prob_hold, prob_up = deeplob_probs
        predicted_class = np.argmax(deeplob_probs)
        confidence = max(deeplob_probs)
        
        # Simulated values matching DeepLOBDynamicEnv observation space
        spread = 0.02
        time_remaining = 0.5  # Middle of candle
        steps_since_entry = 0.0
        position_size = 0.0
        unrealized_pnl = 0.0
        max_pnl = 0.0
        drawdown = 0.0
        trades_this_candle = 0.0
        win_rate = self.wins / max(1, self.wins + self.losses)
        balance_norm = self.balance / self.initial_balance - 1.0
        volatility = 0.5
        model_implied = 0.5 + (prob_up - prob_down) * 0.5
        edge_up = model_implied - market_price
        edge_down = (1 - model_implied) - (1 - market_price)
        
        # 20-dimension observation for DeepLOBDynamicEnv
        obs = np.array([
            # DeepLOB predictions (3)
            prob_down, prob_hold, prob_up,
            # Prediction (2)
            predicted_class / 2.0,
            confidence,
            # Market (3)
            market_price,
            1.0 - market_price,
            spread,
            # Time (2)
            time_remaining,
            steps_since_entry,
            # Position (4)
            position_size,
            unrealized_pnl,
            max_pnl,
            drawdown,
            # History (3)
            trades_this_candle,
            win_rate,
            balance_norm,
            # Volatility (1)
            volatility,
            # Edge (2)
            edge_up,
            edge_down,
        ], dtype=np.float32).reshape(1, -1)
        
        action, _ = self.sac_model.predict(obs, deterministic=True)
        
        direction = float(action[0][0])
        size = float(np.clip(action[0][1], 0.0, 1.0))
        hold_prob = float(action[0][2])
        exit_signal = float(action[0][3]) if len(action[0]) > 3 else 0.0
        
        if hold_prob > 0.5 or size < 0.05:
            return {"action": "hold", "size": 0.0}
        
        if direction > 0.1:
            return {"action": "buy_yes", "size": size * 0.25}
        elif direction < -0.1:
            return {"action": "buy_no", "size": size * 0.25}
        else:
            return {"action": "hold", "size": 0.0}

    
    async def run(self):
        """Run the backtest."""
        self.load_models()
        
        polymarket_df = self.load_polymarket_data()
        
        # Only use closed candles with outcomes
        closed_candles = polymarket_df[polymarket_df["closed"] == True].copy()
        
        if len(closed_candles) == 0:
            console.print("[red]No closed candles found[/red]")
            return
        
        # Limit to max_candles
        test_candles = closed_candles.head(self.max_candles)
        
        console.print(f"\n[bold green]═══ DEEPLOB BACKTEST ═══[/bold green]")
        console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
        console.print(f"Candles to test: {len(test_candles)}")
        console.print()
        
        async with httpx.AsyncClient(timeout=30) as client:
            for idx, (_, candle) in enumerate(test_candles.iterrows()):
                # Get candle timestamp
                ts = candle["timestamp"]
                if isinstance(ts, (int, float)):
                    candle_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    candle_time = ts.to_pydatetime()
                
                # Polymarket outcome
                up_won = candle["up_won"]
                actual_direction = "UP" if up_won else "DOWN"
                
                # Fetch BTC data for this period
                btc_data = await self.fetch_btc_data_for_candle(candle_time, client)
                
                if len(btc_data) < 50:
                    self.skipped += 1
                    continue
                
                # Get DeepLOB prediction using step() method
                try:
                    decision = self.bot.step(
                        btc_data=btc_data,
                        market_yes_price=0.5,
                        market_spread=0.02,
                        time_remaining=0.5,
                    )
                except Exception as e:
                    console.print(f"[dim]Candle {idx+1}: Error - {e}[/dim]")
                    self.skipped += 1
                    continue
                
                # Extract probabilities from decision
                probs = np.array([
                    decision.get("prob_down", 0.33),
                    decision.get("prob_hold", 0.34),
                    decision.get("prob_up", 0.33),
                ])

                
                trade_decision = self.get_sac_action(probs, 0.5)
                
                if trade_decision["action"] == "hold":
                    self.skipped += 1
                    if idx < 10:
                        console.print(f"[dim]Candle {idx+1}: HOLD (skip) | Actual: {actual_direction}[/dim]")
                    continue
                
                # Evaluate trade
                predicted = "UP" if trade_decision["action"] == "buy_yes" else "DOWN"
                size = trade_decision["size"]
                
                if predicted == actual_direction:
                    pnl_pct = size * 0.15  # Assume 15% profit on correct
                    self.wins += 1
                    result = "[green]✓[/green]"
                else:
                    pnl_pct = -size * 0.10  # Assume 10% loss on incorrect
                    self.losses += 1
                    result = "[red]✗[/red]"
                
                pnl = self.balance * pnl_pct
                self.balance += pnl
                
                self.trades.append({
                    "time": candle_time,
                    "predicted": predicted,
                    "actual": actual_direction,
                    "size": size,
                    "pnl": pnl,
                    "balance": self.balance,
                    "correct": predicted == actual_direction,
                })
                
                if idx < 20 or idx % 20 == 0:
                    console.print(
                        f"{result} Candle {idx+1}: Pred={predicted} Actual={actual_direction} "
                        f"PnL=${pnl:+.2f} Bal=${self.balance:,.2f}"
                    )
                
                # Rate limiting
                await asyncio.sleep(0.2)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print backtest summary."""
        console.print()
        console.print("[bold]═══ BACKTEST RESULTS ═══[/bold]")
        console.print()
        
        table = Table(title="Performance Summary")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="green")
        
        pnl = self.balance - self.initial_balance
        pnl_pct = pnl / self.initial_balance * 100
        
        table.add_row("Initial Balance", f"${self.initial_balance:,.2f}")
        table.add_row("Final Balance", f"${self.balance:,.2f}")
        
        color = "green" if pnl >= 0 else "red"
        table.add_row("Total PnL", f"[{color}]${pnl:+,.2f} ({pnl_pct:+.1f}%)[/{color}]")
        
        table.add_row("", "")
        table.add_row("Total Trades", str(self.wins + self.losses))
        table.add_row("Wins", str(self.wins))
        table.add_row("Losses", str(self.losses))
        table.add_row("Skipped", str(self.skipped))
        
        if self.wins + self.losses > 0:
            win_rate = self.wins / (self.wins + self.losses)
            table.add_row("Win Rate", f"{win_rate:.1%}")
        
        if self.trades:
            pnls = [t["pnl"] for t in self.trades]
            table.add_row("Avg Trade", f"${np.mean(pnls):+.2f}")
            table.add_row("Max Win", f"${max(pnls):+.2f}")
            table.add_row("Max Loss", f"${min(pnls):+.2f}")
        
        console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Backtest DeepLOB + SAC with Polymarket data")
    
    parser.add_argument("--deep-lob", "-d", default="./logs/deep_lob_balanced", help="DeepLOB model path")
    parser.add_argument("--sac", "-s", default=None, help="SAC model path (optional)")
    parser.add_argument("--polymarket", "-p", default="./data/polymarket/btc_15min_candles.parquet")
    parser.add_argument("--balance", "-b", type=float, default=10000.0)
    parser.add_argument("--candles", "-n", type=int, default=100, help="Max candles to test")
    
    args = parser.parse_args()
    
    backtester = DeepLOBBacktester(
        deep_lob_model=args.deep_lob,
        sac_model=args.sac,
        polymarket_data=args.polymarket,
        initial_balance=args.balance,
        max_candles=args.candles,
    )
    
    await backtester.run()


if __name__ == "__main__":
    asyncio.run(main())
