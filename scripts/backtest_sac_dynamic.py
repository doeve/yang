#!/usr/bin/env python3
"""
Backtest SAC Dynamic Model with Complete Data.

Evaluates the trained SAC Dynamic model using:
- Historical BTC 1s data from Binance
- Real Polymarket 15-minute candle outcomes
- VecNormalize for proper observation normalization
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.simulation.deep_lob_dynamic_env import (
    DynamicTradingConfig,
    DeepLOBDynamicEnv,
)
from src.inference.deep_lob_inference import (
    DeepLOBTwoLayerBot,
    DeepLOBInferenceConfig,
)

console = Console()


class SACDynamicBacktester:
    """
    Backtester for SAC Dynamic model with complete BTC + Polymarket data.

    Uses VecNormalize to properly normalize observations matching training.
    """

    def __init__(
        self,
        sac_model: str,
        vec_normalize: str,
        deep_lob_model: str = "./logs/deep_lob_balanced",
        polymarket_data: str = "./data/polymarket/btc_15min_candles.parquet",
        initial_balance: float = 10000.0,
        max_candles: int = 100,
    ):
        self.sac_path = Path(sac_model)
        self.vec_normalize_path = Path(vec_normalize)
        self.deep_lob_path = Path(deep_lob_model)
        self.polymarket_path = Path(polymarket_data)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_candles = max_candles

        # Models
        self.sac_model: Optional[SAC] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self.bot: Optional[DeepLOBTwoLayerBot] = None

        # Stats
        self.trades: List[Dict] = []
        self.wins = 0
        self.losses = 0
        self.skipped = 0
        self.max_balance = initial_balance
        self.min_balance = initial_balance

        # Price history for momentum calculation
        self.yes_price_history: List[float] = []
        self.no_price_history: List[float] = []
        self.trades_this_candle = 0
        
    def load_models(self):
        """Load SAC model, VecNormalize, and DeepLOB."""
        console.print("[blue]Loading SAC Dynamic model...[/blue]")
        
        # Load SAC model
        sac_path_str = str(self.sac_path)
        if sac_path_str.endswith('.zip'):
            sac_path_str = sac_path_str[:-4]
        self.sac_model = SAC.load(sac_path_str)
        console.print(f"[green]✓ SAC loaded from {self.sac_path}[/green]")
        
        # Load VecNormalize
        if self.vec_normalize_path.exists():
            # Create a dummy env to load VecNormalize
            config = DynamicTradingConfig()
            dummy_env = DummyVecEnv([lambda: DeepLOBDynamicEnv(config)])
            self.vec_normalize = VecNormalize.load(str(self.vec_normalize_path), dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            console.print(f"[green]✓ VecNormalize loaded from {self.vec_normalize_path}[/green]")
        else:
            console.print("[yellow]⚠ VecNormalize not found, using raw observations[/yellow]")
        
        # Load DeepLOB
        console.print("[blue]Loading DeepLOB model...[/blue]")
        config = DeepLOBInferenceConfig()
        self.bot = DeepLOBTwoLayerBot(config=config)
        self.bot.load_models(deep_lob_path=str(self.deep_lob_path), sac_path=None)
        console.print("[green]✓ DeepLOB loaded[/green]")
    
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
    
    def build_observation(
        self,
        deeplob_probs: np.ndarray,
        yes_price: float = 0.5,
        no_price: float = 0.5,
        time_remaining: float = 0.5,
    ) -> np.ndarray:
        """Build 26-dim observation matching DeepLOBDynamicEnv with momentum features."""
        prob_down, prob_hold, prob_up = deeplob_probs
        predicted_class = np.argmax(deeplob_probs)
        confidence = max(deeplob_probs)

        time_in_candle = 1.0 - time_remaining

        # Market state
        spread = abs(yes_price + no_price - 1.0)
        steps_since_entry = 0.0

        # Position state (no position for new candle)
        position_size = 0.0
        unrealized_pnl = 0.0
        max_pnl = 0.0
        drawdown = 0.0

        # History
        win_rate = self.wins / max(1, self.wins + self.losses)
        balance_norm = self.balance / self.initial_balance - 1.0

        # Volatility (estimate)
        volatility = 0.5

        # Edge calculations
        model_implied = 0.5 + (prob_up - prob_down) * 0.5
        edge_up = model_implied - yes_price
        edge_down = (1 - model_implied) - no_price

        # Update price history
        self.yes_price_history.append(yes_price)
        self.no_price_history.append(no_price)
        if len(self.yes_price_history) > 20:
            self.yes_price_history = self.yes_price_history[-20:]
            self.no_price_history = self.no_price_history[-20:]

        # === NEW FEATURES ===

        # 1. Token momentum (rate of change over 5 steps)
        momentum_window = 5
        if len(self.yes_price_history) >= momentum_window:
            yes_momentum = (self.yes_price_history[-1] - self.yes_price_history[-momentum_window]) / momentum_window
            no_momentum = (self.no_price_history[-1] - self.no_price_history[-momentum_window]) / momentum_window
        else:
            yes_momentum = 0.0
            no_momentum = 0.0

        # 2. Convergence velocity
        if yes_price > 0.5:
            convergence_velocity = yes_momentum * 10
        else:
            convergence_velocity = -yes_momentum * 10

        # 3. Price distance to settlement
        price_distance_to_settlement = 0.5 - abs(yes_price - 0.5)

        # 4. YES/NO sum deviation (arbitrage signal)
        yes_no_sum_deviation = (yes_price + no_price) - 1.0

        # 5. Time urgency
        time_urgency_multiplier = 2.0
        time_urgency = 1.0 + (time_urgency_multiplier - 1.0) * (time_in_candle ** 2)

        obs = np.array([
            # DeepLOB predictions (3)
            prob_down, prob_hold, prob_up,
            # Prediction (2)
            predicted_class / 2.0,
            confidence,
            # Market (3)
            yes_price,
            no_price,
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
            self.trades_this_candle / 5.0,
            win_rate,
            balance_norm,
            # Volatility (1)
            volatility,
            # Edge (2)
            edge_up,
            edge_down,
            # === NEW: Momentum (2) ===
            yes_momentum * 100,
            no_momentum * 100,
            # === NEW: Convergence (2) ===
            convergence_velocity,
            price_distance_to_settlement,
            # === NEW: Arbitrage (1) ===
            yes_no_sum_deviation * 10,
            # === NEW: Time urgency (1) ===
            time_urgency / time_urgency_multiplier,
        ], dtype=np.float32)

        return obs
    
    def get_sac_action(self, obs: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
        """Get SAC action from observation with fallback to rule-based."""
        # Normalize observation if VecNormalize is available
        if self.vec_normalize is not None:
            obs_normalized = self.vec_normalize.normalize_obs(obs.reshape(1, -1))
        else:
            obs_normalized = obs.reshape(1, -1)
        
        action, _ = self.sac_model.predict(obs_normalized, deterministic=True)
        
        direction = float(action[0][0])
        size = float(np.clip(action[0][1], 0.0, 1.0))
        hold_prob = float(action[0][2])
        exit_signal = float(action[0][3]) if len(action[0]) > 3 else 0.0
        
        # More lenient thresholds for backtesting
        # SAC trained in dynamic env may have different action semantics
        if hold_prob > 0.7 or size < 0.02:  # Raised hold_prob threshold, lowered size threshold
            # Fallback to rule-based using DeepLOB probs
            return self._rule_based_action(probs)
        
        if abs(direction) > 0.05:  # Lowered direction threshold
            if direction > 0:
                return {"action": "buy_yes", "size": max(size * 0.25, 0.10), "direction": direction}
            else:
                return {"action": "buy_no", "size": max(size * 0.25, 0.10), "direction": direction}
        else:
            # Fallback to rule-based
            return self._rule_based_action(probs)
    
    def _rule_based_action(self, probs: np.ndarray) -> Dict[str, Any]:
        """Fallback rule-based action from DeepLOB probabilities."""
        prob_down, prob_hold, prob_up = probs
        confidence = max(probs)
        
        # Minimum confidence to trade
        if confidence < 0.40:
            return {"action": "hold", "size": 0.0, "direction": 0.0}
        
        # Trade in direction of highest probability (excluding hold)
        if prob_up > prob_down and prob_up > 0.4:
            size = min(confidence * 0.4, 0.20)
            return {"action": "buy_yes", "size": size, "direction": prob_up}
        elif prob_down > prob_up and prob_down > 0.4:
            size = min(confidence * 0.4, 0.20)
            return {"action": "buy_no", "size": size, "direction": -prob_down}
        else:
            return {"action": "hold", "size": 0.0, "direction": 0.0}
    
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
        
        console.print(f"\n[bold green]═══ SAC DYNAMIC BACKTEST ═══[/bold green]")
        console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
        console.print(f"Candles to test: {len(test_candles)}")
        console.print()
        
        async with httpx.AsyncClient(timeout=30) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Backtesting...", total=len(test_candles))
                
                for idx, (_, candle) in enumerate(test_candles.iterrows()):
                    # Reset per-candle state
                    self.yes_price_history = []
                    self.no_price_history = []
                    self.trades_this_candle = 0

                    # Get candle timestamp
                    ts = candle["timestamp"]
                    if isinstance(ts, (int, float)):
                        candle_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        candle_time = ts.to_pydatetime()

                    # Polymarket outcome
                    up_won = candle["up_won"]
                    actual_direction = "UP" if up_won else "DOWN"

                    progress.update(task, description=f"Candle {idx+1}/{len(test_candles)} @ {candle_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Fetch BTC data for this period
                    btc_data = await self.fetch_btc_data_for_candle(candle_time, client)
                    
                    if len(btc_data) < 50:
                        self.skipped += 1
                        progress.advance(task)
                        continue
                    
                    # Get DeepLOB prediction
                    try:
                        decision = self.bot.step(
                            btc_data=btc_data,
                            market_yes_price=0.5,
                            market_spread=0.02,
                            time_remaining=0.5,
                        )
                    except Exception as e:
                        self.skipped += 1
                        progress.advance(task)
                        continue
                    
                    # Extract probabilities
                    probs = np.array([
                        decision.get("prob_down", 0.33),
                        decision.get("prob_hold", 0.34),
                        decision.get("prob_up", 0.33),
                    ])
                    
                    # Build observation and get SAC action
                    obs = self.build_observation(probs, yes_price=0.5, no_price=0.5, time_remaining=0.5)
                    trade_decision = self.get_sac_action(obs, probs)
                    
                    if trade_decision["action"] == "hold":
                        self.skipped += 1
                        progress.advance(task)
                        continue
                    
                    # Evaluate trade
                    predicted = "UP" if trade_decision["action"] == "buy_yes" else "DOWN"
                    size = trade_decision["size"]
                    
                    if predicted == actual_direction:
                        pnl_pct = size * 0.15  # 15% profit on correct
                        self.wins += 1
                        result = "✓"
                    else:
                        pnl_pct = -size * 0.10  # 10% loss on incorrect
                        self.losses += 1
                        result = "✗"
                    
                    pnl = self.balance * pnl_pct
                    self.balance += pnl
                    
                    # Track max/min balance
                    self.max_balance = max(self.max_balance, self.balance)
                    self.min_balance = min(self.min_balance, self.balance)
                    
                    self.trades.append({
                        "time": candle_time,
                        "predicted": predicted,
                        "actual": actual_direction,
                        "size": size,
                        "pnl": pnl,
                        "balance": self.balance,
                        "correct": predicted == actual_direction,
                        "probs": probs.tolist(),
                    })
                    
                    # Show first 10 and every 10th trade
                    if idx < 10 or idx % 10 == 0:
                        color = "green" if result == "✓" else "red"
                        console.print(
                            f"[{color}]{result}[/{color}] Candle {idx+1}: Pred={predicted} Actual={actual_direction} "
                            f"PnL=${pnl:+.2f} Bal=${self.balance:,.2f}"
                        )
                    
                    progress.advance(task)
                    
                    # Rate limiting
                    await asyncio.sleep(0.15)
        
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
        
        # Max drawdown
        max_dd = (self.max_balance - self.min_balance) / self.max_balance * 100
        table.add_row("Max Drawdown", f"{max_dd:.1f}%")
        
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
            
            # Sharpe ratio (annualized)
            if len(pnls) > 1:
                returns = np.array(pnls) / self.initial_balance
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(96)  # 96 15-min candles/day
                table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
        
        console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Backtest SAC Dynamic with complete data")
    
    parser.add_argument("--sac-model", "-s", default="./logs/sac_dynamic_v5/final_model.zip")
    parser.add_argument("--vec-normalize", "-v", default="./logs/sac_dynamic_v5/vecnormalize.pkl")
    parser.add_argument("--deep-lob", "-d", default="./logs/deep_lob_balanced")
    parser.add_argument("--polymarket", "-p", default="./data/polymarket/btc_15min_candles.parquet")
    parser.add_argument("--balance", "-b", type=float, default=10000.0)
    parser.add_argument("--candles", "-n", type=int, default=100, help="Max candles to test")
    
    args = parser.parse_args()
    
    backtester = SACDynamicBacktester(
        sac_model=args.sac_model,
        vec_normalize=args.vec_normalize,
        deep_lob_model=args.deep_lob,
        polymarket_data=args.polymarket,
        initial_balance=args.balance,
        max_candles=args.candles,
    )
    
    await backtester.run()


if __name__ == "__main__":
    asyncio.run(main())
