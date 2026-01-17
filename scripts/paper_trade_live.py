#!/usr/bin/env python3
"""
Live Paper Trading for Multi-Asset RecurrentPPO Model.

Simulates the exact Polymarket BTC candle betting scenario:
- 15-minute candles aligned to :00, :15, :30, :45 UTC
- Streams BTC tick data from Binance
- Streams DXY and EUR/USD forex data from Yahoo Finance
- Uses trained model with identical observation space as training

Usage:
    python scripts/paper_trade_live.py --model logs/recurrent_multi_asset/multi_asset_model
"""

import asyncio
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
import httpx
import aiohttp
from rich.console import Console
from rich.table import Table

# Model imports
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.simulation.multi_asset_env import MultiAssetEnv, MultiAssetConfig

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

console = Console()


def get_candle_boundaries(candle_minutes: int = 15) -> tuple[datetime, datetime]:
    """Get current candle start and end times aligned to clock."""
    now = datetime.now(timezone.utc)
    minute_slot = (now.minute // candle_minutes) * candle_minutes
    candle_start = now.replace(minute=minute_slot, second=0, microsecond=0)
    candle_end = candle_start + timedelta(minutes=candle_minutes)
    return candle_start, candle_end


class LiveForexStreamer:
    """
    Streams live forex data from Yahoo Finance.
    Uses same data source as training data collection.
    """
    
    YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    YAHOO_SYMBOLS = {
        "DXY": "DX-Y.NYB",
        "EURUSD": "EURUSD=X",
    }
    
    def __init__(self):
        self.dxy_prices: deque = deque(maxlen=100)
        self.eurusd_prices: deque = deque(maxlen=100)
        self.last_fetch: Optional[datetime] = None
        self.fetch_interval = 60  # Fetch new data every 60 seconds
    
    async def _fetch_yahoo(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        """Fetch latest price from Yahoo Finance."""
        yahoo_sym = self.YAHOO_SYMBOLS.get(symbol, symbol)
        url = f"{self.YAHOO_URL}/{yahoo_sym}"
        params = {"interval": "1m", "range": "1d"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        
        try:
            async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
            
            quote = result[0].get("indicators", {}).get("quote", [{}])[0]
            closes = quote.get("close", [])
            
            # Get latest non-null price
            for price in reversed(closes):
                if price is not None:
                    return float(price)
            return None
            
        except Exception as e:
            console.print(f"[dim]Forex fetch error ({symbol}): {e}[/dim]")
            return None
    
    async def update(self):
        """Fetch latest forex data."""
        now = datetime.now(timezone.utc)
        if self.last_fetch and (now - self.last_fetch).total_seconds() < self.fetch_interval:
            return  # Don't fetch too frequently
        
        async with aiohttp.ClientSession() as session:
            dxy = await self._fetch_yahoo(session, "DXY")
            if dxy:
                self.dxy_prices.append(dxy)
            
            eurusd = await self._fetch_yahoo(session, "EURUSD")
            if eurusd:
                self.eurusd_prices.append(eurusd)
        
        self.last_fetch = now
    
    def get_returns(self) -> dict:
        """Get 1h and 4h returns (approximated from available data)."""
        result = {"dxy_1h": 0.0, "dxy_4h": 0.0, "eurusd_1h": 0.0, "eurusd_4h": 0.0}
        
        if len(self.dxy_prices) >= 2:
            result["dxy_1h"] = (self.dxy_prices[-1] / self.dxy_prices[-2] - 1) * 100
        if len(self.dxy_prices) >= 5:
            result["dxy_4h"] = (self.dxy_prices[-1] / self.dxy_prices[-5] - 1) * 100
        
        if len(self.eurusd_prices) >= 2:
            result["eurusd_1h"] = (self.eurusd_prices[-1] / self.eurusd_prices[-2] - 1) * 100
        if len(self.eurusd_prices) >= 5:
            result["eurusd_4h"] = (self.eurusd_prices[-1] / self.eurusd_prices[-5] - 1) * 100
        
        return result
    
    @property
    def has_dxy(self) -> bool:
        return len(self.dxy_prices) > 0
    
    @property
    def has_eurusd(self) -> bool:
        return len(self.eurusd_prices) > 0


class LivePaperTrader:
    """
    Live paper trading matching exact Polymarket scenario.
    
    Candles: 15 minutes, aligned to :00, :15, :30, :45 UTC
    
    Observation (609 dims):
    - BTC prices (300) normalized to candle open
    - BTC volumes (300) normalized to max
    - DXY returns (2) - 1h, 4h
    - EUR/USD returns (2) - 1h, 4h  
    - Context (5) - current_vs_open, time_remaining, hour_sin, hour_cos, atr
    """
    
    BINANCE_REST = "https://api.binance.com/api/v3"
    CANDLE_MINUTES = 15
    HISTORY_LENGTH = 300
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "./data",
        initial_balance: float = 10000.0,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # BTC price/volume history
        self.price_history: deque = deque(maxlen=2000)
        self.volume_history: deque = deque(maxlen=2000)
        self.current_price = 0.0
        
        # Live forex streamer
        self.forex = LiveForexStreamer()
        
        # Candle state
        self.candle_start: Optional[datetime] = None
        self.candle_end: Optional[datetime] = None
        self.candle_open = 0.0
        
        # Model state
        self.model = None
        self.is_recurrent = False
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.vec_normalize = None
        
        # Position state
        self.position = None
        
        # Stats
        self.trades: list[dict] = []
        self.total_trades = 0
        self.correct_trades = 0
    
    async def _load_model(self):
        """Load trained model and VecNormalize with matching observation space."""
        console.print(f"\n[bold blue]Loading model from {self.model_path}...[/bold blue]")
        
        # Load model
        if RecurrentPPO:
            try:
                self.model = RecurrentPPO.load(self.model_path)
                self.is_recurrent = True
                console.print("[green]✓ Loaded RecurrentPPO (LSTM policy)[/green]")
            except Exception:
                pass
        
        if self.model is None:
            try:
                self.model = SAC.load(self.model_path)
                self.is_recurrent = False
                console.print("[green]✓ Loaded SAC (MLP policy)[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to load model: {e}[/red]")
                raise
        
        # Load VecNormalize - we need to create an env with matching observation space
        vec_path = self.model_path.parent / "vec_normalize.pkl"
        if vec_path.exists():
            console.print("[blue]Loading VecNormalize stats...[/blue]")
            
            # First fetch forex data so we can create matching env
            await self.forex.update()
            console.print(f"  DXY available: {self.forex.has_dxy}")
            console.print(f"  EUR/USD available: {self.forex.has_eurusd}")
            
            # Load stored forex data for creating the dummy env
            # (we need data with the right shape, actual values don't matter)
            dxy_data = None
            eurusd_data = None
            
            dxy_path = self.data_dir / "dxy_1h.parquet"
            if dxy_path.exists():
                dxy_data = pd.read_parquet(dxy_path)
                if not pd.api.types.is_datetime64_any_dtype(dxy_data["timestamp"]):
                    dxy_data["timestamp"] = pd.to_datetime(dxy_data["timestamp"], utc=True)
                console.print("[green]  ✓ Loaded DXY historical data[/green]")
            
            eurusd_path = self.data_dir / "eurusd_1h.parquet"
            if eurusd_path.exists():
                eurusd_data = pd.read_parquet(eurusd_path)
                if not pd.api.types.is_datetime64_any_dtype(eurusd_data["timestamp"]):
                    eurusd_data["timestamp"] = pd.to_datetime(eurusd_data["timestamp"], utc=True)
                console.print("[green]  ✓ Loaded EUR/USD historical data[/green]")
            
            # Load BTC data for dummy env
            btc_path = self.data_dir / "btcusdt_100ms.parquet"
            if not btc_path.exists():
                btc_path = self.data_dir / "btcusdt_1s_30days.parquet"
            
            if not btc_path.exists():
                console.print("[red]✗ No BTC data found for creating dummy env[/red]")
                raise FileNotFoundError("BTC data required")
            
            btc_data = pd.read_parquet(btc_path)
            if "close" in btc_data.columns and "price" not in btc_data.columns:
                btc_data = btc_data.rename(columns={"close": "price"})
            console.print("[green]  ✓ Loaded BTC historical data[/green]")
            
            # Create dummy env with SAME configuration as training
            config = MultiAssetConfig(random_start=False)
            
            def make_env():
                return MultiAssetEnv(btc_data, dxy_data, eurusd_data, config=config)
            
            dummy_env = DummyVecEnv([make_env])
            
            console.print(f"  Observation space: {dummy_env.observation_space.shape}")
            
            self.vec_normalize = VecNormalize.load(str(vec_path), dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            console.print("[green]✓ Loaded observation normalization stats[/green]")
            
            # Store whether we have forex for observation building
            self.has_dxy = dxy_data is not None
            self.has_eurusd = eurusd_data is not None
        else:
            console.print("[yellow]! No vec_normalize.pkl found[/yellow]")
            self.has_dxy = False
            self.has_eurusd = False
    
    async def _fetch_initial_history(self, client: httpx.AsyncClient):
        """Fetch recent BTC price history."""
        console.print("[blue]Fetching BTC price history...[/blue]")
        
        params = {"symbol": "BTCUSDT", "interval": "1s", "limit": 1000}
        resp = await client.get(f"{self.BINANCE_REST}/klines", params=params)
        resp.raise_for_status()
        
        for kline in resp.json():
            self.price_history.append(float(kline[4]))
            self.volume_history.append(float(kline[5]))
        
        self.current_price = self.price_history[-1]
        console.print(f"[green]✓ Loaded {len(self.price_history)} price points[/green]")
        console.print(f"[green]  Current BTC: ${self.current_price:,.2f}[/green]")
    
    def _build_observation(self) -> np.ndarray:
        """Build observation vector matching MultiAssetEnv format (609 dims)."""
        seq_len = self.HISTORY_LENGTH
        obs_parts = []
        
        # === BTC Prices (300) ===
        prices = list(self.price_history)[-seq_len:]
        if len(prices) < seq_len:
            pad = [prices[0] if prices else self.candle_open] * (seq_len - len(prices))
            prices = pad + prices
        prices = np.array(prices, dtype=np.float32)
        
        btc_norm = (prices / self.candle_open - 1.0) * 100 if self.candle_open > 0 else np.zeros(seq_len)
        obs_parts.append(btc_norm.astype(np.float32))
        
        # === BTC Volumes (300) ===
        volumes = list(self.volume_history)[-seq_len:]
        if len(volumes) < seq_len:
            volumes = [0.0] * (seq_len - len(volumes)) + volumes
        volumes = np.array(volumes, dtype=np.float32)
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        obs_parts.append((volumes / vol_max).astype(np.float32))
        
        # === Forex Returns (4) ===
        forex = self.forex.get_returns()
        
        if self.has_dxy:
            obs_parts.append(np.array([forex["dxy_1h"], forex["dxy_4h"]], dtype=np.float32))
        
        if self.has_eurusd:
            obs_parts.append(np.array([forex["eurusd_1h"], forex["eurusd_4h"]], dtype=np.float32))
        
        # === Context Features (5) ===
        current_vs_open = (self.current_price / self.candle_open - 1.0) * 100 if self.candle_open > 0 else 0
        
        now = datetime.now(timezone.utc)
        if self.candle_end:
            remaining_secs = (self.candle_end - now).total_seconds()
            time_remaining = max(0, remaining_secs / (self.CANDLE_MINUTES * 60))
        else:
            time_remaining = 1.0
        
        hour = now.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        if len(self.price_history) > 14:
            changes = np.abs(np.diff(list(self.price_history)[-15:]))
            atr = np.mean(changes) / (self.candle_open + 1e-8) * 100
        else:
            atr = 0.0
        
        obs_parts.append(np.array([current_vs_open, time_remaining, hour_sin, hour_cos, atr], dtype=np.float32))
        
        obs = np.concatenate(obs_parts)
        obs = np.clip(obs, -100.0, 100.0).astype(np.float32)
        
        return obs.reshape(1, -1)
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize observation normalization."""
        if self.vec_normalize is not None:
            return self.vec_normalize.normalize_obs(obs)
        return obs
    
    def _start_new_candle(self):
        """Initialize a new candle period."""
        self.candle_start, self.candle_end = get_candle_boundaries(self.CANDLE_MINUTES)
        self.candle_open = self.current_price
        self.position = None
        self.episode_starts[0] = True
        
        console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[bold cyan]NEW CANDLE: {self.candle_start.strftime('%H:%M')} - {self.candle_end.strftime('%H:%M')} UTC[/bold cyan]")
        console.print(f"[cyan]Open Price: ${self.candle_open:,.2f}[/cyan]")
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]\n")
    
    def _get_model_prediction(self, obs: np.ndarray) -> tuple[float, float]:
        """Get direction and size from model."""
        obs_norm = self._normalize_obs(obs)
        
        if self.is_recurrent:
            action, self.lstm_states = self.model.predict(
                obs_norm,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True,
            )
        else:
            action, _ = self.model.predict(obs_norm, deterministic=True)
        
        self.episode_starts[0] = False
        
        direction = float(action[0][0])
        size = float(np.clip(action[0][1], 0.0, 1.0))
        
        return direction, size
    
    def _take_position(self, direction: float, size: float):
        """Take a position if signal is strong enough."""
        if self.position is not None:
            return
        
        if abs(direction) < 0.1 or size < 0.05:
            return
        
        self.position = {
            "direction": "UP" if direction > 0 else "DOWN",
            "size": size * 0.5,
            "entry_price": self.current_price,
            "entry_time": datetime.now(timezone.utc),
        }
        
        console.print(
            f"[bold yellow]📊 POSITION TAKEN: {self.position['direction']} "
            f"(Size: {self.position['size']:.1%}) @ ${self.position['entry_price']:,.2f}[/bold yellow]"
        )
    
    def _settle_candle(self):
        """Settle the candle and calculate PnL."""
        candle_close = self.current_price
        pct_move = (candle_close - self.candle_open) / self.candle_open * 100
        actual_direction = "UP" if pct_move > 0 else "DOWN"
        
        console.print(f"\n[bold]CANDLE CLOSED: ${candle_close:,.2f} ({pct_move:+.3f}%)[/bold]")
        
        if self.position is None:
            console.print("[dim]No position was taken this candle[/dim]")
            return
        
        if self.position["direction"] == actual_direction:
            pnl_pct = abs(pct_move) * self.position["size"]
            self.correct_trades += 1
            result_str = "[bold green]✓ WIN[/bold green]"
        else:
            pnl_pct = -abs(pct_move) * (self.position["size"] ** 2) * 2
            result_str = "[bold red]✗ LOSS[/bold red]"
        
        self.balance *= (1 + pnl_pct / 100)
        self.total_trades += 1
        
        trade = {
            "candle_start": self.candle_start,
            "direction": self.position["direction"],
            "actual": actual_direction,
            "size": self.position["size"],
            "pnl_pct": pnl_pct,
            "balance": self.balance,
            "correct": self.position["direction"] == actual_direction,
        }
        self.trades.append(trade)
        
        console.print(f"{result_str} | Predicted: {self.position['direction']} | Actual: {actual_direction} ({pct_move:+.3f}%)")
        console.print(f"PnL: {pnl_pct:+.3f}% | Balance: ${self.balance:,.2f}")
    
    def _print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        
        table = Table(title="Live Paper Trading", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="green")
        
        table.add_row("Time (UTC)", now.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("BTC Price", f"${self.current_price:,.2f}")
        table.add_row("Candle Open", f"${self.candle_open:,.2f}")
        
        if self.candle_end:
            remaining = (self.candle_end - now).total_seconds()
            mins, secs = divmod(int(max(0, remaining)), 60)
            table.add_row("Time Remaining", f"{mins}:{secs:02d}")
        
        if self.candle_open > 0:
            current_move = (self.current_price / self.candle_open - 1) * 100
            color = "green" if current_move > 0 else "red"
            table.add_row("Current Move", f"[{color}]{current_move:+.3f}%[/{color}]")
        
        table.add_row("─" * 15, "─" * 20)
        
        pnl = self.balance - self.initial_balance
        pnl_color = "green" if pnl >= 0 else "red"
        table.add_row("Balance", f"${self.balance:,.2f}")
        table.add_row("Total PnL", f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")
        
        if self.total_trades > 0:
            accuracy = self.correct_trades / self.total_trades
            table.add_row("Accuracy", f"{accuracy:.1%} ({self.correct_trades}/{self.total_trades})")
        
        if self.position:
            table.add_row("Position", f"[yellow]{self.position['direction']} ({self.position['size']:.1%})[/yellow]")
        else:
            table.add_row("Position", "[dim]None[/dim]")
        
        # Forex
        forex = self.forex.get_returns()
        if self.has_dxy:
            table.add_row("DXY (1h/4h)", f"{forex['dxy_1h']:+.2f}% / {forex['dxy_4h']:+.2f}%")
        if self.has_eurusd:
            table.add_row("EUR/USD", f"{forex['eurusd_1h']:+.2f}% / {forex['eurusd_4h']:+.2f}%")
        
        console.print(table)
    
    async def run(self):
        """Run live paper trading continuously."""
        await self._load_model()
        
        async with httpx.AsyncClient(timeout=30) as client:
            await self._fetch_initial_history(client)
            
            console.print("\n[bold green]═══ STARTING LIVE PAPER TRADING ═══[/bold green]")
            console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
            console.print(f"Candle Duration: {self.CANDLE_MINUTES} minutes")
            console.print(f"Observation Space: {609 if (self.has_dxy and self.has_eurusd) else 605} dims")
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")
            
            self._start_new_candle()
            
            poll_interval = 1
            
            try:
                while True:
                    now = datetime.now(timezone.utc)
                    
                    # Check if candle ended
                    if self.candle_end and now >= self.candle_end:
                        resp = await client.get(
                            f"{self.BINANCE_REST}/ticker/price",
                            params={"symbol": "BTCUSDT"}
                        )
                        self.current_price = float(resp.json()["price"])
                        
                        self._settle_candle()
                        self._start_new_candle()
                    
                    # Update forex data periodically
                    await self.forex.update()
                    
                    # Fetch current BTC price
                    try:
                        resp = await client.get(
                            f"{self.BINANCE_REST}/ticker/price",
                            params={"symbol": "BTCUSDT"}
                        )
                        resp.raise_for_status()
                        self.current_price = float(resp.json()["price"])
                        self.price_history.append(self.current_price)
                        self.volume_history.append(0)
                    except httpx.HTTPError as e:
                        console.print(f"[red]Price fetch error: {e}[/red]")
                    
                    # Get model prediction
                    if self.position is None:
                        obs = self._build_observation()
                        direction, size = self._get_model_prediction(obs)
                        self._take_position(direction, size)
                    
                    # Print status every 5 seconds
                    if int(now.timestamp()) % 5 == 0:
                        console.clear()
                        self._print_status()
                    
                    await asyncio.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]Stopping...[/bold yellow]")
        
        # Final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final trading summary."""
        console.print("\n[bold]═══ SESSION COMPLETE ═══[/bold]\n")
        
        pnl = self.balance - self.initial_balance
        pnl_pct = pnl / self.initial_balance * 100
        
        console.print(f"Initial: ${self.initial_balance:,.2f}")
        console.print(f"Final:   ${self.balance:,.2f}")
        
        color = "green" if pnl >= 0 else "red"
        console.print(f"PnL:     [{color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{color}]")
        
        if self.total_trades > 0:
            console.print(f"\nTrades: {self.total_trades}")
            console.print(f"Accuracy: {self.correct_trades / self.total_trades:.1%}")


async def main():
    parser = argparse.ArgumentParser(description="Live paper trading")
    parser.add_argument("--model", "-m", default="logs/recurrent_multi_asset/multi_asset_model")
    parser.add_argument("--data", "-d", default="./data")
    parser.add_argument("--balance", "-b", type=float, default=10000.0)
    
    args = parser.parse_args()
    
    trader = LivePaperTrader(
        model_path=args.model,
        data_dir=args.data,
        initial_balance=args.balance,
    )
    
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
