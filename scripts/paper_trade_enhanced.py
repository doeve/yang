#!/usr/bin/env python3
"""
Live Paper Trading for Enhanced Multi-Asset Model.

Supports the EnhancedMultiAssetEnv features:
- 500-length price/volume history
- Order flow features (volume delta, trade imbalance, large trades, etc.)
- Multi-timeframe features (1h/4h momentum)
- 3D action space with skip action
- Risk-adjusted rewards tracking

Data sources:
- Binance REST API: BTC price/volume history
- Yahoo Finance: DXY and EUR/USD forex data

Usage:
    python scripts/paper_trade_enhanced.py --model logs/my_enhanced_model/enhanced_model.zip
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
from rich.live import Live

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.simulation.enhanced_multi_asset_env import (
    EnhancedMultiAssetEnv,
    EnhancedMultiAssetConfig,
)

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
    """Streams live forex data from Yahoo Finance."""
    
    YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    YAHOO_SYMBOLS = {
        "DXY": "DX-Y.NYB",
        "EURUSD": "EURUSD=X",
    }
    
    def __init__(self):
        self.dxy_prices: deque = deque(maxlen=100)
        self.eurusd_prices: deque = deque(maxlen=100)
        self.last_fetch: Optional[datetime] = None
        self.fetch_interval = 60
    
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
            return
        
        async with aiohttp.ClientSession() as session:
            dxy = await self._fetch_yahoo(session, "DXY")
            if dxy:
                self.dxy_prices.append(dxy)
            
            eurusd = await self._fetch_yahoo(session, "EURUSD")
            if eurusd:
                self.eurusd_prices.append(eurusd)
        
        self.last_fetch = now
    
    def get_returns(self) -> dict:
        """Get 1h and 4h returns."""
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


class LiveOrderFlowTracker:
    """Tracks order flow features from live price/volume data."""
    
    def __init__(self, window: int = 100):
        self.window = window
        self.prices: deque = deque(maxlen=2000)
        self.volumes: deque = deque(maxlen=2000)
        
        # Computed features
        self.volume_delta = 0.0
        self.trade_imbalance = 0.5
        self.large_trade_ratio = 0.0
        self.volume_momentum = 0.0
        self.pv_correlation = 0.0
        self.vwap_distance = 0.0
    
    def update(self, price: float, volume: float):
        """Add new price/volume and recompute features."""
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.window:
            return
        
        prices = np.array(list(self.prices))
        volumes = np.array(list(self.volumes))
        n = len(prices)
        
        # Price changes for buy/sell detection
        price_changes = np.diff(prices, prepend=prices[0])
        buy_volume = np.where(price_changes > 0, volumes, 0)
        sell_volume = np.where(price_changes < 0, volumes, 0)
        
        # Volume Delta (last window)
        recent_buy = buy_volume[-self.window:].sum()
        recent_sell = sell_volume[-self.window:].sum()
        vd = recent_buy - recent_sell
        vd_std = np.std(buy_volume[-self.window:] - sell_volume[-self.window:]) + 1e-8
        self.volume_delta = np.clip(vd / vd_std, -5, 5)
        
        # Trade Imbalance
        total = recent_buy + recent_sell
        self.trade_imbalance = recent_buy / total if total > 0 else 0.5
        
        # Large Trade Ratio
        vol_window = volumes[-self.window:]
        threshold = np.percentile(vol_window, 90)
        self.large_trade_ratio = (vol_window > threshold).sum() / self.window
        
        # Volume Momentum
        vol_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        self.volume_momentum = np.clip((volumes[-1] / (vol_ma + 1e-8) - 1.0), -5, 5)
        
        # Price-Volume Correlation
        p = prices[-self.window:]
        v = volumes[-self.window:]
        if np.std(p) > 0 and np.std(v) > 0:
            self.pv_correlation = np.corrcoef(p, v)[0, 1]
        else:
            self.pv_correlation = 0.0
        
        # VWAP Distance
        cum_vol = np.cumsum(volumes)
        cum_pv = np.cumsum(prices * volumes)
        vwap = cum_pv[-1] / (cum_vol[-1] + 1e-8)
        self.vwap_distance = np.clip((prices[-1] - vwap) / (vwap + 1e-8) * 100, -5, 5)
    
    def get_features(self) -> np.ndarray:
        """Return order flow feature vector (6 dims)."""
        return np.array([
            self.volume_delta,
            self.trade_imbalance,
            self.large_trade_ratio,
            self.volume_momentum,
            self.pv_correlation,
            self.vwap_distance,
        ], dtype=np.float32)


class LiveMultiTimeframeTracker:
    """Tracks multi-timeframe momentum features."""
    
    def __init__(self, ticks_per_hour: int = 3600):
        self.ticks_per_hour = ticks_per_hour
        self.prices: deque = deque(maxlen=ticks_per_hour * 5)
        self.volumes: deque = deque(maxlen=ticks_per_hour * 5)
    
    def update(self, price: float, volume: float):
        """Add new price/volume."""
        self.prices.append(price)
        self.volumes.append(volume)
    
    def get_features(self) -> np.ndarray:
        """Return multi-timeframe feature vector (4 dims)."""
        prices = list(self.prices)
        volumes = list(self.volumes)
        n = len(prices)
        
        lookback_1h = min(self.ticks_per_hour, n - 1)
        lookback_4h = min(self.ticks_per_hour * 4, n - 1)
        
        # Price momentum
        momentum_1h = 0.0
        momentum_4h = 0.0
        if n > lookback_1h and lookback_1h > 0:
            momentum_1h = (prices[-1] / prices[-lookback_1h] - 1) * 100
        if n > lookback_4h and lookback_4h > 0:
            momentum_4h = (prices[-1] / prices[-lookback_4h] - 1) * 100
        
        # Volume momentum
        vol_momentum_1h = 0.0
        vol_momentum_4h = 0.0
        if n > 100:
            vol_now = np.mean(volumes[-100:])
            if n > lookback_1h + 100:
                vol_1h_ago = np.mean(volumes[-lookback_1h-100:-lookback_1h]) if lookback_1h > 0 else vol_now
                vol_momentum_1h = vol_now / (vol_1h_ago + 1e-8) - 1
            if n > lookback_4h + 100:
                vol_4h_ago = np.mean(volumes[-lookback_4h-100:-lookback_4h]) if lookback_4h > 0 else vol_now
                vol_momentum_4h = vol_now / (vol_4h_ago + 1e-8) - 1
        
        return np.array([
            np.clip(momentum_1h, -10, 10),
            np.clip(momentum_4h, -20, 20),
            np.clip(vol_momentum_1h, -5, 5),
            np.clip(vol_momentum_4h, -5, 5),
        ], dtype=np.float32)


class EnhancedLivePaperTrader:
    """
    Live paper trading for EnhancedMultiAssetEnv models.
    
    Observation space matches training env:
    - BTC prices (500) normalized to candle open
    - BTC volumes (500) normalized
    - Order flow features (6)
    - Multi-timeframe features (4)
    - DXY returns (2)
    - EUR/USD returns (2)
    - Context features (7)
    
    Action space: [direction, size, skip_prob]
    """
    
    BINANCE_REST = "https://api.binance.com/api/v3"
    CANDLE_MINUTES = 15
    HISTORY_LENGTH = 100  # Must match EnhancedMultiAssetConfig.price_history_length
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "./data",
        initial_balance: float = 10000.0,
        candle_minutes: int = 15,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.CANDLE_MINUTES = candle_minutes
        
        # BTC price/volume history
        self.price_history: deque = deque(maxlen=3000)
        self.volume_history: deque = deque(maxlen=3000)
        self.current_price = 0.0
        
        # Feature trackers
        self.forex = LiveForexStreamer()
        self.order_flow = LiveOrderFlowTracker()
        self.multi_tf = LiveMultiTimeframeTracker(ticks_per_hour=3600)
        
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
        self.position_entry_time_pct = 0.0
        
        # Config flags
        self.has_dxy = True
        self.has_eurusd = True
        self.include_order_flow = True
        self.include_multi_tf = True
        
        # Stats
        self.trades: list[dict] = []
        self.total_trades = 0
        self.correct_trades = 0
        self.skipped_candles = 0
    
    async def _load_model(self):
        """Load trained model and VecNormalize."""
        console.print(f"\n[bold blue]Loading model from {self.model_path}...[/bold blue]")
        
        # Check if path exists (with or without .zip)
        actual_path = self.model_path
        if not actual_path.exists() and not str(actual_path).endswith('.zip'):
            zip_path = Path(str(actual_path) + '.zip')
            if zip_path.exists():
                actual_path = zip_path
        
        if not actual_path.exists() and not Path(str(self.model_path).replace('.zip', '')).exists():
            console.print(f"[red]✗ Model not found: {self.model_path}[/red]")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Try RecurrentPPO first
        if RecurrentPPO:
            try:
                self.model = RecurrentPPO.load(str(self.model_path).replace('.zip', ''))
                self.is_recurrent = True
                console.print("[green]✓ Loaded RecurrentPPO (LSTM policy)[/green]")
            except Exception:
                pass
        
        if self.model is None:
            try:
                self.model = SAC.load(str(self.model_path).replace('.zip', ''))
                self.is_recurrent = False
                console.print("[green]✓ Loaded SAC (MLP policy)[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to load model: {e}[/red]")
                raise
        
        # Load VecNormalize
        vec_path = self.model_path.parent / "vec_normalize.pkl"
        if vec_path.exists():
            console.print("[blue]Loading VecNormalize stats...[/blue]")
            
            await self.forex.update()
            
            # Load historical data for dummy env
            dxy_data = None
            eurusd_data = None
            
            dxy_path = self.data_dir / "dxy_1h.parquet"
            if dxy_path.exists():
                dxy_data = pd.read_parquet(dxy_path)
                console.print("[green]  ✓ Loaded DXY historical data[/green]")
            else:
                self.has_dxy = False
            
            eurusd_path = self.data_dir / "eurusd_1h.parquet"
            if eurusd_path.exists():
                eurusd_data = pd.read_parquet(eurusd_path)
                console.print("[green]  ✓ Loaded EUR/USD historical data[/green]")
            else:
                self.has_eurusd = False
            
            btc_path = self.data_dir / "btcusdt_100ms.parquet"
            if not btc_path.exists():
                btc_path = self.data_dir / "btcusdt_1s_30days.parquet"
            
            if not btc_path.exists():
                console.print("[red]✗ No BTC data found[/red]")
                raise FileNotFoundError("BTC data required")
            
            btc_data = pd.read_parquet(btc_path)
            if "close" in btc_data.columns and "price" not in btc_data.columns:
                btc_data = btc_data.rename(columns={"close": "price"})
            console.print("[green]  ✓ Loaded BTC historical data[/green]")
            
            # Create dummy env matching training config
            config = EnhancedMultiAssetConfig(
                random_start=False,
                include_order_flow=self.include_order_flow,
                include_multi_timeframe=self.include_multi_tf,
                enable_skip=True,
            )
            
            def make_env():
                return EnhancedMultiAssetEnv(btc_data, dxy_data, eurusd_data, config=config)
            
            dummy_env = DummyVecEnv([make_env])
            console.print(f"  Observation space: {dummy_env.observation_space.shape}")
            
            self.vec_normalize = VecNormalize.load(str(vec_path), dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            console.print("[green]✓ Loaded observation normalization stats[/green]")
        else:
            console.print("[yellow]! No vec_normalize.pkl found[/yellow]")
    
    async def _fetch_initial_history(self, client: httpx.AsyncClient):
        """Fetch recent BTC price history."""
        console.print("[blue]Fetching BTC price history...[/blue]")
        
        # Fetch 1s klines for history
        params = {"symbol": "BTCUSDT", "interval": "1s", "limit": 1000}
        resp = await client.get(f"{self.BINANCE_REST}/klines", params=params)
        resp.raise_for_status()
        
        for kline in resp.json():
            price = float(kline[4])  # Close price
            volume = float(kline[5])  # Volume
            self.price_history.append(price)
            self.volume_history.append(volume)
            self.order_flow.update(price, volume)
            self.multi_tf.update(price, volume)
        
        self.current_price = self.price_history[-1]
        console.print(f"[green]✓ Loaded {len(self.price_history)} price points[/green]")
        console.print(f"[green]  Current BTC: ${self.current_price:,.2f}[/green]")
    
    def _build_observation(self) -> np.ndarray:
        """Build observation vector matching EnhancedMultiAssetEnv format."""
        seq_len = self.HISTORY_LENGTH
        obs_parts = []
        
        # === BTC Prices (500) ===
        prices = list(self.price_history)[-seq_len:]
        if len(prices) < seq_len:
            pad = [prices[0] if prices else self.candle_open] * (seq_len - len(prices))
            prices = pad + prices
        prices = np.array(prices, dtype=np.float32)
        
        btc_norm = (prices / self.candle_open - 1.0) * 100 if self.candle_open > 0 else np.zeros(seq_len)
        obs_parts.append(btc_norm.astype(np.float32))
        
        # === BTC Volumes (500) ===
        volumes = list(self.volume_history)[-seq_len:]
        if len(volumes) < seq_len:
            volumes = [0.0] * (seq_len - len(volumes)) + volumes
        volumes = np.array(volumes, dtype=np.float32)
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        obs_parts.append((volumes / vol_max).astype(np.float32))
        
        # === Order Flow Features (6) ===
        if self.include_order_flow:
            obs_parts.append(self.order_flow.get_features())
        
        # === Multi-Timeframe Features (4) ===
        if self.include_multi_tf:
            obs_parts.append(self.multi_tf.get_features())
        
        # === Forex Returns ===
        forex = self.forex.get_returns()
        
        if self.has_dxy:
            obs_parts.append(np.array([forex["dxy_1h"], forex["dxy_4h"]], dtype=np.float32))
        
        if self.has_eurusd:
            obs_parts.append(np.array([forex["eurusd_1h"], forex["eurusd_4h"]], dtype=np.float32))
        
        # === Context Features (7) ===
        current_vs_open = (self.current_price / self.candle_open - 1.0) * 100 if self.candle_open > 0 else 0
        
        now = datetime.now(timezone.utc)
        if self.candle_end:
            total_secs = self.CANDLE_MINUTES * 60
            elapsed_secs = (now - self.candle_start).total_seconds() if self.candle_start else 0
            time_in_candle = min(elapsed_secs / total_secs, 1.0)
            time_remaining = 1.0 - time_in_candle
        else:
            time_remaining = 1.0
            time_in_candle = 0.0
        
        hour = now.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # ATR
        if len(self.price_history) > 14:
            changes = np.abs(np.diff(list(self.price_history)[-15:]))
            atr = np.mean(changes) / (self.candle_open + 1e-8) * 100
        else:
            atr = 0.0
        
        # Volatility regime
        if len(self.price_history) > 100:
            recent_changes = np.abs(np.diff(list(self.price_history)[-100:]))
            vol_regime = (np.mean(recent_changes[-14:]) - np.mean(recent_changes)) / (np.std(recent_changes) + 1e-8)
        else:
            vol_regime = 0.0
        
        context = np.array([
            current_vs_open,
            time_remaining,
            time_in_candle,
            hour_sin,
            hour_cos,
            atr,
            np.clip(vol_regime, -3, 3),
        ], dtype=np.float32)
        obs_parts.append(context)
        
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
        self.position_entry_time_pct = 0.0
        self.episode_starts[0] = True
        
        console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[bold cyan]NEW CANDLE: {self.candle_start.strftime('%H:%M')} - {self.candle_end.strftime('%H:%M')} UTC[/bold cyan]")
        console.print(f"[cyan]Open Price: ${self.candle_open:,.2f}[/cyan]")
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]\n")
    
    def _get_model_prediction(self, obs: np.ndarray) -> tuple[float, float, float]:
        """Get direction, size, and skip_prob from model."""
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
        skip_prob = float(action[0][2]) if len(action[0]) > 2 else 0.0
        
        return direction, size, skip_prob
    
    def _take_position(self, direction: float, size: float, skip_prob: float) -> bool:
        """Take a position if signal is strong enough. Returns True if position taken or skipped."""
        if self.position is not None:
            return False
        
        # Check skip action
        if skip_prob > 0.5:
            console.print(f"[dim]⏭ SKIP: Model chose to skip (skip_prob={skip_prob:.2f})[/dim]")
            self.position = {"skipped": True}
            self.skipped_candles += 1
            return True
        
        if abs(direction) < 0.1 or size < 0.05:
            return False
        
        # Calculate entry time
        now = datetime.now(timezone.utc)
        if self.candle_start and self.candle_end:
            elapsed = (now - self.candle_start).total_seconds()
            total = (self.candle_end - self.candle_start).total_seconds()
            self.position_entry_time_pct = elapsed / total if total > 0 else 0
        
        self.position = {
            "direction": "UP" if direction > 0 else "DOWN",
            "size": size * 0.5,  # Max 50% position
            "entry_price": self.current_price,
            "entry_time": now,
            "entry_time_pct": self.position_entry_time_pct,
            "skipped": False,
        }
        
        console.print(
            f"[bold yellow]📊 POSITION TAKEN: {self.position['direction']} "
            f"(Size: {self.position['size']:.1%}, Entry: {self.position_entry_time_pct:.0%} into candle) "
            f"@ ${self.position['entry_price']:,.2f}[/bold yellow]"
        )
        return True
    
    def _settle_candle(self):
        """Settle the candle and calculate PnL."""
        candle_close = self.current_price
        pct_move = (candle_close - self.candle_open) / self.candle_open * 100
        actual_direction = "UP" if pct_move > 0 else "DOWN"
        
        console.print(f"\n[bold]CANDLE CLOSED: ${candle_close:,.2f} ({pct_move:+.3f}%)[/bold]")
        
        if self.position is None:
            console.print("[dim]No position was taken this candle[/dim]")
            return
        
        if self.position.get("skipped"):
            # Evaluate skip decision
            if abs(pct_move) < 0.1:
                console.print("[green]⏭ Good skip - small move[/green]")
            else:
                console.print(f"[yellow]⏭ Skipped - missed {abs(pct_move):.2f}% move[/yellow]")
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
            "entry_time_pct": self.position.get("entry_time_pct", 0),
            "pnl_pct": pnl_pct,
            "balance": self.balance,
            "correct": self.position["direction"] == actual_direction,
        }
        self.trades.append(trade)
        
        console.print(f"{result_str} | Predicted: {self.position['direction']} | Actual: {actual_direction} ({pct_move:+.3f}%)")
        console.print(f"PnL: {pnl_pct:+.3f}% | Balance: ${self.balance:,.2f}")
    
    def _build_status_table(self) -> Table:
        """Build status table for display."""
        now = datetime.now(timezone.utc)
        
        table = Table(title="Enhanced Live Paper Trading", show_header=True, header_style="bold cyan")
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
        
        table.add_row("Skipped Candles", str(self.skipped_candles))
        
        if self.position and not self.position.get("skipped"):
            table.add_row("Position", f"[yellow]{self.position['direction']} ({self.position['size']:.1%})[/yellow]")
        elif self.position and self.position.get("skipped"):
            table.add_row("Position", "[dim]Skipped[/dim]")
        else:
            table.add_row("Position", "[dim]None[/dim]")
        
        table.add_row("─" * 15, "─" * 20)
        
        # Order flow features
        of = self.order_flow
        table.add_row("Vol Delta", f"{of.volume_delta:+.2f}")
        table.add_row("Trade Imbalance", f"{of.trade_imbalance:.2f}")
        table.add_row("Large Trade %", f"{of.large_trade_ratio:.1%}")
        
        # Forex
        forex = self.forex.get_returns()
        if self.has_dxy:
            table.add_row("DXY (1h/4h)", f"{forex['dxy_1h']:+.2f}% / {forex['dxy_4h']:+.2f}%")
        if self.has_eurusd:
            table.add_row("EUR/USD", f"{forex['eurusd_1h']:+.2f}% / {forex['eurusd_4h']:+.2f}%")
        
        return table
    
    async def run(self):
        """Run live paper trading continuously."""
        await self._load_model()
        
        async with httpx.AsyncClient(timeout=30) as client:
            await self._fetch_initial_history(client)
            
            console.print("\n[bold green]═══ STARTING ENHANCED LIVE PAPER TRADING ═══[/bold green]")
            console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
            console.print(f"Candle Duration: {self.CANDLE_MINUTES} minutes")
            console.print(f"History Length: {self.HISTORY_LENGTH}")
            console.print(f"Features: Order Flow={self.include_order_flow}, Multi-TF={self.include_multi_tf}")
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
                        
                        # Also get recent trade volume (approximate)
                        volume = 0.1  # Placeholder - could use trades endpoint
                        
                        self.price_history.append(self.current_price)
                        self.volume_history.append(volume)
                        self.order_flow.update(self.current_price, volume)
                        self.multi_tf.update(self.current_price, volume)
                        
                    except httpx.HTTPError as e:
                        console.print(f"[red]Price fetch error: {e}[/red]")
                    
                    # Get model prediction and maybe take position
                    if self.position is None:
                        obs = self._build_observation()
                        direction, size, skip_prob = self._get_model_prediction(obs)
                        self._take_position(direction, size, skip_prob)
                    
                    # Print status every 5 seconds
                    if int(now.timestamp()) % 5 == 0:
                        console.clear()
                        console.print(self._build_status_table())
                    
                    await asyncio.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]Stopping...[/bold yellow]")
        
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
        
        console.print(f"Skipped Candles: {self.skipped_candles}")


class HistoricalBacktester:
    """
    Backtester that fetches random historical data from Binance and tests the model.
    
    This allows testing on unseen data that wasn't in the training set.
    """
    
    BINANCE_REST = "https://api.binance.com/api/v3"
    HISTORY_LENGTH = 100  # Must match EnhancedMultiAssetConfig.price_history_length
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "./data",
        initial_balance: float = 10000.0,
        candle_minutes: int = 15,
        num_candles: int = 100,
        days_back_min: int = 30,
        days_back_max: int = 365,
        use_polymarket_outcomes: bool = False,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.candle_minutes = candle_minutes
        self.num_candles = num_candles
        self.days_back_min = days_back_min
        self.days_back_max = days_back_max
        self.use_polymarket_outcomes = use_polymarket_outcomes
        
        # Polymarket candle data
        self.polymarket_candles = None
        if use_polymarket_outcomes:
            polymarket_path = self.data_dir / "polymarket" / "btc_15min_candles.parquet"
            if polymarket_path.exists():
                self.polymarket_candles = pd.read_parquet(polymarket_path)
                console.print(f"[green]✓ Loaded {len(self.polymarket_candles)} Polymarket candles[/green]")
            else:
                console.print(f"[yellow]⚠ Polymarket candles not found at {polymarket_path}[/yellow]")
                console.print("[yellow]  Run: yang fetch-btc-candles --hours-back 168[/yellow]")
                self.use_polymarket_outcomes = False
        
        # Model state
        self.model = None
        self.is_recurrent = False
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.vec_normalize = None
        
        # Config flags
        self.has_dxy = True
        self.has_eurusd = True
        self.include_order_flow = True
        self.include_multi_tf = True
        
        # Feature trackers
        self.order_flow = LiveOrderFlowTracker()
        self.multi_tf = LiveMultiTimeframeTracker(ticks_per_hour=60)  # 1m data = 60 ticks/hour
        
        # Stats
        self.trades: list[dict] = []
        self.total_trades = 0
        self.correct_trades = 0
        self.skipped_candles = 0

    
    async def _load_model(self):
        """Load trained model and VecNormalize."""
        console.print(f"\n[bold blue]Loading model from {self.model_path}...[/bold blue]")
        
        # Check path
        actual_path = self.model_path
        if not actual_path.exists() and not str(actual_path).endswith('.zip'):
            zip_path = Path(str(actual_path) + '.zip')
            if zip_path.exists():
                actual_path = zip_path
        
        # Try loading
        if RecurrentPPO:
            try:
                self.model = RecurrentPPO.load(str(self.model_path).replace('.zip', ''))
                self.is_recurrent = True
                console.print("[green]✓ Loaded RecurrentPPO (LSTM policy)[/green]")
            except Exception:
                pass
        
        if self.model is None:
            try:
                self.model = SAC.load(str(self.model_path).replace('.zip', ''))
                self.is_recurrent = False
                console.print("[green]✓ Loaded SAC (MLP policy)[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to load model: {e}[/red]")
                raise
        
        # Load VecNormalize
        vec_path = self.model_path.parent / "vec_normalize.pkl"
        if vec_path.exists():
            console.print("[blue]Loading VecNormalize stats...[/blue]")
            
            dxy_data = None
            eurusd_data = None
            
            dxy_path = self.data_dir / "dxy_1h.parquet"
            if dxy_path.exists():
                dxy_data = pd.read_parquet(dxy_path)
            else:
                self.has_dxy = False
            
            eurusd_path = self.data_dir / "eurusd_1h.parquet"
            if eurusd_path.exists():
                eurusd_data = pd.read_parquet(eurusd_path)
            else:
                self.has_eurusd = False
            
            btc_path = self.data_dir / "btcusdt_100ms.parquet"
            if not btc_path.exists():
                btc_path = self.data_dir / "btcusdt_1s_30days.parquet"
            
            if not btc_path.exists():
                console.print("[red]✗ No BTC data found for VecNormalize[/red]")
                raise FileNotFoundError("BTC data required")
            
            btc_data = pd.read_parquet(btc_path)
            if "close" in btc_data.columns and "price" not in btc_data.columns:
                btc_data = btc_data.rename(columns={"close": "price"})
            
            config = EnhancedMultiAssetConfig(
                random_start=False,
                include_order_flow=self.include_order_flow,
                include_multi_timeframe=self.include_multi_tf,
                enable_skip=True,
            )
            
            def make_env():
                return EnhancedMultiAssetEnv(btc_data, dxy_data, eurusd_data, config=config)
            
            dummy_env = DummyVecEnv([make_env])
            self.vec_normalize = VecNormalize.load(str(vec_path), dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            console.print("[green]✓ Loaded VecNormalize[/green]")
    
    async def _fetch_historical_data(self, client: httpx.AsyncClient) -> pd.DataFrame:
        """Fetch historical 1m kline data for a random time period."""
        console.print("\n[blue]Fetching historical data from Binance...[/blue]")
        
        # Select random start time
        now = datetime.now(timezone.utc)
        days_back = np.random.randint(self.days_back_min, self.days_back_max + 1)
        
        # Calculate how much data we need
        # For N candles of M minutes, plus 500 history, plus buffer
        minutes_needed = (self.num_candles + 10) * self.candle_minutes + self.HISTORY_LENGTH * 2
        
        # Random start within the range
        max_start = now - timedelta(days=self.days_back_min)
        min_start = now - timedelta(days=self.days_back_max) 
        
        range_seconds = (max_start - min_start).total_seconds()
        random_offset = np.random.randint(0, max(1, int(range_seconds)))
        start_time = min_start + timedelta(seconds=random_offset)
        end_time = start_time + timedelta(minutes=minutes_needed)
        
        console.print(f"  Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        console.print(f"  ({days_back} days ago, ~{minutes_needed} minutes of data)")
        
        # Fetch in chunks (Binance limit is 1000)
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        while current_start < end_ms:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            
            try:
                resp = await client.get(f"{self.BINANCE_REST}/klines", params=params)
                resp.raise_for_status()
                klines = resp.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_start = int(klines[-1][0]) + 60000  # Next minute
                
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                console.print(f"[red]Error fetching klines: {e}[/red]")
                break
        
        if not all_klines:
            raise ValueError("No historical data fetched")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["price"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df[["timestamp", "price", "volume"]].copy()
        
        console.print(f"[green]✓ Fetched {len(df)} data points[/green]")
        console.print(f"  Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
        
        return df
    
    def _build_observation(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        candle_open: float,
        time_in_candle: float,
        hour: int,
    ) -> np.ndarray:
        """Build observation vector from historical data."""
        seq_len = self.HISTORY_LENGTH
        obs_parts = []
        
        # === BTC Prices (500) ===
        if len(prices) < seq_len:
            pad = np.full(seq_len - len(prices), prices[0] if len(prices) > 0 else candle_open)
            prices = np.concatenate([pad, prices])
        else:
            prices = prices[-seq_len:]
        
        btc_norm = (prices / candle_open - 1.0) * 100 if candle_open > 0 else np.zeros(seq_len)
        obs_parts.append(btc_norm.astype(np.float32))
        
        # === BTC Volumes (500) ===
        if len(volumes) < seq_len:
            pad = np.zeros(seq_len - len(volumes))
            volumes = np.concatenate([pad, volumes])
        else:
            volumes = volumes[-seq_len:]
        
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        obs_parts.append((volumes / vol_max).astype(np.float32))
        
        # === Order Flow Features (6) ===
        if self.include_order_flow:
            obs_parts.append(self.order_flow.get_features())
        
        # === Multi-Timeframe Features (4) ===
        if self.include_multi_tf:
            obs_parts.append(self.multi_tf.get_features())
        
        # === Forex Returns (placeholder zeros - not available for historical) ===
        if self.has_dxy:
            obs_parts.append(np.array([0.0, 0.0], dtype=np.float32))
        if self.has_eurusd:
            obs_parts.append(np.array([0.0, 0.0], dtype=np.float32))
        
        # === Context Features (7) ===
        current_price = prices[-1] if len(prices) > 0 else candle_open
        current_vs_open = (current_price / candle_open - 1.0) * 100 if candle_open > 0 else 0
        time_remaining = 1.0 - time_in_candle
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # ATR
        if len(prices) > 14:
            changes = np.abs(np.diff(prices[-15:]))
            atr = np.mean(changes) / (candle_open + 1e-8) * 100
        else:
            atr = 0.0
        
        # Volatility regime
        if len(prices) > 100:
            recent_changes = np.abs(np.diff(prices[-100:]))
            vol_regime = (np.mean(recent_changes[-14:]) - np.mean(recent_changes)) / (np.std(recent_changes) + 1e-8)
        else:
            vol_regime = 0.0
        
        context = np.array([
            current_vs_open,
            time_remaining,
            time_in_candle,
            hour_sin,
            hour_cos,
            atr,
            np.clip(vol_regime, -3, 3),
        ], dtype=np.float32)
        obs_parts.append(context)
        
        obs = np.concatenate(obs_parts)
        return np.clip(obs, -100.0, 100.0).astype(np.float32).reshape(1, -1)
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize observation normalization."""
        if self.vec_normalize is not None:
            return self.vec_normalize.normalize_obs(obs)
        return obs
    
    def _get_model_prediction(self, obs: np.ndarray) -> tuple[float, float, float]:
        """Get direction, size, and skip_prob from model."""
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
        skip_prob = float(action[0][2]) if len(action[0]) > 2 else 0.0
        
        return direction, size, skip_prob
    
    async def run(self):
        """Run historical backtest."""
        await self._load_model()
        
        async with httpx.AsyncClient(timeout=60) as client:
            df = await self._fetch_historical_data(client)
        
        console.print("\n[bold green]═══ STARTING HISTORICAL BACKTEST ═══[/bold green]")
        console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
        console.print(f"Candle Duration: {self.candle_minutes} minutes")
        console.print(f"Target Candles: {self.num_candles}")
        
        # Group data into candles
        candle_seconds = self.candle_minutes * 60
        df["candle_idx"] = (df["timestamp"].astype(np.int64) // 10**9 // candle_seconds).astype(int)
        
        candle_groups = list(df.groupby("candle_idx"))
        
        # Skip first few candles to build up history
        start_idx = max(0, self.HISTORY_LENGTH // self.candle_minutes + 5)
        end_idx = min(len(candle_groups), start_idx + self.num_candles)
        
        console.print(f"Processing candles {start_idx} to {end_idx} ({end_idx - start_idx} candles)\n")
        
        # Initialize order flow with early data
        for _, group in candle_groups[:start_idx]:
            for _, row in group.iterrows():
                self.order_flow.update(row["price"], row["volume"])
                self.multi_tf.update(row["price"], row["volume"])
        
        # Collect all prices/volumes for history
        all_prices = df["price"].values
        all_volumes = df["volume"].values
        
        # Process each candle
        candles_processed = 0
        
        for candle_idx, (_, group) in enumerate(candle_groups[start_idx:end_idx]):
            if len(group) < 2:
                continue
            
            candle_open = group["price"].iloc[0]
            candle_close = group["price"].iloc[-1]
            candle_return = (candle_close - candle_open) / candle_open * 100
            candle_time = group["timestamp"].iloc[0]
            hour = candle_time.hour
            
            # Determine actual outcome - use Polymarket if available
            if self.use_polymarket_outcomes and self.polymarket_candles is not None:
                # Find matching Polymarket candle by timestamp
                candle_ts = int(candle_time.timestamp())
                # Round to 15-min boundary
                candle_ts_rounded = (candle_ts // 900) * 900
                
                # Handle both int and datetime timestamp columns
                def get_ts(x):
                    if isinstance(x, (int, float)):
                        return int(x)
                    else:
                        return int(x.timestamp())
                
                polymarket_ts = self.polymarket_candles["timestamp"].apply(get_ts) // 900 * 900
                matching = self.polymarket_candles[polymarket_ts == candle_ts_rounded]
                
                if len(matching) > 0:
                    polymarket_outcome = matching.iloc[0]["up_won"]
                    actual_direction = "UP" if polymarket_outcome else "DOWN"
                    # Also note if Polymarket disagrees with BTC move
                    btc_direction = "UP" if candle_return > 0 else "DOWN"
                    if actual_direction != btc_direction:
                        console.print(f"[dim]Note: Polymarket={actual_direction} vs BTC={btc_direction}[/dim]")
                else:
                    # Fall back to BTC direction
                    actual_direction = "UP" if candle_return > 0 else "DOWN"
            else:
                actual_direction = "UP" if candle_return > 0 else "DOWN"


            
            # Reset episode for new candle
            self.episode_starts[0] = True
            
            # Get all data up to this candle for history
            candle_end_in_df = group.index[-1]
            history_prices = all_prices[:candle_end_in_df]
            history_volumes = all_volumes[:candle_end_in_df]
            
            # Simulate decision point (middle of candle)
            mid_point = len(group) // 2
            decision_time = mid_point / len(group)
            
            # Update order flow with candle data up to decision point
            for i in range(mid_point):
                row = group.iloc[i]
                self.order_flow.update(row["price"], row["volume"])
                self.multi_tf.update(row["price"], row["volume"])
            
            # Build observation at decision point
            prices_at_decision = np.concatenate([history_prices[:-len(group)+mid_point], group["price"].iloc[:mid_point].values])
            volumes_at_decision = np.concatenate([history_volumes[:-len(group)+mid_point], group["volume"].iloc[:mid_point].values])
            
            obs = self._build_observation(
                prices_at_decision[-self.HISTORY_LENGTH*2:],
                volumes_at_decision[-self.HISTORY_LENGTH*2:],
                candle_open,
                decision_time,
                hour,
            )
            
            # Get model prediction
            direction, size, skip_prob = self._get_model_prediction(obs)
            
            # Update order flow with rest of candle
            for i in range(mid_point, len(group)):
                row = group.iloc[i]
                self.order_flow.update(row["price"], row["volume"])
                self.multi_tf.update(row["price"], row["volume"])
            
            # Evaluate trade
            if skip_prob > 0.5:
                self.skipped_candles += 1
                if candles_processed < 20:
                    console.print(f"[dim]Candle {candles_processed + 1}: SKIPPED (skip_prob={skip_prob:.2f}) | Actual: {actual_direction} ({candle_return:+.2f}%)[/dim]")
            elif abs(direction) > 0.1 and size > 0.05:
                predicted = "UP" if direction > 0 else "DOWN"
                position_size = size * 0.5
                
                if predicted == actual_direction:
                    pnl_pct = abs(candle_return) * position_size
                    self.correct_trades += 1
                    result = "[green]✓[/green]"
                else:
                    pnl_pct = -abs(candle_return) * (position_size ** 2) * 2
                    result = "[red]✗[/red]"
                
                self.balance *= (1 + pnl_pct / 100)
                self.total_trades += 1
                
                self.trades.append({
                    "time": candle_time,
                    "predicted": predicted,
                    "actual": actual_direction,
                    "size": position_size,
                    "pnl_pct": pnl_pct,
                    "balance": self.balance,
                })
                
                if candles_processed < 20:
                    console.print(
                        f"Candle {candles_processed + 1}: {predicted} (size={position_size:.1%}) → "
                        f"{actual_direction} ({candle_return:+.2f}%) | PnL: {pnl_pct:+.2f}% {result}"
                    )
            else:
                if candles_processed < 20:
                    console.print(f"[dim]Candle {candles_processed + 1}: No position (weak signal)[/dim]")
            
            candles_processed += 1
        
        self._print_summary()
    
    def _print_summary(self):
        """Print backtest summary."""
        console.print("\n" + "═" * 50)
        console.print("[bold]BACKTEST RESULTS[/bold]")
        console.print("═" * 50)
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value", style="green")
        
        pnl = self.balance - self.initial_balance
        pnl_pct = pnl / self.initial_balance * 100
        pnl_color = "green" if pnl >= 0 else "red"
        
        table.add_row("Initial Balance", f"${self.initial_balance:,.2f}")
        table.add_row("Final Balance", f"${self.balance:,.2f}")
        table.add_row("Total PnL", f"[{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]")
        
        table.add_row("─" * 15, "─" * 20)
        
        table.add_row("Total Trades", str(self.total_trades))
        if self.total_trades > 0:
            accuracy = self.correct_trades / self.total_trades
            table.add_row("Correct Trades", str(self.correct_trades))
            table.add_row("Accuracy", f"{accuracy:.1%}")
        
        table.add_row("Skipped Candles", str(self.skipped_candles))
        
        if self.trades:
            pnls = [t["pnl_pct"] for t in self.trades]
            table.add_row("─" * 15, "─" * 20)
            table.add_row("Best Trade", f"+{max(pnls):.2f}%")
            table.add_row("Worst Trade", f"{min(pnls):.2f}%")
            table.add_row("Avg Trade", f"{np.mean(pnls):+.2f}%")
        
        console.print(table)


async def main():
    parser = argparse.ArgumentParser(
        description="Enhanced paper trading (live or historical backtest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live trading
  python scripts/paper_trade_enhanced.py --model logs/my_enhanced_model/enhanced_model.zip
  
  # Backtest on random historical period
  python scripts/paper_trade_enhanced.py --backtest --model logs/my_enhanced_model/enhanced_model.zip
  
  # Backtest with custom settings
  python scripts/paper_trade_enhanced.py --backtest --num-candles 200 --days-back-min 60 --days-back-max 180
"""
    )
    parser.add_argument("--model", "-m", default="logs/my_enhanced_model/enhanced_model")
    parser.add_argument("--data", "-d", default="./data")
    parser.add_argument("--balance", "-b", type=float, default=10000.0)
    parser.add_argument("--candle-minutes", "-c", type=int, default=15)
    
    # Backtest-specific arguments
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest instead of live trading")
    parser.add_argument("--num-candles", "-n", type=int, default=100, help="Number of candles to backtest")
    parser.add_argument("--days-back-min", type=int, default=30, help="Minimum days back for random period")
    parser.add_argument("--days-back-max", type=int, default=365, help="Maximum days back for random period")
    parser.add_argument("--polymarket", action="store_true", help="Use real Polymarket outcomes instead of BTC direction")
    parser.add_argument("--sac-model", type=str, default=None, help="Path to SAC model (alternative to --model)")
    
    args = parser.parse_args()
    
    # Use SAC model if specified
    model_path = args.sac_model if args.sac_model else args.model
    
    if args.backtest:
        backtester = HistoricalBacktester(
            model_path=model_path,
            data_dir=args.data,
            initial_balance=args.balance,
            candle_minutes=args.candle_minutes,
            num_candles=args.num_candles,
            days_back_min=args.days_back_min,
            days_back_max=args.days_back_max,
            use_polymarket_outcomes=args.polymarket,
        )
        await backtester.run()

    else:
        trader = EnhancedLivePaperTrader(
            model_path=args.model,
            data_dir=args.data,
            initial_balance=args.balance,
            candle_minutes=args.candle_minutes,
        )
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())

