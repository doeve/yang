"""
Multi-Asset Candle Prediction Environment.

Combines BTC with correlated assets (DXY, EUR/USD) for improved predictions.
Uses asymmetric reward structure for proper position sizing.
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import structlog
from gymnasium import spaces

logger = structlog.get_logger(__name__)


@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset environment."""
    
    candle_minutes: int = 15
    price_history_length: int = 300
    max_position_size: float = 0.5
    random_start: bool = True
    
    # Assets to include (BTC is always primary)
    include_dxy: bool = True
    include_eurusd: bool = True


class MultiAssetEnv(gym.Env):
    """
    Multi-asset candle prediction environment.
    
    Observation includes:
    - BTC price history (normalized)
    - BTC volume history
    - DXY price history (normalized) - if available
    - EUR/USD price history (normalized) - if available
    - Context features (time, volatility, etc.)
    
    Action: [direction (-1 to +1), position_size (0 to 1)]
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        btc_data: pd.DataFrame,
        dxy_data: pd.DataFrame | None = None,
        eurusd_data: pd.DataFrame | None = None,
        config: MultiAssetConfig | None = None,
    ):
        super().__init__()
        
        self.config = config or MultiAssetConfig()
        self.has_dxy = dxy_data is not None and self.config.include_dxy
        self.has_eurusd = eurusd_data is not None and self.config.include_eurusd
        
        self._setup_data(btc_data, dxy_data, eurusd_data)
        self._setup_spaces()
        
        # Episode state
        self._current_candle_idx = 0
        self._current_step = 0
        self._position_taken = False
        self._position_direction = 0.0
        self._position_size = 0.0
        self._candle_open = 0.0
        self._balance = 1000.0
        
        logger.info(
            "MultiAssetEnv initialized",
            total_candles=len(self._candles),
            has_dxy=self.has_dxy,
            has_eurusd=self.has_eurusd,
        )
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.config.random_start:
            min_idx = self.config.price_history_length
            max_idx = len(self._candles) - 1
            if max_idx > min_idx:
                self._current_candle_idx = self.np_random.integers(min_idx, max_idx)
            else:
                self._current_candle_idx = min_idx
        else:
            if not hasattr(self, '_current_candle_idx'):
                 self._current_candle_idx = self.config.price_history_length
            else:
                 self._current_candle_idx = (self._current_candle_idx + 1) % len(self._candles)
        
        candle = self._candles[self._current_candle_idx]
        self._current_step = candle["start_idx"]
        self._candle_end_idx = candle["end_idx"]
        self._candle_open = candle["open"]
        self._candle_close = candle["close"] # Store for reward calc
        self._candle_return = candle["return"]
        
        self._position_taken = False
        self._position_direction = 0.0
        self._position_size = 0.0
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        
        reward = 0.0
        terminated = False
        
        # Get current price before incrementing step
        idx = min(self._current_step, len(self._btc_prices) - 1)
        current_price = self._btc_prices[idx]
        
        if not self._position_taken and abs(direction) > 0.1:
            self._position_taken = True
            self._position_direction = 1.0 if direction > 0 else -1.0
            self._position_size = size * self.config.max_position_size
            
            terminated = True
            
            # Calculate REALIZED precision (Forecast vs Reality)
            # Reward = (Exit Price - Entry Price) / Entry Price
            # Exit is Candle Close (since we hold to end of candle)
            
            move_pct = (self._candle_close - current_price) / current_price * 100
            
            # Asymmetric Reward
            if self._position_direction > 0: # BET UP
                if move_pct > 0:
                    reward = move_pct * self._position_size
                else:
                    reward = -1.0 * (self._position_size ** 2) * abs(move_pct) * 2.0
            else: # BET DOWN
                if move_pct < 0:
                    reward = abs(move_pct) * self._position_size
                else:
                    reward = -1.0 * (self._position_size ** 2) * abs(move_pct) * 2.0
                    
        else:
            self._current_step += 1
            if self._current_step >= self._candle_end_idx:
                terminated = True
                # Small penalty for indecision / missing the move
                # But don't punish too hard, sometimes waiting is correct
                reward = -0.0
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _compute_reward(self) -> float:
        # Deprecated: Logic moved to step() for access to current_price
        return 0.0

    def _setup_data(
        self, 
        btc_data: pd.DataFrame, 
        dxy_data: pd.DataFrame | None,
        eurusd_data: pd.DataFrame | None,
    ) -> None:
        """Process and align multi-asset data featuring distilled returns."""
        # Process BTC (primary asset)
        btc = btc_data.copy().sort_values("timestamp").reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(btc["timestamp"]):
            btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
        
        self._btc_timestamps = btc["timestamp"].values
        self._btc_prices = btc["price"].values.astype(np.float64)
        self._btc_volumes = btc["volume"].values.astype(np.float64) if "volume" in btc else np.ones(len(btc))
        
        # Helper to compute returns on hourly data then align
        def prepare_forex(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            
            # Compute returns on the hourly dataframe
            # 1h return = current / lag(1) - 1
            # 4h return = current / lag(4) - 1
            prices = df["price"]
            ret_1h = prices.pct_change(1).fillna(0)
            ret_4h = prices.pct_change(4).fillna(0)
            
            # Align to BTC
            # Create series indexed by timestamp
            s_1h = pd.Series(ret_1h.values, index=df["timestamp"]).sort_index()
            s_4h = pd.Series(ret_4h.values, index=df["timestamp"]).sort_index()
            
            btc_index = pd.DatetimeIndex(btc["timestamp"])
            
            # Reindex with forward fill (propagates latest hourly return)
            # This is correct: every 100ms step sees the same "last 1h return" until a new hour comes
            a_1h = s_1h.reindex(btc_index, method="ffill").fillna(0).values.astype(np.float32)
            a_4h = s_4h.reindex(btc_index, method="ffill").fillna(0).values.astype(np.float32)
            
            return a_1h, a_4h

        if self.has_dxy and dxy_data is not None:
            self._dxy_ret_1h, self._dxy_ret_4h = prepare_forex(dxy_data)
            logger.info("DXY features ready")
        else:
            self._dxy_ret_1h = None
            self._dxy_ret_4h = None
        
        if self.has_eurusd and eurusd_data is not None:
            self._eurusd_ret_1h, self._eurusd_ret_4h = prepare_forex(eurusd_data)
            logger.info("EUR/USD features ready")
        else:
            self._eurusd_ret_1h = None
            self._eurusd_ret_4h = None
        
        # Create candles from BTC data
        self._create_candles(btc)
        self._compute_atr()
    
    def _align_secondary_asset(self, *args):
        # Deprecated
        pass

    def _create_candles(self, btc: pd.DataFrame) -> None:
        """Create candle boundaries from BTC data."""
        candle_seconds = self.config.candle_minutes * 60
        btc["candle_idx"] = (btc["timestamp"].astype(np.int64) // 10**9 // candle_seconds).astype(int)
        
        candles = []
        for candle_id, group in btc.groupby("candle_idx"):
            if len(group) < 10:
                continue
            
            candles.append({
                "start_idx": group.index[0],
                "end_idx": group.index[-1],
                "open": group["price"].iloc[0],
                "close": group["price"].iloc[-1],
                "return": (group["price"].iloc[-1] - group["price"].iloc[0]) / group["price"].iloc[0],
            })
        
        self._candles = candles
    
    def _compute_atr(self, period: int = 14) -> None:
        """Compute ATR for volatility."""
        changes = np.abs(np.diff(self._btc_prices))
        self._atr = np.zeros(len(self._btc_prices))
        if len(changes) > period:
            for i in range(period, len(self._btc_prices)):
                self._atr[i] = np.mean(changes[max(0, i-period):i])
        self._atr[:period] = self._atr[period] if period < len(self._atr) else 1.0
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        seq_len = self.config.price_history_length
        
        # BTC: prices + volumes (300 + 300)
        obs_dim = seq_len + seq_len
        
        # Forex (Scalars):
        # DXY: 1h, 4h
        if self.has_dxy:
            obs_dim += 2
            
        # EUR/USD: 1h, 4h
        if self.has_eurusd:
            obs_dim += 2
        
        # Context features
        obs_dim += 5  # current_vs_open, time_remaining, hour_sin, hour_cos, atr
        
        self._obs_dim = obs_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """Build multi-asset observation with distilled features."""
        idx = min(self._current_step, len(self._btc_prices) - 1)
        seq_len = self.config.price_history_length
        
        obs_parts = []
        
        # BTC prices (normalized to candle open)
        if idx < seq_len:
            btc_prices = np.full(seq_len, self._candle_open)
            if idx > 0:
                btc_prices[seq_len - idx:] = self._btc_prices[:idx]
        else:
            btc_prices = self._btc_prices[idx - seq_len:idx]
        
        btc_norm = (btc_prices / self._candle_open - 1.0) * 100
        obs_parts.append(btc_norm.astype(np.float32))
        
        # BTC volumes
        if idx < seq_len:
            volumes = np.zeros(seq_len)
            if idx > 0:
                volumes[seq_len - idx:] = self._btc_volumes[:idx]
        else:
            volumes = self._btc_volumes[idx - seq_len:idx]
        
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        obs_parts.append((volumes / vol_max).astype(np.float32))
        
        # Forex Features (Scalars)
        # We multiply by 100 to keep scale similar to price % change
        if self._dxy_ret_1h is not None:
            dxy_1h = self._dxy_ret_1h[idx] * 100
            dxy_4h = self._dxy_ret_4h[idx] * 100
            obs_parts.append(np.array([dxy_1h, dxy_4h], dtype=np.float32))
            
        if self._eurusd_ret_1h is not None:
            eur_1h = self._eurusd_ret_1h[idx] * 100
            eur_4h = self._eurusd_ret_4h[idx] * 100
            obs_parts.append(np.array([eur_1h, eur_4h], dtype=np.float32))
        
        # Context features
        current_price = self._btc_prices[idx]
        current_vs_open = (current_price / self._candle_open - 1.0) * 100
        
        candle = self._candles[self._current_candle_idx]
        candle_length = candle["end_idx"] - candle["start_idx"]
        time_remaining = 1.0 - (idx - candle["start_idx"]) / max(candle_length, 1)
        
        timestamp = pd.Timestamp(self._btc_timestamps[idx])
        hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
        
        atr = self._atr[idx] / (self._candle_open + 1e-8) * 100
        
        obs_parts.append(np.array([current_vs_open, time_remaining, hour_sin, hour_cos, atr], dtype=np.float32))
        
        obs = np.concatenate(obs_parts)
        return np.clip(obs, -100.0, 100.0)
    
    def _get_info(self) -> dict[str, Any]:
        """Build info dictionary."""
        return {
            "candle_idx": self._current_candle_idx,
            "balance": self._balance,
            "position_taken": self._position_taken,
            "position_direction": "UP" if self._position_direction > 0 else "DOWN" if self._position_direction < 0 else None,
            "position_size": self._position_size,
            "candle_return": self._candle_return,
        }


def make_multi_asset_vec_env(
    btc_data: pd.DataFrame,
    dxy_data: pd.DataFrame | None = None,
    eurusd_data: pd.DataFrame | None = None,
    num_envs: int = 4,
    config: MultiAssetConfig | None = None,
) -> Any:
    """Create vectorized multi-asset environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        cfg = config or MultiAssetConfig()
        return MultiAssetEnv(btc_data, dxy_data, eurusd_data, config=cfg)
    
    return DummyVecEnv([make_env for _ in range(num_envs)])
