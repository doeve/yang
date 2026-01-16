"""
Binary Candle Prediction Environment.

Matches d3v's betting flow exactly:
- 15-minute candles with defined OPEN and CLOSE
- Bet anytime during the candle on direction (CLOSE > OPEN or CLOSE < OPEN)
- Reward at candle close based on prediction accuracy
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import structlog
from gymnasium import spaces

logger = structlog.get_logger(__name__)


class Action(IntEnum):
    """Available actions for the agent."""
    WAIT = 0      # Don't bet yet, observe more
    BET_UP = 1    # Predict close > open
    BET_DOWN = 2  # Predict close < open


@dataclass
class CandleEnvConfig:
    """Configuration for the candle prediction environment."""
    
    # Candle settings
    candle_minutes: int = 15  # Length of each candle
    
    # Observation settings
    price_history_length: int = 300  # ~5 min at 1 sec resolution
    
    # Reward settings
    correct_reward: float = 1.0
    wrong_reward: float = -1.0
    no_bet_reward: float = -0.1  # Small penalty for not betting
    
    # Kelly-inspired timing bonus
    early_bet_bonus: float = 1.2  # Bonus for betting early (more risk)
    late_bet_penalty: float = 0.8  # Penalty for betting late (less risk)
    early_threshold: float = 0.3   # First 30% of candle = early
    late_threshold: float = 0.7    # Last 30% of candle = late
    
    # Risk adjustment
    scale_by_move_size: bool = True  # Scale reward by price move
    max_move_multiplier: float = 2.0  # Cap the move multiplier
    
    # Episode settings
    max_steps_per_candle: int = 900  # 15 min * 60 sec
    random_start: bool = True


class CandlePredictionEnv(gym.Env):
    """
    Binary candle direction prediction environment.
    
    The agent observes price history and must predict whether the
    15-minute candle will close higher or lower than it opened.
    
    Key features:
    - Can bet at any point during the candle (WAIT action available)
    - Once bet is placed, it's locked until candle close
    - Reward only given at end of candle
    - Kelly-inspired bonus for early/confident bets
    
    Observation space:
    - Price history (normalized to candle open): [price_history_length]
    - Volume history (normalized): [price_history_length]
    - Current price vs open: [1]
    - Time remaining in candle (0-1): [1]
    - Hour sin/cos encoding: [2]
    - Recent volatility (ATR): [1]
    
    Action space:
    - 0: WAIT (don't bet yet)
    - 1: BET_UP (predict close > open)
    - 2: BET_DOWN (predict close < open)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        config: CandleEnvConfig | None = None,
    ):
        """
        Initialize the environment.
        
        Args:
            price_data: DataFrame with 'timestamp', 'price', 'volume' columns
                       at 1-second resolution
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or CandleEnvConfig()
        
        # Process data
        self._setup_data(price_data)
        
        # Setup spaces
        self._setup_spaces()
        
        # Episode state
        self._current_candle_idx: int = 0
        self._current_step: int = 0
        self._bet_placed: Action | None = None
        self._bet_step: int | None = None
        self._candle_open: float = 0.0
        self._candle_close: float = 0.0
        
        logger.info(
            "CandlePredictionEnv initialized",
            total_candles=len(self._candles),
            candle_minutes=self.config.candle_minutes,
        )
    
    def _setup_data(self, price_data: pd.DataFrame) -> None:
        """Process price data into candles."""
        df = price_data.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # Store raw data as numpy for fast access
        self._timestamps = df["timestamp"].values
        self._prices = df["price"].values.astype(np.float64)
        self._volumes = df["volume"].values.astype(np.float64) if "volume" in df else np.ones(len(df))
        
        # Compute candle boundaries
        candle_seconds = self.config.candle_minutes * 60
        
        # Group into candles
        df["candle_idx"] = (df["timestamp"].astype(np.int64) // 10**9 // candle_seconds).astype(int)
        
        candles = []
        for candle_id, group in df.groupby("candle_idx"):
            if len(group) < 10:  # Skip tiny candles
                continue
            
            start_idx = group.index[0]
            end_idx = group.index[-1]
            open_price = group["price"].iloc[0]
            close_price = group["price"].iloc[-1]
            high = group["price"].max()
            low = group["price"].min()
            
            candles.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "open": open_price,
                "close": close_price,
                "high": high,
                "low": low,
                "direction": 1 if close_price > open_price else 0,  # 1=UP, 0=DOWN
                "move_size": abs(close_price - open_price),
            })
        
        self._candles = candles
        
        # Compute ATR for volatility context
        self._compute_atr()
        
        logger.debug(f"Processed {len(self._candles)} candles from {len(df)} data points")
    
    def _compute_atr(self, period: int = 14) -> None:
        """Compute Average True Range for volatility context."""
        # Simplified ATR using price changes
        price_changes = np.abs(np.diff(self._prices))
        
        # Rolling ATR approximation
        self._atr = np.zeros(len(self._prices))
        if len(price_changes) > period:
            for i in range(period, len(self._prices)):
                self._atr[i] = np.mean(price_changes[max(0, i-period):i])
        
        # Fill early values
        self._atr[:period] = self._atr[period] if period < len(self._atr) else 1.0
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        # Observation dimensions
        price_dim = self.config.price_history_length
        volume_dim = self.config.price_history_length
        current_price_dim = 1
        time_remaining_dim = 1
        hour_encoding_dim = 2
        volatility_dim = 1
        
        self._obs_dim = price_dim + volume_dim + current_price_dim + time_remaining_dim + hour_encoding_dim + volatility_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        
        # 3 discrete actions: WAIT, BET_UP, BET_DOWN
        self.action_space = spaces.Discrete(3)
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset to start of a new candle."""
        super().reset(seed=seed)
        
        # Select candle
        if self.config.random_start:
            # Random candle, but ensure enough history
            min_candle = max(0, self.config.price_history_length // 100)  # Rough estimate
            max_candle = len(self._candles) - 1
            if max_candle > min_candle:
                self._current_candle_idx = self.np_random.integers(min_candle, max_candle)
            else:
                self._current_candle_idx = min_candle
        else:
            self._current_candle_idx = 0
        
        candle = self._candles[self._current_candle_idx]
        
        # Reset episode state
        self._current_step = candle["start_idx"]
        self._bet_placed = None
        self._bet_step = None
        self._candle_open = candle["open"]
        self._candle_close = candle["close"]
        self._candle_end_idx = candle["end_idx"]
        self._candle_direction = candle["direction"]
        self._candle_move_size = candle["move_size"]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=WAIT, 1=BET_UP, 2=BET_DOWN
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action = Action(action)
        reward = 0.0
        terminated = False
        truncated = False
        
        # Handle bet placement
        if self._bet_placed is None and action != Action.WAIT:
            self._bet_placed = action
            self._bet_step = self._current_step
        
        # Advance time
        self._current_step += 1
        
        # Check if candle is complete
        if self._current_step >= self._candle_end_idx:
            terminated = True
            reward = self._compute_reward()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self) -> float:
        """Compute reward at end of candle."""
        if self._bet_placed is None:
            # No bet placed - small penalty
            return self.config.no_bet_reward
        
        # Check if prediction was correct
        predicted_up = (self._bet_placed == Action.BET_UP)
        actual_up = (self._candle_direction == 1)
        correct = (predicted_up == actual_up)
        
        # Base reward
        if correct:
            reward = self.config.correct_reward
        else:
            reward = self.config.wrong_reward
        
        # Timing bonus (Kelly-inspired)
        candle_start = self._candles[self._current_candle_idx]["start_idx"]
        candle_length = self._candle_end_idx - candle_start
        bet_progress = (self._bet_step - candle_start) / max(candle_length, 1)
        
        if bet_progress < self.config.early_threshold:
            # Early bet - bonus for confidence/risk
            reward *= self.config.early_bet_bonus
        elif bet_progress > self.config.late_threshold:
            # Late bet - penalty for waiting too long
            reward *= self.config.late_bet_penalty
        
        # Scale by move size (risk adjustment)
        if self.config.scale_by_move_size and correct:
            atr = self._atr[self._current_step] if self._current_step < len(self._atr) else 1.0
            if atr > 0:
                move_ratio = self._candle_move_size / atr
                move_multiplier = min(move_ratio, self.config.max_move_multiplier)
                reward *= max(0.5, move_multiplier)  # At least 0.5x
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Build observation array."""
        idx = min(self._current_step, len(self._prices) - 1)
        
        # Price history (normalized to candle open)
        seq_len = self.config.price_history_length
        if idx < seq_len:
            # Pad with open price
            prices = np.full(seq_len, self._candle_open)
            prices[seq_len - idx:] = self._prices[:idx]
        else:
            prices = self._prices[idx - seq_len:idx]
        
        # Normalize to candle open
        normalized_prices = (prices / self._candle_open - 1.0) * 100  # % change
        
        # Volume history (normalized)
        if idx < seq_len:
            volumes = np.zeros(seq_len)
            volumes[seq_len - idx:] = self._volumes[:idx]
        else:
            volumes = self._volumes[idx - seq_len:idx]
        
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        normalized_volumes = volumes / vol_max
        
        # Current price vs open
        current_price = self._prices[idx]
        current_vs_open = (current_price / self._candle_open - 1.0) * 100
        
        # Time remaining in candle (0 = just started, 1 = about to close)
        candle_start = self._candles[self._current_candle_idx]["start_idx"]
        candle_length = self._candle_end_idx - candle_start
        time_progress = (idx - candle_start) / max(candle_length, 1)
        time_remaining = 1.0 - time_progress
        
        # Hour encoding (market cycle patterns)
        timestamp = pd.Timestamp(self._timestamps[idx])
        hour = timestamp.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Volatility context
        atr = self._atr[idx] if idx < len(self._atr) else 1.0
        normalized_atr = atr / (self._candle_open + 1e-8) * 100  # % of price
        
        # Concatenate all features
        obs = np.concatenate([
            normalized_prices.astype(np.float32),
            normalized_volumes.astype(np.float32),
            np.array([current_vs_open], dtype=np.float32),
            np.array([time_remaining], dtype=np.float32),
            np.array([hour_sin, hour_cos], dtype=np.float32),
            np.array([normalized_atr], dtype=np.float32),
        ])
        
        # Clip extreme values
        obs = np.clip(obs, -100.0, 100.0)
        
        return obs
    
    def _get_info(self) -> dict[str, Any]:
        """Build info dictionary."""
        idx = min(self._current_step, len(self._prices) - 1)
        
        candle_start = self._candles[self._current_candle_idx]["start_idx"]
        candle_length = self._candle_end_idx - candle_start
        progress = (idx - candle_start) / max(candle_length, 1)
        
        return {
            "candle_idx": self._current_candle_idx,
            "candle_open": self._candle_open,
            "current_price": self._prices[idx],
            "time_progress": progress,
            "bet_placed": self._bet_placed.name if self._bet_placed else None,
            "candle_direction": "UP" if self._candle_direction == 1 else "DOWN",
        }
    
    def render(self) -> None:
        """Render the environment state."""
        info = self._get_info()
        print(f"Candle {info['candle_idx']}: Open={info['candle_open']:.2f}, "
              f"Current={info['current_price']:.2f}, "
              f"Progress={info['time_progress']:.1%}, "
              f"Bet={info['bet_placed']}, "
              f"Actual={info['candle_direction']}")


def make_candle_vec_env(
    price_data: pd.DataFrame,
    num_envs: int = 4,
    config: CandleEnvConfig | None = None,
) -> Any:
    """
    Create vectorized candle prediction environments.
    
    Args:
        price_data: Price data DataFrame
        num_envs: Number of parallel environments
        config: Environment configuration
        
    Returns:
        SubprocVecEnv with parallel environments
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    def make_env_fn(seed: int):
        def _init():
            cfg = config or CandleEnvConfig()
            env = CandlePredictionEnv(price_data, config=cfg)
            env.reset(seed=seed)
            return env
        return _init
    
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)
