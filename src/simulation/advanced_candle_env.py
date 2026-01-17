"""
Advanced Candle Prediction Environment with Position Sizing.

The model learns BOTH:
1. Direction (UP/DOWN/WAIT)
2. Position size (0.0 - 1.0 of available capital)

Uses Box action space for continuous bet sizing.
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
class AdvancedCandleConfig:
    """Configuration for the advanced candle prediction environment."""
    
    # Candle settings
    candle_minutes: int = 15
    
    # Observation settings
    price_history_length: int = 300
    
    # Portfolio settings
    initial_balance: float = 1000.0
    max_position_size: float = 0.5  # Max 50% of portfolio per trade
    
    # Reward settings
    # Reward = position_size * price_move * direction_correctness
    # This naturally teaches Kelly-like sizing
    
    # Episode settings
    random_start: bool = True


class AdvancedCandleEnv(gym.Env):
    """
    Candle prediction environment with position sizing.
    
    Action space: Box(2)
    - [0] direction: -1.0 (DOWN) to +1.0 (UP), near 0 = WAIT
    - [1] position_size: 0.0 to 1.0 (fraction of max position)
    
    The model learns to:
    - Predict direction with action[0]
    - Size positions based on confidence with action[1]
    - High confidence + correct = big reward
    - Low confidence + wrong = small loss
    - WAIT (action[0] near 0) when uncertain
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        config: AdvancedCandleConfig | None = None,
    ):
        super().__init__()
        
        self.config = config or AdvancedCandleConfig()
        self._setup_data(price_data)
        self._setup_spaces()
        
        # Portfolio state
        self._balance: float = self.config.initial_balance
        self._initial_balance: float = self.config.initial_balance
        
        # Episode state
        self._current_candle_idx: int = 0
        self._current_step: int = 0
        self._position_taken: bool = False
        self._position_direction: float = 0.0
        self._position_size: float = 0.0
        self._candle_open: float = 0.0
        
        logger.info(
            "AdvancedCandleEnv initialized",
            total_candles=len(self._candles),
            initial_balance=self.config.initial_balance,
        )
    
    def _setup_data(self, price_data: pd.DataFrame) -> None:
        """Process price data into candles."""
        df = price_data.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        self._timestamps = df["timestamp"].values
        self._prices = df["price"].values.astype(np.float64)
        self._volumes = df["volume"].values.astype(np.float64) if "volume" in df else np.ones(len(df))
        
        candle_seconds = self.config.candle_minutes * 60
        df["candle_idx"] = (df["timestamp"].astype(np.int64) // 10**9 // candle_seconds).astype(int)
        
        candles = []
        for candle_id, group in df.groupby("candle_idx"):
            if len(group) < 10:
                continue
            
            start_idx = group.index[0]
            end_idx = group.index[-1]
            open_price = group["price"].iloc[0]
            close_price = group["price"].iloc[-1]
            
            candles.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "open": open_price,
                "close": close_price,
                "return": (close_price - open_price) / open_price,  # Actual return
            })
        
        self._candles = candles
        self._compute_atr()
    
    def _compute_atr(self, period: int = 14) -> None:
        """Compute ATR for volatility context."""
        price_changes = np.abs(np.diff(self._prices))
        self._atr = np.zeros(len(self._prices))
        if len(price_changes) > period:
            for i in range(period, len(self._prices)):
                self._atr[i] = np.mean(price_changes[max(0, i-period):i])
        self._atr[:period] = self._atr[period] if period < len(self._atr) else 1.0
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        # Observation: prices + volumes + context (NO balance - model is balance-agnostic)
        price_dim = self.config.price_history_length
        volume_dim = self.config.price_history_length
        context_dim = 5  # current_vs_open, time_remaining, hour_sin, hour_cos, atr
        
        self._obs_dim = price_dim + volume_dim + context_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        
        # Action: [direction (-1 to 1), position_percent (0 to 1)]
        # position_percent is fraction of portfolio to risk (0% to max_position_size%)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset to a new candle."""
        super().reset(seed=seed)
        
        # Select candle
        if self.config.random_start:
            min_candle = max(0, self.config.price_history_length // 100)
            max_candle = len(self._candles) - 1
            self._current_candle_idx = self.np_random.integers(min_candle, max_candle)
        else:
            self._current_candle_idx = 0
        
        candle = self._candles[self._current_candle_idx]
        
        # Reset state
        self._current_step = candle["start_idx"]
        self._position_taken = False
        self._position_direction = 0.0
        self._position_size = 0.0
        self._candle_open = candle["open"]
        self._candle_end_idx = candle["end_idx"]
        self._candle_return = candle["return"]
        
        # Reset balance for each episode (fresh start)
        self._balance = self.config.initial_balance
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.
        
        Action:
        - action[0]: direction, -1=DOWN, +1=UP, ~0=WAIT
        - action[1]: position size, 0-1 fraction of max position
        """
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        
        reward = 0.0
        terminated = False
        
        # Take position if not already taken and direction is significant
        if not self._position_taken and abs(direction) > 0.1:
            self._position_taken = True
            self._position_direction = 1.0 if direction > 0 else -1.0  # Normalize to +1/-1
            self._position_size = size * self.config.max_position_size
            
            # End episode when position is taken (evaluate at candle close)
            terminated = True
            reward = self._compute_reward()
        else:
            # Still waiting
            self._current_step += 1
            
            # Candle ended without taking position
            if self._current_step >= self._candle_end_idx:
                terminated = True
                # Small penalty for not trading (opportunity cost)
                reward = -0.01 * abs(self._candle_return) * 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _compute_reward(self) -> float:
        """
        Compute reward with ASYMMETRIC penalty for wrong bets.
        
        Correct: reward = position_size × price_move (linear)
        Wrong:   reward = -(position_size²) × price_move (SQUARED - much harsher!)
        
        This teaches: "If you're not confident, bet small!"
        - Big bet + Correct = Big reward
        - Big bet + Wrong = HUGE penalty (squared)
        - Small bet + Wrong = Small penalty
        """
        # Check if prediction was correct
        actual_direction = 1.0 if self._candle_return > 0 else -1.0
        correct = (self._position_direction == actual_direction)
        
        # Price move magnitude
        price_move_pct = abs(self._candle_return) * 100
        
        if correct:
            # Linear reward for correct predictions
            reward = self._position_size * price_move_pct
        else:
            # SQUARED penalty for wrong predictions!
            # This heavily punishes overconfident wrong bets
            reward = -(self._position_size ** 2) * price_move_pct * 2  # 2x multiplier for emphasis
        
        # Update balance for tracking
        direction_match = 1.0 if correct else -1.0
        pnl = self._balance * self._position_size * direction_match * abs(self._candle_return)
        self._balance += pnl
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Build observation array."""
        idx = min(self._current_step, len(self._prices) - 1)
        seq_len = self.config.price_history_length
        
        # Price history
        if idx < seq_len:
            prices = np.full(seq_len, self._candle_open)
            prices[seq_len - idx:] = self._prices[:idx] if idx > 0 else []
        else:
            prices = self._prices[idx - seq_len:idx]
        
        normalized_prices = (prices / self._candle_open - 1.0) * 100
        
        # Volume history
        if idx < seq_len:
            volumes = np.zeros(seq_len)
            if idx > 0:
                volumes[seq_len - idx:] = self._volumes[:idx]
        else:
            volumes = self._volumes[idx - seq_len:idx]
        
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        normalized_volumes = volumes / vol_max
        
        # Context features
        current_price = self._prices[idx]
        current_vs_open = (current_price / self._candle_open - 1.0) * 100
        
        candle = self._candles[self._current_candle_idx]
        candle_length = candle["end_idx"] - candle["start_idx"]
        time_remaining = 1.0 - (idx - candle["start_idx"]) / max(candle_length, 1)
        
        timestamp = pd.Timestamp(self._timestamps[idx])
        hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
        
        atr = self._atr[idx] / (self._candle_open + 1e-8) * 100
        
        # NO balance in observation - model outputs percentage, works with any balance
        obs = np.concatenate([
            normalized_prices.astype(np.float32),
            normalized_volumes.astype(np.float32),
            np.array([current_vs_open, time_remaining, hour_sin, hour_cos, atr], dtype=np.float32),
        ])
        
        return np.clip(obs, -100.0, 100.0)
    
    def _get_info(self) -> dict[str, Any]:
        """Build info dictionary."""
        return {
            "candle_idx": self._current_candle_idx,
            "balance": self._balance,
            "balance_ratio": self._balance / self._initial_balance,
            "position_taken": self._position_taken,
            "position_direction": "UP" if self._position_direction > 0 else "DOWN" if self._position_direction < 0 else None,
            "position_size": self._position_size,
            "candle_return": self._candle_return,
        }


def make_advanced_vec_env(
    price_data: pd.DataFrame,
    num_envs: int = 4,
    config: AdvancedCandleConfig | None = None,
) -> Any:
    """Create vectorized environments."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    def make_env_fn(seed: int):
        def _init():
            cfg = config or AdvancedCandleConfig()
            env = AdvancedCandleEnv(price_data, config=cfg)
            env.reset(seed=seed)
            return env
        return _init
    
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)
