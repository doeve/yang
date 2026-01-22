"""
Historical Trading Environment for SAC Training.

Uses real Polymarket BTC 15min candle data with:
- Normalized rewards [-1, 1]
- Real market outcomes
- Proper risk-adjusted rewards
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any, List
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HistoricalTradingConfig:
    """Configuration for historical trading environment."""
    
    # === Data ===
    data_path: str = "./data/polymarket/btc_15min_candles.parquet"
    predictions_path: str = ""  # Optional: pre-generated DeepLOB predictions
    
    # === Position Sizing ===
    max_position_size: float = 0.25
    
    # === Trading Costs ===
    spread_cost: float = 0.02  # 2% spread
    slippage: float = 0.01     # 1% slippage
    
    # === Reward Normalization ===
    win_reward: float = 1.0    # Normalized win
    loss_penalty: float = -1.0  # Normalized loss
    hold_reward: float = 0.0   # No reward for holding
    
    # === Risk Penalties ===
    wrong_direction_penalty: float = -0.5  # Extra penalty for wrong prediction
    confidence_bonus_scale: float = 0.2    # Bonus for high confidence correct trades


class HistoricalTradingEnv(gym.Env):
    """
    Trading environment using real Polymarket historical data.
    
    Single-step episodes: one decision per candle.
    Normalized rewards in [-1, 1] range.
    
    Observation Space (15D):
    - DeepLOB probabilities (3): down, hold, up
    - Predicted class & confidence (2)
    - Market price features (3): yes_price, no_price, volume_norm
    - Time features (2): hour_of_day, day_of_week
    - Position state (3): position, consecutive_wins, consecutive_losses
    - Edge (2): edge_up, edge_down
    
    Action Space (3D):
    - direction: [-1, 1] - Short to Long
    - size: [0, 1] - Position size fraction
    - hold_prob: [0, 1] - Hold/wait probability
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[HistoricalTradingConfig] = None,
        candle_indices: Optional[np.ndarray] = None,  # Filter for specific fold
    ):
        super().__init__()
        
        self.config = config or HistoricalTradingConfig()
        self._candle_indices = candle_indices
        
        # Load historical data (with optional fold filtering)
        self._load_data()
        
        # Observation space: 15 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32,
        )
        
        # Action space: 3D [direction, size, hold_prob]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        
        # Episode state
        self._current_idx = 0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._total_trades = 0
        self._wins = 0
        
        logger.info(
            "HistoricalTradingEnv initialized",
            num_candles=len(self.candles),
            uses_real_predictions=self._use_real_predictions,
            obs_dim=15,
            action_dim=3,
        )
    
    def _load_data(self) -> None:
        """Load historical candle data with optional pre-generated predictions."""
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        # Check for pre-generated predictions
        if self.config.predictions_path and Path(self.config.predictions_path).exists():
            # Load predictions dataset (has real DeepLOB probabilities)
            df = pd.read_parquet(self.config.predictions_path)
            self._use_real_predictions = True
            logger.info(f"Loaded {len(df)} candles with real predictions")
        else:
            # Load raw candles (will generate synthetic predictions)
            df = pd.read_parquet(data_path)
            self._use_real_predictions = False
            # Filter for closed candles with outcomes
            df = df[df["closed"] == True].reset_index(drop=True)
            logger.info(f"Loaded {len(df)} historical candles (synthetic predictions)")
        
        # Apply fold filtering if provided
        if self._candle_indices is not None:
            df = df.iloc[self._candle_indices].reset_index(drop=True)
            logger.info(f"Filtered to {len(df)} candles for fold")
        
        self.candles = df
        
        if len(self.candles) == 0:
            raise ValueError("No candles found in data")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Random starting point
        self._current_idx = self.np_random.integers(0, len(self.candles))
        
        # Generate synthetic DeepLOB predictions for training
        self._generate_predictions()
        
        return self._get_observation(), self._get_info()
    
    def _generate_predictions(self) -> None:
        """Load or generate DeepLOB predictions for current candle."""
        candle = self.candles.iloc[self._current_idx]
        up_won = candle["up_won"]
        
        # Use real predictions if available
        if self._use_real_predictions:
            self._prob_down = candle["prob_down"]
            self._prob_hold = candle["prob_hold"]
            self._prob_up = candle["prob_up"]
        else:
            # Simulate DeepLOB with varying accuracy
            accuracy = self.np_random.uniform(0.4, 0.7)  # 40-70% accuracy
            
            if self.np_random.random() < accuracy:
                # Correct prediction
                if up_won:
                    base = np.array([0.15, 0.25, 0.60])  # Predict Up
                else:
                    base = np.array([0.60, 0.25, 0.15])  # Predict Down
            else:
                # Wrong prediction (add noise)
                if up_won:
                    base = np.array([0.50, 0.30, 0.20])  # Wrong - predict Down
                else:
                    base = np.array([0.20, 0.30, 0.50])  # Wrong - predict Up
            
            # Add noise
            noise = self.np_random.normal(0, 0.05, size=3)
            probs = np.clip(base + noise, 0.01, 0.99)
            probs = probs / probs.sum()  # Normalize
            
            self._prob_down, self._prob_hold, self._prob_up = probs
        
        # Simulate market prices (with some mispricing)
        fair_price = 0.55 if up_won else 0.45
        noise = self.np_random.normal(0, 0.08)
        self._yes_price = np.clip(fair_price + noise, 0.20, 0.80)
        
        # Volume normalized (0-1)
        self._volume_norm = min(candle["volume"] / 300000, 1.0)
        
        # Time features
        dt = candle["datetime"]
        self._hour = dt.hour / 24.0
        self._day = dt.weekday() / 7.0
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading decision."""
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        hold_prob = float(action[2])
        
        candle = self.candles.iloc[self._current_idx]
        up_won = candle["up_won"]
        
        # === Compute Reward ===
        if hold_prob > 0.5 or size < 0.1:
            # Agent chose to hold
            reward = self.config.hold_reward
            action_taken = "hold"
        else:
            # Agent took position
            trade_size = size * self.config.max_position_size
            went_long = direction > 0
            
            # Did we win?
            if (went_long and up_won) or (not went_long and not up_won):
                # Won the trade
                base_reward = self.config.win_reward
                
                # Bonus for confidence alignment
                predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
                confidence = max(self._prob_down, self._prob_hold, self._prob_up)
                
                correct_prediction = (went_long and predicted_class == 2) or \
                                   (not went_long and predicted_class == 0)
                
                if correct_prediction:
                    confidence_bonus = confidence * self.config.confidence_bonus_scale
                else:
                    confidence_bonus = 0.0
                
                # Scale by size (encourage proper sizing)
                reward = (base_reward + confidence_bonus) * (0.5 + trade_size)
                
                self._wins += 1
                self._consecutive_wins += 1
                self._consecutive_losses = 0
                action_taken = "win"
            else:
                # Lost the trade
                base_penalty = self.config.loss_penalty
                
                # Extra penalty if prediction was wrong
                predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
                
                wrong_prediction = (went_long and predicted_class == 0) or \
                                  (not went_long and predicted_class == 2)
                
                if wrong_prediction:
                    wrong_penalty = self.config.wrong_direction_penalty
                else:
                    wrong_penalty = 0.0
                
                # Trading costs
                cost_penalty = (self.config.spread_cost + self.config.slippage) * trade_size
                
                reward = (base_penalty + wrong_penalty - cost_penalty) * (0.5 + trade_size)
                
                self._consecutive_losses += 1
                self._consecutive_wins = 0
                action_taken = "loss"
            
            self._total_trades += 1
        
        # Clip reward to [-2, 2] for stability (allows some bonus range)
        reward = np.clip(reward, -2.0, 2.0)
        
        # Episode ends after one candle (single-step)
        terminated = True
        
        info = self._get_info()
        info["action_taken"] = action_taken
        info["up_won"] = up_won
        
        return self._get_observation(), reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Build 15-dimensional observation."""
        predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        confidence = max(self._prob_down, self._prob_hold, self._prob_up)
        
        # Edge calculations
        model_implied = 0.5 + (self._prob_up - self._prob_down) * 0.5
        edge_up = model_implied - self._yes_price
        edge_down = (1 - model_implied) - (1 - self._yes_price)
        
        # Win/loss streaks (normalized)
        win_streak = min(self._consecutive_wins / 5.0, 1.0)
        loss_streak = min(self._consecutive_losses / 5.0, 1.0)
        
        # Win rate
        if self._total_trades > 0:
            win_rate = self._wins / self._total_trades
        else:
            win_rate = 0.5
        
        obs = np.array([
            # DeepLOB predictions (3)
            self._prob_down,
            self._prob_hold,
            self._prob_up,
            # Prediction features (2)
            predicted_class / 2.0,
            confidence,
            # Market features (3)
            self._yes_price,
            1 - self._yes_price,
            self._volume_norm,
            # Time features (2)
            self._hour,
            self._day,
            # Position/history state (3)
            win_rate,
            win_streak,
            loss_streak,
            # Edge (2)
            edge_up,
            edge_down,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> dict:
        candle = self.candles.iloc[self._current_idx]
        return {
            "candle_idx": self._current_idx,
            "timestamp": candle["timestamp"],
            "total_trades": self._total_trades,
            "wins": self._wins,
            "win_rate": self._wins / max(1, self._total_trades),
        }


def make_historical_vec_env(
    num_envs: int = 4,
    config: Optional[HistoricalTradingConfig] = None,
    candle_indices: Optional[np.ndarray] = None,
):
    """Create vectorized historical trading environments with optional fold filtering."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        return HistoricalTradingEnv(config=config, candle_indices=candle_indices)
    
    return DummyVecEnv([make_env for _ in range(num_envs)])

