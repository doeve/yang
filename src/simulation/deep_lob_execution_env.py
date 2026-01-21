"""
DeepLOB Execution Environment for SAC training.

Layer 2 using 3-class predictions from DeepLOB (Down/Hold/Up).
The SAC agent learns optimal trade execution based on class probabilities.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

logger = structlog.get_logger(__name__)


@dataclass
class DeepLOBExecutionConfig:
    """Configuration for DeepLOB execution environment."""
    
    # Market parameters
    candle_minutes: int = 15
    fee_percent: float = 0.1
    slippage_percent: float = 0.05
    
    # Position limits
    max_position_size: float = 0.25
    
    # Confidence thresholds (when to act on predictions)
    min_confidence_to_trade: float = 0.35  # Lower threshold to encourage trading
    high_confidence_threshold: float = 0.55  # Lower for more trading
    
    # Reward shaping (adjusted to encourage trading on clear signals)
    pnl_scale: float = 10.0
    correct_abstention_bonus: float = 0.02  # Lower bonus for holding
    overtrading_penalty: float = 0.01  # Lower penalty
    missed_signal_penalty: float = 0.1  # Higher penalty for missing good signals
    trade_alignment_bonus: float = 0.05  # Higher bonus for aligned trades
    
    # Episode parameters
    initial_balance: float = 1000.0
    max_trades_per_candle: int = 1


class DeepLOBExecutionEnv(gym.Env):
    """
    SAC execution environment using 3-class DeepLOB predictions.
    
    Observation includes:
    - DeepLOB class probabilities (Down, Hold, Up)
    - Predicted class
    - Prediction confidence
    - Market prices and spread
    - Position and PnL
    - Time features
    
    Action space:
    - Direction: -1 (sell/short) to 1 (buy/long)
    - Size: 0 to 1 (fraction of max_position_size)
    - Hold probability: 0 to 1 (agent's decision to wait)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[DeepLOBExecutionConfig] = None,
    ):
        super().__init__()
        
        self.config = config or DeepLOBExecutionConfig()
        
        # State space: 15 dimensions
        # [prob_down, prob_hold, prob_up, predicted_class_normalized,
        #  confidence, market_yes, market_no, spread, time_remaining,
        #  position, unrealized_pnl, trades_this_candle, win_rate,
        #  balance_norm, edge_up]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32,
        )
        
        # Action space: [direction, size, hold_prob]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        
        self._reset_state()
        self._trade_outcomes: list[bool] = []
        
        logger.info("DeepLOBExecutionEnv initialized")
    
    def _reset_state(self) -> None:
        """Reset episode state."""
        self._balance = self.config.initial_balance
        self._position = 0.0
        self._position_entry_price = 0.0
        self._trades_this_candle = 0
        self._candle_step = 0
        self._unrealized_pnl = 0.0
        
        # DeepLOB predictions (3 classes)
        self._prob_down = 0.33
        self._prob_hold = 0.34
        self._prob_up = 0.33
        
        # Market prices
        self._market_yes_price = 0.5
        self._market_no_price = 0.5
        self._spread = 0.02
        
        # Outcome (set at candle end)
        self._outcome: Optional[int] = None  # 0=Down, 1=Hold, 2=Up
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        self._generate_random_state()
        return self._get_observation(), self._get_info()
    
    def _generate_random_state(self) -> None:
        """Generate random state for training."""
        # Generate 3-class probabilities (from Dirichlet for realistic spread)
        alpha = self.np_random.uniform(0.5, 2.0, size=3)
        probs = self.np_random.dirichlet(alpha)
        self._prob_down, self._prob_hold, self._prob_up = probs
        
        # Predicted class
        self._predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        
        # Market prices (can be misaligned with predictions)
        if self.np_random.random() < 0.7:
            # Efficient market: aligned with predictions
            self._market_yes_price = 0.5 + (self._prob_up - self._prob_down) * 0.3
        else:
            # Mispriced market: edge opportunity
            self._market_yes_price = self.np_random.uniform(0.3, 0.7)
        
        self._market_yes_price = np.clip(self._market_yes_price, 0.1, 0.9)
        self._market_no_price = 1.0 - self._market_yes_price
        self._spread = self.np_random.uniform(0.01, 0.04)
        
        # Generate outcome based on true probabilities (with noise)
        outcome_rand = self.np_random.random()
        if outcome_rand < self._prob_down * 0.8:
            self._outcome = 0  # Down
        elif outcome_rand < (self._prob_down + self._prob_hold) * 0.8:
            self._outcome = 1  # Hold (no significant move)
        else:
            self._outcome = 2  # Up
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step."""
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        hold_prob = float(action[2])
        
        reward = 0.0
        terminated = False
        
        candle_total_steps = 60
        time_in_candle = self._candle_step / candle_total_steps
        
        # Confidence of prediction
        confidence = max(self._prob_down, self._prob_hold, self._prob_up)
        predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        
        # Agent chooses to hold
        if hold_prob > 0.5:
            # Reward for holding when Hold class is predicted
            if predicted_class == 1:  # Model predicts Hold
                reward += self.config.correct_abstention_bonus * self.config.pnl_scale
            elif confidence < self.config.min_confidence_to_trade:
                # Correct to hold when uncertain
                reward += self.config.correct_abstention_bonus * 0.5 * self.config.pnl_scale
            
            self._candle_step += 1
            if self._candle_step >= candle_total_steps:
                terminated = True
                reward += self._compute_settlement_reward()
        
        elif self._trades_this_candle < self.config.max_trades_per_candle:
            # Agent wants to trade
            if abs(direction) > 0.1 and size > 0.05:
                # Check if trade aligns with prediction
                trade_direction = 1.0 if direction > 0 else -1.0
                
                # Trade alignment bonus/penalty
                if trade_direction > 0 and predicted_class == 2:  # Long when Up predicted
                    alignment_bonus = confidence * 0.02
                elif trade_direction < 0 and predicted_class == 0:  # Short when Down predicted
                    alignment_bonus = confidence * 0.02
                elif predicted_class == 1:  # Trading when Hold predicted
                    alignment_bonus = -0.03  # Penalty for trading against Hold signal
                else:
                    alignment_bonus = -0.02  # Trading against direction
                
                reward += alignment_bonus * self.config.pnl_scale
                
                # Execute trade
                trade_size = size * self.config.max_position_size
                
                if trade_direction > 0:
                    entry_price = self._market_yes_price + self._spread * 0.5
                else:
                    entry_price = self._market_no_price + self._spread * 0.5
                
                entry_price += self.config.slippage_percent * 0.01
                
                self._position = trade_direction * trade_size
                self._position_entry_price = entry_price
                self._trades_this_candle += 1
                
                # Trading costs
                fee = trade_size * self.config.fee_percent * 0.01
                reward -= fee * self.config.pnl_scale
                
                terminated = True
                reward += self._compute_trade_reward()
        else:
            self._candle_step += 1
            if self._candle_step >= candle_total_steps:
                terminated = True
                reward += self._compute_settlement_reward()
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _compute_trade_reward(self) -> float:
        """Compute reward based on trade outcome."""
        if self._outcome is None:
            return 0.0
        
        position_direction = 1.0 if self._position > 0 else -1.0
        position_size = abs(self._position)
        
        # Map outcome to price movement
        if self._outcome == 2:  # Up
            target_price = 1.0
            pnl_yes = target_price - self._position_entry_price
        elif self._outcome == 0:  # Down
            target_price = 0.0
            pnl_yes = target_price - self._position_entry_price
        else:  # Hold - small random movement
            target_price = 0.5 + np.random.uniform(-0.05, 0.05)
            pnl_yes = target_price - self._position_entry_price
        
        if position_direction > 0:  # Long YES
            pnl = pnl_yes * position_size
        else:  # Short YES (Long NO)
            pnl = -pnl_yes * position_size
        
        self._trade_outcomes.append(pnl > 0)
        if len(self._trade_outcomes) > 100:
            self._trade_outcomes.pop(0)
        
        return pnl * self.config.pnl_scale
    
    def _compute_settlement_reward(self) -> float:
        """Reward at candle end without trading."""
        if abs(self._position) > 0:
            return self._compute_trade_reward()
        
        # Check if we missed a clear signal
        predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        confidence = max(self._prob_down, self._prob_hold, self._prob_up)
        
        if predicted_class != 1 and confidence > self.config.high_confidence_threshold:
            # Missed high confidence Up/Down signal
            if (predicted_class == 2 and self._outcome == 2) or \
               (predicted_class == 0 and self._outcome == 0):
                return -self.config.missed_signal_penalty * self.config.pnl_scale
        
        return 0.01 * self.config.pnl_scale  # Small reward for not trading
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        candle_total_steps = 60
        time_remaining = 1.0 - self._candle_step / candle_total_steps
        
        # Normalized predicted class: 0=Down, 0.5=Hold, 1=Up
        predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        predicted_class_norm = predicted_class / 2.0
        
        # Confidence
        confidence = max(self._prob_down, self._prob_hold, self._prob_up)
        
        # Edge: difference between model's implied price and market
        model_implied_price = 0.5 + (self._prob_up - self._prob_down) * 0.5
        edge_up = model_implied_price - self._market_yes_price
        
        balance_norm = self._balance / self.config.initial_balance
        win_rate = np.mean(self._trade_outcomes[-10:]) if self._trade_outcomes else 0.5
        
        obs = np.array([
            self._prob_down,
            self._prob_hold,
            self._prob_up,
            predicted_class_norm,
            confidence,
            self._market_yes_price,
            self._market_no_price,
            self._spread,
            time_remaining,
            self._position,
            self._unrealized_pnl,
            self._trades_this_candle,
            win_rate,
            balance_norm,
            edge_up,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> dict[str, Any]:
        return {
            "balance": self._balance,
            "position": self._position,
            "prob_down": self._prob_down,
            "prob_hold": self._prob_hold,
            "prob_up": self._prob_up,
            "predicted_class": np.argmax([self._prob_down, self._prob_hold, self._prob_up]),
            "outcome": self._outcome,
        }


def make_deep_lob_vec_env(
    num_envs: int = 4,
    config: Optional[DeepLOBExecutionConfig] = None,
) -> Any:
    """Create vectorized DeepLOB execution environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        return DeepLOBExecutionEnv(config=config)
    
    return DummyVecEnv([make_env for _ in range(num_envs)])
