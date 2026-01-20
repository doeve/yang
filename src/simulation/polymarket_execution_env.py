"""
Polymarket Execution Environment for SAC training.

Layer 2 of the two-layer trading system. The SAC agent learns:
- When to trade (selectivity)
- How much to trade (position sizing)
- When to hold/wait

Uses probability output from Layer 1 LSTM as input.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

logger = structlog.get_logger(__name__)


@dataclass
class PolymarketExecutionConfig:
    """Configuration for Polymarket execution environment."""
    
    # Market parameters
    candle_minutes: int = 15
    fee_percent: float = 0.1        # Polymarket trading fee (~0.1%)
    slippage_percent: float = 0.05  # Estimated slippage
    
    # Position limits
    max_position_size: float = 0.25  # Max 25% of balance per trade
    min_edge_to_trade: float = 0.02  # Only trade if edge > 2%
    
    # Reward shaping
    pnl_scale: float = 10.0          # Scale PnL for better gradient signal
    inventory_penalty: float = 0.01  # Penalty for holding near expiry
    late_entry_penalty_start: float = 0.7  # Penalty after 70% of candle
    late_entry_penalty_scale: float = 0.5
    overtrading_penalty: float = 0.05
    correct_abstention_bonus: float = 0.02  # Bonus for not trading low-edge
    
    # Episode parameters
    initial_balance: float = 1000.0
    max_trades_per_candle: int = 1   # Only allow one trade per candle
    
    # Simulation
    simulate_market_movement: bool = True  # Simulate price changes within candle


class PolymarketExecutionEnv(gym.Env):
    """
    SAC decision environment for Polymarket execution.
    
    This environment receives:
    - Probability output from Layer 1 LSTM
    - Current Polymarket market prices (YES/NO)
    - Market conditions (spread, time, etc.)
    
    And learns to decide:
    - Whether to trade or hold
    - Direction (buy YES or buy NO)
    - Position size
    
    The key insight is that the SAC agent should learn:
    1. To only trade when model_prob differs significantly from market_price (edge)
    2. To size positions proportionally to edge confidence
    3. To NOT trade when uncertain (most value comes from selectivity)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[PolymarketExecutionConfig] = None,
        historical_data: Optional[dict] = None,
    ):
        super().__init__()
        
        self.config = config or PolymarketExecutionConfig()
        self.historical_data = historical_data  # For backtesting
        
        # State space: 14 dimensions
        # [model_prob, market_yes, market_no, spread, time_remaining, time_in_candle,
        #  position, unrealized_pnl, model_confidence, dxy_momentum, 
        #  trades_this_candle, recent_win_rate, balance_normalized, edge]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )
        
        # Action space: 3 dimensions
        # [direction (-1 to 1), size (0 to 1), hold_prob (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        
        # Episode state
        self._reset_episode_state()
        
        # Historical tracking for win rate calculation
        self._trade_outcomes: list[bool] = []
        
        logger.info(
            "PolymarketExecutionEnv initialized",
            config=self.config,
        )
    
    def _reset_episode_state(self) -> None:
        """Reset all episode-specific state."""
        self._balance = self.config.initial_balance
        self._position = 0.0  # -1 to 1 (short to long)
        self._position_entry_price = 0.0
        self._trades_this_candle = 0
        self._candle_step = 0
        self._unrealized_pnl = 0.0
        
        # Current market state
        self._model_prob = 0.5
        self._market_yes_price = 0.5
        self._market_no_price = 0.5
        self._spread = 0.02
        self._dxy_momentum = 0.0
        
        # Outcome (set at candle end)
        self._outcome: Optional[bool] = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self._reset_episode_state()
        
        # Generate random initial state (or use historical data)
        if self.historical_data is not None and options and "candle_idx" in options:
            self._load_historical_candle(options["candle_idx"])
        else:
            self._generate_random_state()
        
        return self._get_observation(), self._get_info()
    
    def _generate_random_state(self) -> None:
        """Generate random market state for training."""
        # Random model probability (Layer 1 output)
        self._model_prob = self.np_random.uniform(0.3, 0.7)
        
        # Market prices (should differ from model by some edge)
        # Sometimes aligned, sometimes mispriced
        edge = self.np_random.uniform(-0.15, 0.15)
        self._market_yes_price = np.clip(self._model_prob + edge, 0.05, 0.95)
        self._market_no_price = 1.0 - self._market_yes_price
        
        # Spread
        self._spread = self.np_random.uniform(0.01, 0.05)
        
        # DXY momentum
        self._dxy_momentum = self.np_random.normal(0, 0.3)
        
        # Generate outcome (for reward calculation)
        # Outcome follows model probability + noise
        self._outcome = self.np_random.random() < self._model_prob
    
    def _load_historical_candle(self, candle_idx: int) -> None:
        """Load state from historical data."""
        candle = self.historical_data[candle_idx]
        self._model_prob = candle["model_prob"]
        self._market_yes_price = candle["market_yes"]
        self._market_no_price = candle["market_no"]
        self._spread = candle.get("spread", 0.02)
        self._dxy_momentum = candle.get("dxy_momentum", 0.0)
        self._outcome = candle["outcome"]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [direction, size, hold_prob]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        hold_prob = float(action[2])
        
        reward = 0.0
        terminated = False
        
        # Calculate timing
        candle_total_steps = 60  # Simulated steps per candle
        time_in_candle = self._candle_step / candle_total_steps
        time_remaining = 1.0 - time_in_candle
        
        # Calculate edge
        edge = self._model_prob - self._market_yes_price
        
        # Check if agent chooses to hold
        if hold_prob > 0.5:
            # Agent chose to hold/skip
            if abs(edge) < self.config.min_edge_to_trade:
                # Correct decision to abstain on low edge
                reward += self.config.correct_abstention_bonus * self.config.pnl_scale
            
            # Advance time
            self._candle_step += 1
            
            if self._candle_step >= candle_total_steps:
                terminated = True
                reward += self._compute_settlement_reward()
        
        elif self._trades_this_candle < self.config.max_trades_per_candle:
            # Agent wants to trade
            if abs(direction) > 0.1 and size > 0.05:
                # Execute trade
                trade_direction = 1.0 if direction > 0 else -1.0
                trade_size = size * self.config.max_position_size
                
                # Entry price with slippage
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
                
                # Late entry penalty
                if time_in_candle > self.config.late_entry_penalty_start:
                    late_penalty = (time_in_candle - self.config.late_entry_penalty_start) * self.config.late_entry_penalty_scale
                    reward -= late_penalty * self.config.pnl_scale
                
                # End episode after trade (settle at candle end)
                terminated = True
                reward += self._compute_trade_reward()
        else:
            # Max trades reached, just advance time
            self._candle_step += 1
            
            if self._candle_step >= candle_total_steps:
                terminated = True
                reward += self._compute_settlement_reward()
        
        # Update unrealized PnL if in position
        if abs(self._position) > 0:
            self._update_unrealized_pnl()
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _compute_trade_reward(self) -> float:
        """Compute reward for a trade based on outcome."""
        if self._outcome is None:
            return 0.0
        
        # Determine if position was correct
        position_direction = 1.0 if self._position > 0 else -1.0
        position_size = abs(self._position)
        
        if self._outcome:  # YES outcome
            target_price = 1.0
            if position_direction > 0:  # Bought YES, correct
                pnl = (target_price - self._position_entry_price) * position_size
            else:  # Bought NO (short YES), wrong
                pnl = -(target_price - (1.0 - self._position_entry_price)) * position_size
        else:  # NO outcome
            target_price = 0.0
            if position_direction < 0:  # Bought NO, correct
                pnl = (1.0 - self._position_entry_price - target_price) * position_size
            else:  # Bought YES, wrong
                pnl = (target_price - self._position_entry_price) * position_size
        
        # Scale PnL
        reward = pnl * self.config.pnl_scale
        
        # Track outcome for win rate
        self._trade_outcomes.append(pnl > 0)
        if len(self._trade_outcomes) > 100:
            self._trade_outcomes.pop(0)
        
        return reward
    
    def _compute_settlement_reward(self) -> float:
        """Compute reward at candle end without trading."""
        if abs(self._position) > 0:
            # Position exists, settle it
            return self._compute_trade_reward()
        else:
            # No position - check if we should have traded
            edge = abs(self._model_prob - self._market_yes_price)
            
            if edge > 0.15:  # Large edge missed
                return -0.05 * self.config.pnl_scale  # Small penalty for missed opportunity
            else:
                return 0.01 * self.config.pnl_scale  # Small reward for correct skip
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL based on current market prices."""
        if abs(self._position) == 0:
            self._unrealized_pnl = 0.0
            return
        
        position_direction = 1.0 if self._position > 0 else -1.0
        position_size = abs(self._position)
        
        if position_direction > 0:
            current_price = self._market_yes_price
        else:
            current_price = self._market_no_price
        
        self._unrealized_pnl = (current_price - self._position_entry_price) * position_size
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        candle_total_steps = 60
        time_in_candle = self._candle_step / candle_total_steps
        time_remaining = 1.0 - time_in_candle
        
        # Model confidence: distance from 0.5
        model_confidence = abs(self._model_prob - 0.5) * 2
        
        # Edge
        edge = self._model_prob - self._market_yes_price
        
        # Balance normalized
        balance_norm = self._balance / self.config.initial_balance
        
        # Recent win rate
        recent_win_rate = np.mean(self._trade_outcomes[-10:]) if self._trade_outcomes else 0.5
        
        obs = np.array([
            self._model_prob,
            self._market_yes_price,
            self._market_no_price,
            self._spread,
            time_remaining,
            time_in_candle,
            self._position,
            self._unrealized_pnl,
            model_confidence,
            self._dxy_momentum,
            self._trades_this_candle,
            recent_win_rate,
            balance_norm,
            edge,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> dict[str, Any]:
        """Build info dictionary."""
        return {
            "balance": self._balance,
            "position": self._position,
            "model_prob": self._model_prob,
            "market_yes": self._market_yes_price,
            "edge": self._model_prob - self._market_yes_price,
            "outcome": self._outcome,
            "trades_this_candle": self._trades_this_candle,
        }


class PolymarketBacktestEnv(PolymarketExecutionEnv):
    """
    Backtesting environment that uses historical data.
    
    Iterates through historical candles in order for realistic backtesting.
    """
    
    def __init__(
        self,
        historical_candles: list[dict],
        config: Optional[PolymarketExecutionConfig] = None,
    ):
        super().__init__(config=config)
        
        self.historical_candles = historical_candles
        self._candle_idx = 0
        self._max_candles = len(historical_candles)
        
        logger.info(
            "PolymarketBacktestEnv initialized",
            num_candles=self._max_candles,
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Load next candle
        if self._candle_idx >= self._max_candles:
            self._candle_idx = 0
        
        candle = self.historical_candles[self._candle_idx]
        self._model_prob = candle["model_prob"]
        self._market_yes_price = candle["market_yes"]
        self._market_no_price = 1.0 - candle["market_yes"]
        self._spread = candle.get("spread", 0.02)
        self._dxy_momentum = candle.get("dxy_momentum", 0.0)
        self._outcome = candle["outcome"]
        
        self._candle_idx += 1
        
        return self._get_observation(), self._get_info()


def make_polymarket_vec_env(
    num_envs: int = 4,
    config: Optional[PolymarketExecutionConfig] = None,
) -> Any:
    """Create vectorized Polymarket execution environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        return PolymarketExecutionEnv(config=config)
    
    return DummyVecEnv([make_env for _ in range(num_envs)])
