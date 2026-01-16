"""
Reward calculation for trading environment.

Provides sophisticated reward shaping for RL training:
- Risk-adjusted returns
- Transaction cost penalties
- Position holding incentives
- Resolution bonus for prediction markets
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    
    # Scaling
    reward_scale: float = 1.0
    
    # PnL component
    pnl_weight: float = 1.0
    risk_penalty: float = 0.1  # Penalty for volatility exposure
    
    # Transaction costs
    transaction_penalty: float = 0.5
    
    # Position penalties/incentives
    holding_penalty: float = 0.001  # Small penalty for holding
    wrong_side_penalty: float = 0.1  # Penalty for being on wrong side
    
    # Resolution bonus
    resolution_bonus_weight: float = 2.0
    resolution_threshold: float = 0.1  # Time threshold for bonus
    
    # Drawdown penalty
    drawdown_penalty: float = 0.5
    drawdown_threshold: float = 0.1  # 10% drawdown threshold
    
    # Consistency bonus
    consistency_bonus: float = 0.01
    
    # Clipping
    clip_min: float = -10.0
    clip_max: float = 10.0


class RewardCalculator:
    """
    Calculates rewards for trading actions.
    
    The reward function is designed to encourage:
    1. Profitable trading (PnL maximization)
    2. Risk management (volatility-adjusted returns)
    3. Capital efficiency (avoid excessive trading)
    4. Correct positioning near resolution
    
    Example:
        calculator = RewardCalculator()
        reward = calculator.compute_reward(
            pnl_delta=100,
            position=0.5,
            volatility=0.02,
            transaction_cost=5,
            time_to_resolution=0.05,
            outcome=True,
        )
    """
    
    def __init__(self, config: RewardConfig | None = None):
        """
        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        
        # Tracking for running statistics
        self._pnl_history: list[float] = []
        self._reward_history: list[float] = []
        self._peak_pnl: float = 0.0
    
    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._pnl_history = []
        self._reward_history = []
        self._peak_pnl = 0.0
    
    def compute_reward(
        self,
        pnl_delta: float,
        position: float,
        volatility: float,
        transaction_cost: float,
        time_to_resolution: float,
        current_pnl: float,
        outcome: bool | None = None,
        price: float = 0.5,
        **kwargs: Any,
    ) -> float:
        """
        Compute the reward for a trading step.
        
        Args:
            pnl_delta: Change in PnL this step
            position: Current position (positive = long, negative = short)
            volatility: Current price volatility
            transaction_cost: Transaction cost incurred this step
            time_to_resolution: Normalized time to resolution (0-1)
            current_pnl: Current total PnL
            outcome: Market outcome if known (True = Yes, False = No)
            price: Current market price
            **kwargs: Additional context
            
        Returns:
            Computed reward value
        """
        reward_components = {}
        
        # 1. PnL Component (risk-adjusted)
        # ================================
        # Reward profitable moves, penalize for volatility exposure
        
        pnl_reward = self.config.pnl_weight * pnl_delta
        
        # Risk adjustment: penalize for being exposed to volatility
        risk_penalty = self.config.risk_penalty * abs(position) * volatility * 100
        
        reward_components["pnl"] = pnl_reward - risk_penalty
        
        # 2. Transaction Cost Penalty
        # ============================
        # Discourage excessive trading
        
        reward_components["transaction"] = -self.config.transaction_penalty * transaction_cost
        
        # 3. Position Penalties
        # =====================
        # Small penalty for holding (encourage decisive action)
        
        holding_penalty = self.config.holding_penalty * abs(position) * 10
        reward_components["holding"] = -holding_penalty
        
        # 4. Resolution Bonus
        # ===================
        # Reward correct positioning near resolution
        
        resolution_bonus = 0.0
        if time_to_resolution < self.config.resolution_threshold and outcome is not None:
            target_price = 1.0 if outcome else 0.0
            
            # Is position aligned with outcome?
            if (position > 0 and outcome) or (position < 0 and not outcome):
                # Correct side - reward
                confidence = 1 - abs(price - target_price)
                resolution_bonus = (
                    self.config.resolution_bonus_weight 
                    * abs(position) 
                    * confidence 
                    * (1 - time_to_resolution)
                )
            elif (position > 0 and not outcome) or (position < 0 and outcome):
                # Wrong side - penalize
                resolution_bonus = (
                    -self.config.wrong_side_penalty 
                    * abs(position) 
                    * (1 - time_to_resolution)
                )
        
        reward_components["resolution"] = resolution_bonus
        
        # 5. Drawdown Penalty
        # ===================
        # Penalize for significant drawdowns
        
        self._pnl_history.append(current_pnl)
        if current_pnl > self._peak_pnl:
            self._peak_pnl = current_pnl
        
        drawdown = (self._peak_pnl - current_pnl) / (abs(self._peak_pnl) + 1) if self._peak_pnl != 0 else 0
        
        drawdown_penalty = 0.0
        if drawdown > self.config.drawdown_threshold:
            drawdown_penalty = self.config.drawdown_penalty * (drawdown - self.config.drawdown_threshold)
        
        reward_components["drawdown"] = -drawdown_penalty
        
        # 6. Consistency Bonus
        # ====================
        # Small bonus for consistent positive returns
        
        consistency_bonus = 0.0
        if len(self._pnl_history) > 10:
            recent_pnl = self._pnl_history[-10:]
            if all(p >= r for p, r in zip(recent_pnl[1:], recent_pnl[:-1])):
                # Monotonically increasing
                consistency_bonus = self.config.consistency_bonus
        
        reward_components["consistency"] = consistency_bonus
        
        # Total Reward
        # ============
        total_reward = sum(reward_components.values())
        total_reward *= self.config.reward_scale
        
        # Clip to prevent extreme values
        total_reward = np.clip(total_reward, self.config.clip_min, self.config.clip_max)
        
        self._reward_history.append(total_reward)
        
        return float(total_reward)
    
    def get_episode_stats(self) -> dict[str, float]:
        """
        Get statistics for the current episode.
        
        Returns:
            Dict with episode statistics
        """
        if not self._reward_history:
            return {
                "total_reward": 0.0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
                "final_pnl": 0.0,
                "max_drawdown": 0.0,
            }
        
        rewards = np.array(self._reward_history)
        pnls = np.array(self._pnl_history) if self._pnl_history else np.array([0.0])
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(pnls)
        drawdown = (peak - pnls) / (np.abs(peak) + 1)
        
        return {
            "total_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "max_reward": float(rewards.max()),
            "min_reward": float(rewards.min()),
            "final_pnl": float(pnls[-1]),
            "max_drawdown": float(drawdown.max()),
        }


class SharpeRewardCalculator(RewardCalculator):
    """
    Reward calculator based on Sharpe ratio.
    
    Uses rolling returns to compute a Sharpe-like reward,
    encouraging consistent risk-adjusted returns.
    """
    
    def __init__(
        self,
        config: RewardConfig | None = None,
        lookback: int = 60,
        risk_free_rate: float = 0.0,
    ):
        super().__init__(config)
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        self._returns: list[float] = []
    
    def reset(self) -> None:
        super().reset()
        self._returns = []
    
    def compute_reward(
        self,
        pnl_delta: float,
        position: float,
        volatility: float,
        transaction_cost: float,
        time_to_resolution: float,
        current_pnl: float,
        outcome: bool | None = None,
        price: float = 0.5,
        portfolio_value: float = 10000,
        **kwargs: Any,
    ) -> float:
        """Compute Sharpe-based reward."""
        # Calculate return
        ret = pnl_delta / (portfolio_value + 1e-8)
        self._returns.append(ret)
        
        # Only compute Sharpe after enough data
        if len(self._returns) < self.lookback:
            # Use standard reward during warmup
            return super().compute_reward(
                pnl_delta, position, volatility, transaction_cost,
                time_to_resolution, current_pnl, outcome, price, **kwargs
            )
        
        # Rolling Sharpe ratio
        recent_returns = np.array(self._returns[-self.lookback:])
        excess_returns = recent_returns - self.risk_free_rate / (365 * 24 * 60)  # Per-minute rate
        
        mean_return = excess_returns.mean()
        std_return = excess_returns.std() + 1e-8
        
        sharpe = mean_return / std_return
        
        # Scale Sharpe to reasonable reward range
        sharpe_reward = np.tanh(sharpe) * self.config.reward_scale
        
        # Add transaction penalty
        sharpe_reward -= self.config.transaction_penalty * transaction_cost / portfolio_value
        
        # Add resolution bonus
        if time_to_resolution < 0.1 and outcome is not None:
            if (position > 0 and outcome) or (position < 0 and not outcome):
                sharpe_reward += self.config.resolution_bonus_weight * 0.1
        
        return float(np.clip(sharpe_reward, self.config.clip_min, self.config.clip_max))


def create_reward_calculator(
    reward_type: str = "standard",
    config: RewardConfig | None = None,
    **kwargs: Any,
) -> RewardCalculator:
    """
    Factory function to create reward calculators.
    
    Args:
        reward_type: Type of reward calculator ("standard", "sharpe")
        config: Reward configuration
        **kwargs: Additional arguments
        
    Returns:
        Reward calculator instance
    """
    if reward_type == "standard":
        return RewardCalculator(config)
    elif reward_type == "sharpe":
        return SharpeRewardCalculator(config, **kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
