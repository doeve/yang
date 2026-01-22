"""
Dynamic Trading Environment for SAC with Entry/Exit Actions.

Extends DeepLOBExecutionEnv with:
- 4D action space: direction, size, hold_prob, exit_signal
- Multi-step episodes within candles
- Risk-aware rewards (Sharpe ratio, max drawdown)
- Enhanced observations (max PnL, drawdown, volatility)

Based on research:
- Avellaneda-Stoikov inventory control
- HARLF hierarchical framework
- Differential Sharpe ratio reward shaping
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DynamicTradingConfig:
    """Configuration for dynamic trading environment."""
    
    # === Position Sizing ===
    initial_balance: float = 1000.0
    max_position_size: float = 0.25
    max_trades_per_candle: int = 5  # Allow multiple entries/exits
    
    # === Trading Costs (Almgren-Chriss) ===
    spread_cost: float = 0.002
    slippage_linear: float = 0.001
    slippage_quadratic: float = 0.0005
    fee_percent: float = 0.1
    
    # === Exit Thresholds ===
    exit_signal_threshold: float = 0.5  # Exit if exit_signal > threshold
    min_confidence_to_trade: float = 0.40
    
    # === Risk Management ===
    max_drawdown_pct: float = 0.15  # 15% max drawdown before penalty
    stop_loss_pct: float = 0.10  # 10% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    
    # === Reward Shaping (NORMALIZED) ===
    # Rewards are now normalized to [-1, 1] range
    win_reward: float = 1.0
    loss_penalty: float = -1.0
    hold_reward: float = 0.1
    sharpe_weight: float = 0.1  # Reduced for normalized rewards
    drawdown_penalty_weight: float = 0.1
    use_dsr_reward: bool = False  # Disable DSR for simpler rewards
    confidence_bonus: float = 0.2  # Bonus for high-confidence correct trades
    dsr_profit_multiplier_base: float = 0.5  # reward = pnl × (0.5 + quality)
    dsr_loss_multiplier_base: float = 1.5  # loss = pnl × (1.5 - quality)
    
    # === Episode Structure ===
    steps_per_candle: int = 60  # Multiple decision points per candle
    
    # === Observation Enhancements ===
    returns_window: int = 20  # Window for Sharpe calculation


@dataclass
class Position:
    """Track a position with PnL history."""
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_step: int
    max_pnl: float = 0.0  # For trailing stop
    min_pnl: float = 0.0


class DeepLOBDynamicEnv(gym.Env):
    """
    Dynamic trading environment with entry/exit actions.
    
    Action Space (4D):
    - direction: [-1, 1] - Short to Long
    - size: [0, 1] - Position size fraction
    - hold_prob: [0, 1] - Hold/wait probability
    - exit_signal: [0, 1] - Exit current position if > threshold
    
    Observation Space (20D):
    - DeepLOB probabilities (3)
    - Predicted class, confidence (2)
    - Market prices, spread (3)
    - Time features (2)
    - Position state (4): position, unrealized_pnl, max_pnl, drawdown
    - History (3): trades_this_candle, win_rate, balance_norm
    - Volatility regime (1)
    - Edge (2): edge_up, edge_down
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[DynamicTradingConfig] = None,
    ):
        super().__init__()
        
        self.config = config or DynamicTradingConfig()
        
        # Observation space: 20 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )
        
        # Action space: 4D [direction, size, hold_prob, exit_signal]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        
        self._reset_state()
        self._returns_history: List[float] = []
        self._trade_outcomes: List[bool] = []
        
        logger.info("DeepLOBDynamicEnv initialized", obs_dim=20, action_dim=4)
    
    def _reset_state(self) -> None:
        """Reset episode state."""
        self._balance = self.config.initial_balance
        self._position: Optional[Position] = None
        self._trades_this_candle = 0
        self._candle_step = 0
        
        # DeepLOB predictions
        self._prob_down = 0.33
        self._prob_hold = 0.34
        self._prob_up = 0.33
        
        # Market prices
        self._market_yes_price = 0.5
        self._market_no_price = 0.5
        self._spread = 0.02
        
        # Volatility regime
        self._volatility = 0.0
        
        # Episode tracking
        self._episode_pnl = 0.0
        self._episode_max_balance = self.config.initial_balance
        self._episode_returns: List[float] = []
        
        # Outcome
        self._outcome: Optional[int] = None
    
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
        # Generate 3-class probabilities
        alpha = self.np_random.uniform(0.5, 2.0, size=3)
        probs = self.np_random.dirichlet(alpha)
        self._prob_down, self._prob_hold, self._prob_up = probs
        
        # Market prices (sometimes mispriced)
        if self.np_random.random() < 0.7:
            self._market_yes_price = 0.5 + (self._prob_up - self._prob_down) * 0.3
        else:
            self._market_yes_price = self.np_random.uniform(0.3, 0.7)
        
        self._market_yes_price = np.clip(self._market_yes_price, 0.1, 0.9)
        self._market_no_price = 1.0 - self._market_yes_price
        self._spread = self.np_random.uniform(0.01, 0.04)
        
        # Volatility regime
        self._volatility = self.np_random.uniform(0.0, 1.0)
        
        # Generate outcome
        outcome_rand = self.np_random.random()
        if outcome_rand < self._prob_down * 0.8:
            self._outcome = 0  # Down
        elif outcome_rand < (self._prob_down + self._prob_hold) * 0.8:
            self._outcome = 1  # Hold
        else:
            self._outcome = 2  # Up
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step with dynamic entry/exit."""
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        hold_prob = float(action[2])
        exit_signal = float(action[3])
        
        reward = 0.0
        terminated = False
        
        time_in_candle = self._candle_step / self.config.steps_per_candle
        time_remaining = 1.0 - time_in_candle
        
        # Update market prices with small random walk
        price_change = self.np_random.normal(0, 0.005)
        self._market_yes_price = np.clip(self._market_yes_price + price_change, 0.1, 0.9)
        self._market_no_price = 1.0 - self._market_yes_price
        
        # === Check for Exit ===
        if self._position is not None:
            # Update unrealized PnL
            unrealized_pnl = self._compute_unrealized_pnl()
            
            # Update max PnL for trailing stop
            if unrealized_pnl > self._position.max_pnl:
                self._position.max_pnl = unrealized_pnl
            if unrealized_pnl < self._position.min_pnl:
                self._position.min_pnl = unrealized_pnl
            
            should_exit = False
            exit_reason = ""
            
            # Agent's exit signal
            if exit_signal > self.config.exit_signal_threshold:
                should_exit = True
                exit_reason = "agent_signal"
            
            # Stop loss
            if unrealized_pnl < -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Take profit
            if unrealized_pnl > self.config.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            
            # Trailing stop (if max PnL > 50% of TP, exit if drops by 30% of max)
            if self._position.max_pnl > self.config.take_profit_pct * 0.5:
                drawdown_from_max = self._position.max_pnl - unrealized_pnl
                if drawdown_from_max > self._position.max_pnl * 0.3:
                    should_exit = True
                    exit_reason = "trailing_stop"
            
            if should_exit:
                reward += self._execute_exit(exit_reason)
                self._trades_this_candle += 1
        
        # === Check for Entry or Hold ===
        if self._position is None:
            if hold_prob > 0.5:
                # Agent chooses to wait
                predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
                if predicted_class == 1:  # Correct to hold when Hold predicted
                    reward += 0.01  # Small bonus for correct hold
            elif self._trades_this_candle < self.config.max_trades_per_candle:
                # Agent wants to enter
                if abs(direction) > 0.1 and size > 0.05:
                    reward += self._execute_entry(direction, size)
                    self._trades_this_candle += 1
        
        # === Advance Step ===
        self._candle_step += 1
        
        # Check for candle end
        if self._candle_step >= self.config.steps_per_candle:
            terminated = True
            # Settle any open position
            if self._position is not None:
                reward += self._execute_settlement()
            
            # Add Sharpe component to reward
            reward += self._compute_sharpe_reward()
            
            # Add drawdown penalty
            reward += self._compute_drawdown_penalty()
        
        # Record return for Sharpe calculation
        if reward != 0:
            self._episode_returns.append(reward)
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _execute_entry(self, direction: float, size: float) -> float:
        """Execute position entry."""
        trade_size = size * self.config.max_position_size
        side = "long" if direction > 0 else "short"
        
        if side == "long":
            entry_price = self._market_yes_price + self._spread * 0.5
        else:
            entry_price = self._market_no_price + self._spread * 0.5
        
        # Apply slippage
        slippage = self.config.slippage_linear * trade_size
        entry_price += slippage
        
        self._position = Position(
            side=side,
            size=trade_size,
            entry_price=entry_price,
            entry_step=self._candle_step,
        )
        
        # Entry reward: small penalty for trading costs
        predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        confidence = max(self._prob_down, self._prob_hold, self._prob_up)
        
        # Small alignment bonus/penalty for entry
        if (side == "long" and predicted_class == 2) or (side == "short" and predicted_class == 0):
            alignment_bonus = 0.05  # Good entry
        elif predicted_class == 1:
            alignment_bonus = -0.1  # Penalty for trading against Hold
        else:
            alignment_bonus = -0.05  # Wrong direction
        
        return np.clip(alignment_bonus, -0.2, 0.2)  # Normalized entry reward
    
    def _execute_exit(self, reason: str) -> float:
        """Execute position exit before settlement."""
        if self._position is None:
            return 0.0
        
        # Calculate exit price
        if self._position.side == "long":
            exit_price = self._market_yes_price - self._spread * 0.5
        else:
            exit_price = self._market_no_price - self._spread * 0.5
        
        # Apply slippage
        slippage = self.config.slippage_linear * self._position.size
        exit_price -= slippage
        
        # Calculate PnL
        if self._position.side == "long":
            pnl = (exit_price - self._position.entry_price) * self._position.size
        else:
            pnl = (self._position.entry_price - exit_price) * self._position.size
        
        # Apply DSR
        if self.config.use_dsr_reward:
            predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
            confidence = max(self._prob_down, self._prob_hold, self._prob_up)
            
            if (self._position.side == "long" and predicted_class == 2) or \
               (self._position.side == "short" and predicted_class == 0):
                quality = confidence
            else:
                quality = 0.3
            
            if pnl > 0:
                multiplier = self.config.dsr_profit_multiplier_base + quality
            else:
                multiplier = self.config.dsr_loss_multiplier_base - quality
            
            pnl = pnl * multiplier
        
        # Bonus for good exit reasons
        if reason == "take_profit":
            pnl += 0.1
        elif reason == "trailing_stop" and pnl > 0:
            pnl += 0.05
        elif reason == "stop_loss":
            pnl += 0.02  # Small bonus for using stop loss
        
        # Update balance and track outcome
        self._trade_outcomes.append(pnl > 0)
        
        # Clear position
        self._position = None
        
        # Normalized exit reward based on win/loss
        if pnl > 0:
            reward = self.config.win_reward * min(1.0, pnl * 2)  # Scale small wins
        else:
            reward = self.config.loss_penalty * min(1.0, abs(pnl) * 2)
        
        # Bonus for good exit reasons
        if reason == "take_profit":
            reward += 0.1
        elif reason == "trailing_stop" and pnl > 0:
            reward += 0.05
        
        return np.clip(reward, -1.5, 1.5)  # Allow some bonus range
    
    def _execute_settlement(self) -> float:
        """Settle position at candle end."""
        if self._position is None:
            return 0.0
        
        # Settlement price based on outcome
        if self._outcome == 2:
            settlement_price = 1.0
        elif self._outcome == 0:
            settlement_price = 0.0
        else:
            settlement_price = 0.5
        
        # Calculate PnL
        if self._position.side == "long":
            pnl = (settlement_price - self._position.entry_price) * self._position.size
        else:
            pnl = (1.0 - settlement_price - self._position.entry_price) * self._position.size
        
        # Apply DSR
        if self.config.use_dsr_reward:
            predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
            confidence = max(self._prob_down, self._prob_hold, self._prob_up)
            
            if (self._position.side == "long" and predicted_class == 2) or \
               (self._position.side == "short" and predicted_class == 0):
                quality = confidence
            else:
                quality = 0.3
            
            if pnl > 0:
                multiplier = self.config.dsr_profit_multiplier_base + quality
            else:
                multiplier = self.config.dsr_loss_multiplier_base - quality
            
        self._trade_outcomes.append(pnl > 0)
        self._position = None
        
        # Normalized settlement reward
        if pnl > 0:
            reward = self.config.win_reward
            # Confidence bonus if prediction was correct
            if (self._position is None):  # Already cleared
                pass
        else:
            reward = self.config.loss_penalty
        
        return np.clip(reward, -1.5, 1.5)
    
    def _compute_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL percentage."""
        if self._position is None:
            return 0.0
        
        if self._position.side == "long":
            current_price = self._market_yes_price
            pnl = (current_price - self._position.entry_price) / self._position.entry_price
        else:
            current_price = self._market_no_price
            pnl = (self._position.entry_price - (1 - current_price)) / self._position.entry_price
        
        return pnl
    
    def _compute_sharpe_reward(self) -> float:
        """Compute Sharpe ratio reward component."""
        if len(self._episode_returns) < 2:
            return 0.0
        
        returns = np.array(self._episode_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8
        
        # Differential Sharpe (annualized approximation)
        sharpe = mean_ret / std_ret
        
        return np.clip(sharpe * self.config.sharpe_weight, -0.2, 0.2)  # Bounded Sharpe reward
    
    def _compute_drawdown_penalty(self) -> float:
        """Compute drawdown penalty."""
        if self._episode_max_balance <= 0:
            return 0.0
        
        drawdown = (self._episode_max_balance - self._balance) / self._episode_max_balance
        
        if drawdown > self.config.max_drawdown_pct:
            penalty = (drawdown - self.config.max_drawdown_pct) * 2.0
            return np.clip(-penalty * self.config.drawdown_penalty_weight, -0.2, 0.0)
        
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Build 20-dimensional observation."""
        time_remaining = 1.0 - self._candle_step / self.config.steps_per_candle
        steps_since_entry = 0
        
        predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
        confidence = max(self._prob_down, self._prob_hold, self._prob_up)
        
        # Position state
        if self._position is not None:
            position_sign = 1.0 if self._position.side == "long" else -1.0
            position_size = self._position.size * position_sign
            unrealized_pnl = self._compute_unrealized_pnl()
            max_pnl = self._position.max_pnl
            drawdown = max_pnl - unrealized_pnl
            steps_since_entry = self._candle_step - self._position.entry_step
        else:
            position_size = 0.0
            unrealized_pnl = 0.0
            max_pnl = 0.0
            drawdown = 0.0
        
        # Edge calculations
        model_implied = 0.5 + (self._prob_up - self._prob_down) * 0.5
        edge_up = model_implied - self._market_yes_price
        edge_down = (1 - model_implied) - self._market_no_price
        
        # Win rate and balance
        win_rate = np.mean(self._trade_outcomes[-20:]) if self._trade_outcomes else 0.5
        balance_norm = (self._balance / self.config.initial_balance) - 1.0
        
        obs = np.array([
            # DeepLOB predictions (3)
            self._prob_down,
            self._prob_hold,
            self._prob_up,
            # Prediction (2)
            predicted_class / 2.0,
            confidence,
            # Market (3)
            self._market_yes_price,
            self._market_no_price,
            self._spread,
            # Time (2)
            time_remaining,
            steps_since_entry / self.config.steps_per_candle,
            # Position (4)
            position_size,
            unrealized_pnl,
            max_pnl,
            drawdown,
            # History (3)
            self._trades_this_candle / self.config.max_trades_per_candle,
            win_rate,
            balance_norm,
            # Volatility (1)
            self._volatility,
            # Edge (2)
            edge_up,
            edge_down,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> dict[str, Any]:
        return {
            "balance": self._balance,
            "position": self._position.side if self._position else None,
            "position_size": self._position.size if self._position else 0,
            "prob_up": self._prob_up,
            "prob_down": self._prob_down,
            "prob_hold": self._prob_hold,
            "predicted_class": np.argmax([self._prob_down, self._prob_hold, self._prob_up]),
            "outcome": self._outcome,
            "episode_pnl": self._episode_pnl,
            "trades_this_candle": self._trades_this_candle,
        }


def make_dynamic_vec_env(
    num_envs: int = 4,
    config: Optional[DynamicTradingConfig] = None,
) -> Any:
    """Create vectorized dynamic trading environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        return DeepLOBDynamicEnv(config=config)
    
    return DummyVecEnv([make_env for _ in range(num_envs)])
