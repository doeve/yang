"""
Token-Centric Trading Environment for Polymarket.

This environment focuses on token price dynamics rather than BTC prediction.
The agent learns to:
1. Identify mispricing (edge detection)
2. Time entries/exits based on token price movement
3. Exploit convergence toward settlement (0 or 1)
4. Manage theta decay (time value erosion)

Key differences from BTC-centric env:
- Observation: token features + edge signal (not BTC prediction)
- Reward: actual token P&L (not prediction accuracy)
- Dynamics: realistic token price convergence
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import structlog

from src.data.token_features import TokenFeatureBuilder, TokenFeatureConfig

logger = structlog.get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    side: str  # "long" (YES) or "short" (NO)
    size: float  # Position size as fraction of portfolio
    entry_price: float  # Entry price of the token
    entry_step: int  # Step when position was opened
    max_pnl: float = 0.0  # Maximum unrealized PnL
    prev_pnl: float = 0.0  # Previous step PnL for improvement tracking


@dataclass
class TokenTradingConfig:
    """Configuration for token trading environment."""

    # Episode structure
    steps_per_candle: int = 180  # 15 min × 12 steps/min = 180 (5-second steps)
    max_trades_per_candle: int = 5

    # Position sizing
    max_position_size: float = 0.25  # Max 25% of portfolio

    # Transaction costs (Almgren-Chriss style)
    spread_cost: float = 0.005  # 0.5% bid-ask spread
    slippage: float = 0.002  # 0.2% slippage
    fee_percent: float = 0.001  # 0.1% fee

    # Risk management
    stop_loss_pct: float = 0.15  # 15% stop loss
    take_profit_pct: float = 0.25  # 25% take profit

    # Reward shaping
    win_reward: float = 1.0
    loss_penalty: float = -1.0
    hold_reward: float = 0.1  # Reward for holding profitable position
    edge_bonus: float = 0.2  # Bonus for trading with positive edge
    convergence_bonus: float = 0.3  # Bonus for holding in convergence zone

    # Price simulation (for training)
    price_noise: float = 0.02  # Base noise level
    mean_reversion_strength: float = 0.05  # Mean reversion in early candle
    convergence_acceleration: float = 0.1  # Convergence speed increase over time


class TokenTradingEnv(gym.Env):
    """
    Gymnasium environment for token-centric trading.

    Observation Space (51 + 4 = 55 dimensions):
    - Token features from TokenFeatureBuilder (51 dims, includes EMA-smoothed momentum + non-linear BTC)
    - Position state (4 dims): has_position, position_side, position_pnl, steps_in_position

    Action Space (4 dimensions, continuous):
    - direction: [-1, 1] → -1 = buy NO, +1 = buy YES
    - size: [0, 1] → position size fraction
    - hold_prob: [0, 1] → probability of waiting (if > 0.5, don't trade)
    - exit_signal: [0, 1] → signal to exit (if > threshold, exit)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[TokenTradingConfig] = None,
        feature_builder: Optional[TokenFeatureBuilder] = None,
    ):
        super().__init__()

        self.config = config or TokenTradingConfig()
        self.feature_builder = feature_builder or TokenFeatureBuilder()

        # Observation: token features + position state
        obs_dim = self.feature_builder.feature_dim + 4  # +4 for position state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: direction, size, hold_prob, exit_signal
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # State variables (initialized in reset)
        self._position: Optional[Position] = None
        self._candle_step = 0
        self._trades_this_candle = 0
        self._episode_returns: List[float] = []
        self._trade_outcomes: List[bool] = []

        # Price histories
        self._yes_prices: List[float] = []
        self._no_prices: List[float] = []
        self._btc_prices: List[float] = []

        # Simulated market state
        self._outcome = 0  # 0 = NO wins, 1 = YES wins
        self._convergence_target = 0.0

        logger.info(
            "TokenTradingEnv initialized",
            obs_dim=obs_dim,
            action_dim=4,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset state
        self._position = None
        self._candle_step = 0
        self._trades_this_candle = 0
        self._episode_returns = []
        self._trade_outcomes = []

        # Generate random market scenario
        self._generate_market_scenario()

        return self._get_observation(), self._get_info()

    def _generate_market_scenario(self):
        """Generate random market scenario for training."""
        # Random outcome (YES wins or NO wins)
        self._outcome = self.np_random.choice([0, 1])
        self._convergence_target = float(self._outcome)  # 0.0 or 1.0

        # Initial price near 0.5 with some randomness
        initial_yes = 0.5 + self.np_random.uniform(-0.1, 0.1)
        initial_yes = np.clip(initial_yes, 0.2, 0.8)

        # Initialize price histories
        self._yes_prices = [initial_yes]
        self._no_prices = [1.0 - initial_yes]

        # BTC prices (random walk, used for guidance features)
        btc_base = 100000.0
        self._btc_prices = [btc_base]
        self._btc_open = btc_base

    def _simulate_price_step(self, time_in_candle: float):
        """Simulate realistic token price dynamics."""
        current_yes = self._yes_prices[-1]

        # Three-phase dynamics
        if time_in_candle < 0.3:
            # Phase 1: Mean reversion (uncertainty)
            mean_reversion = (0.5 - current_yes) * self.config.mean_reversion_strength
            noise = self.np_random.normal(0, self.config.price_noise)
            weak_trend = (self._convergence_target - current_yes) * 0.01
            price_change = mean_reversion + noise + weak_trend

        elif time_in_candle < 0.7:
            # Phase 2: Trending
            trend_strength = 0.03 + (time_in_candle - 0.3) * 0.05
            trend = (self._convergence_target - current_yes) * trend_strength
            noise = self.np_random.normal(0, self.config.price_noise * 0.7)
            price_change = trend + noise

        else:
            # Phase 3: Strong convergence
            convergence = self.config.convergence_acceleration * (time_in_candle - 0.7) + 0.05
            trend = (self._convergence_target - current_yes) * convergence
            noise = self.np_random.normal(0, self.config.price_noise * 0.3)
            price_change = trend + noise

            # Accelerate in final 10%
            if time_in_candle > 0.9:
                extra_convergence = (time_in_candle - 0.9) * 0.5
                price_change += (self._convergence_target - current_yes) * extra_convergence

        # Update prices
        new_yes = np.clip(current_yes + price_change, 0.01, 0.99)
        self._yes_prices.append(new_yes)
        self._no_prices.append(1.0 - new_yes)

        # Simulate BTC price (correlated with outcome)
        btc_direction = 1 if self._outcome == 1 else -1
        btc_change = btc_direction * abs(self.np_random.normal(0, 50)) + self.np_random.normal(0, 20)
        new_btc = self._btc_prices[-1] + btc_change
        self._btc_prices.append(new_btc)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        hold_prob = float(action[2])
        exit_signal = float(action[3])

        reward = 0.0
        terminated = False

        time_in_candle = self._candle_step / self.config.steps_per_candle
        time_remaining = 1.0 - time_in_candle

        # Simulate price movement
        self._simulate_price_step(time_in_candle)
        current_yes = self._yes_prices[-1]
        current_no = self._no_prices[-1]

        # === EXIT LOGIC ===
        if self._position is not None:
            unrealized_pnl = self._compute_unrealized_pnl()

            # Update max PnL
            if unrealized_pnl > self._position.max_pnl:
                self._position.max_pnl = unrealized_pnl

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            # Dynamic exit threshold (higher in convergence zone)
            exit_threshold = 0.5
            if current_yes > 0.85 or current_yes < 0.15:
                exit_threshold = 0.7  # Harder to exit in convergence zone

            if exit_signal > exit_threshold:
                should_exit = True
                exit_reason = "agent_signal"

            if unrealized_pnl < -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"

            if unrealized_pnl > self.config.take_profit_pct and time_remaining > 0.2:
                should_exit = True
                exit_reason = "take_profit"

            # Trailing stop
            if self._position.max_pnl > 0.1 and unrealized_pnl < self._position.max_pnl * 0.5:
                should_exit = True
                exit_reason = "trailing_stop"

            if should_exit:
                reward += self._execute_exit(exit_reason)
            else:
                # Hold reward
                reward += self._compute_hold_reward(unrealized_pnl, time_in_candle)
                self._position.prev_pnl = unrealized_pnl

        # === ENTRY LOGIC ===
        if self._position is None:
            if hold_prob > 0.5:
                # Agent chooses to wait
                reward += self._compute_wait_reward(time_in_candle)
            elif self._trades_this_candle < self.config.max_trades_per_candle:
                if abs(direction) > 0.1 and size > 0.05:
                    reward += self._execute_entry(direction, size)
                    self._trades_this_candle += 1

        # Check for settlement
        self._candle_step += 1
        if self._candle_step >= self.config.steps_per_candle:
            if self._position is not None:
                reward += self._execute_settlement()
            terminated = True

        # Track episode returns
        self._episode_returns.append(reward)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_entry(self, direction: float, size: float) -> float:
        """Execute position entry."""
        trade_size = size * self.config.max_position_size
        side = "long" if direction > 0 else "short"

        if side == "long":
            entry_price = self._yes_prices[-1] + self.config.spread_cost / 2
        else:
            entry_price = self._no_prices[-1] + self.config.spread_cost / 2

        # Apply slippage
        entry_price += self.config.slippage * trade_size

        self._position = Position(
            side=side,
            size=trade_size,
            entry_price=entry_price,
            entry_step=self._candle_step,
        )

        # Entry cost penalty
        return -self.config.fee_percent * trade_size

    def _execute_exit(self, reason: str) -> float:
        """Execute position exit."""
        if self._position is None:
            return 0.0

        if self._position.side == "long":
            exit_price = self._yes_prices[-1] - self.config.spread_cost / 2
        else:
            exit_price = self._no_prices[-1] - self.config.spread_cost / 2

        # Apply slippage
        exit_price -= self.config.slippage * self._position.size

        # Calculate PnL
        if self._position.side == "long":
            pnl = (exit_price - self._position.entry_price) / self._position.entry_price
        else:
            pnl = (exit_price - self._position.entry_price) / self._position.entry_price

        self._trade_outcomes.append(pnl > 0)
        self._position = None

        # Reward based on PnL
        if pnl > 0:
            return self.config.win_reward * min(pnl * 5, 1.0)
        else:
            return self.config.loss_penalty * min(abs(pnl) * 5, 1.0)

    def _execute_settlement(self) -> float:
        """Execute settlement at candle end."""
        if self._position is None:
            return 0.0

        # Settlement price
        settlement_price = float(self._outcome)  # 1.0 if YES wins, 0.0 if NO wins

        # Calculate PnL
        if self._position.side == "long":
            pnl = (settlement_price - self._position.entry_price) * self._position.size
        else:
            # Short NO token: profit if NO wins (settlement = 0 for YES)
            pnl = ((1.0 - settlement_price) - self._position.entry_price) * self._position.size

        self._trade_outcomes.append(pnl > 0)
        self._position = None

        # Reward with magnitude bonus
        magnitude = abs(pnl) * 5
        if pnl > 0:
            return self.config.win_reward * (0.5 + min(magnitude, 1.5))
        else:
            return self.config.loss_penalty * (0.5 + min(magnitude, 1.5))

    def _compute_unrealized_pnl(self) -> float:
        """Compute unrealized PnL."""
        if self._position is None:
            return 0.0

        if self._position.side == "long":
            current_price = self._yes_prices[-1]
        else:
            current_price = self._no_prices[-1]

        return (current_price - self._position.entry_price) / self._position.entry_price

    def _compute_hold_reward(self, unrealized_pnl: float, time_in_candle: float) -> float:
        """Compute reward for holding position."""
        reward = 0.0

        # Time urgency multiplier
        time_urgency = 1.0 + time_in_candle

        if unrealized_pnl > 0:
            # Reward for holding profitable position
            reward = self.config.hold_reward * min(unrealized_pnl * 5, 1.0) * time_urgency

            # Convergence bonus
            current_yes = self._yes_prices[-1]
            if (self._position.side == "long" and current_yes > 0.75) or \
               (self._position.side == "short" and current_yes < 0.25):
                reward += self.config.convergence_bonus * time_urgency

        elif unrealized_pnl > self._position.prev_pnl:
            # Small reward for improving position
            improvement = unrealized_pnl - self._position.prev_pnl
            reward = self.config.hold_reward * 0.3 * min(improvement * 10, 1.0)

        return reward

    def _compute_wait_reward(self, time_in_candle: float) -> float:
        """Compute reward for waiting (not entering)."""
        reward = 0.01  # Base wait reward

        # Bonus for waiting early (more information to come)
        if time_in_candle < 0.3:
            reward += 0.02 * (0.3 - time_in_candle) / 0.3

        # Bonus for waiting during high volatility
        if len(self._yes_prices) > 10:
            volatility = np.std(self._yes_prices[-10:])
            reward += min(volatility * 2, 0.03)

        return reward

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        # Token features
        time_remaining = 1.0 - self._candle_step / self.config.steps_per_candle

        token_features = self.feature_builder.compute_features(
            yes_prices=np.array(self._yes_prices),
            no_prices=np.array(self._no_prices),
            time_remaining=time_remaining,
            btc_prices=np.array(self._btc_prices),
            btc_open=self._btc_open,
        )

        # Position state
        if self._position is not None:
            has_position = 1.0
            position_side = 1.0 if self._position.side == "long" else -1.0
            position_pnl = self._compute_unrealized_pnl()
            steps_in_position = (self._candle_step - self._position.entry_step) / self.config.steps_per_candle
        else:
            has_position = 0.0
            position_side = 0.0
            position_pnl = 0.0
            steps_in_position = 0.0

        position_state = np.array([has_position, position_side, position_pnl, steps_in_position])

        # Combine
        obs = np.concatenate([token_features, position_state])
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            "step": self._candle_step,
            "time_remaining": 1.0 - self._candle_step / self.config.steps_per_candle,
            "yes_price": self._yes_prices[-1] if self._yes_prices else 0.5,
            "outcome": self._outcome,
            "has_position": self._position is not None,
            "trades_this_candle": self._trades_this_candle,
            "win_rate": np.mean(self._trade_outcomes) if self._trade_outcomes else 0.0,
        }


def make_token_envs(
    num_envs: int = 4,
    config: Optional[TokenTradingConfig] = None,
) -> Any:
    """Create vectorized token trading environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return TokenTradingEnv(config=config)

    return DummyVecEnv([make_env for _ in range(num_envs)])
