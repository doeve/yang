"""
Gymnasium-compatible trading environment for Polymarket.

Provides a standard RL interface for training trading agents:
- Observation space: price features, position, time
- Action space: discrete or continuous trading actions
- Reward: risk-adjusted returns with transaction cost penalties
"""

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import structlog

from .replay_engine import ReplayConfig, ReplayEngine, MarketState
from .market_simulator import (
    MarketSimulator,
    SimulatorConfig,
    OrderSide,
    Portfolio,
)
from ..data.preprocessor import FeatureConfig

logger = structlog.get_logger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the trading environment."""
    
    # Action space type
    action_type: Literal["discrete", "continuous"] = "discrete"
    
    # Discrete action definitions for Polymarket with fund management
    # Extended action space for learning position sizing and risk management:
    # 0 = hold (keep current position)
    # 1-3 = buy YES with different sizes (small/medium/large)
    # 4-6 = buy NO with different sizes (small/medium/large)
    # 7 = reduce position by 50% (take profit / cut loss)
    # 8 = close all positions (full exit)
    discrete_actions: int = 9
    
    # Position sizing percentages (of available portfolio)
    small_trade_pct: float = 0.05   # 5% - conservative bet
    medium_trade_pct: float = 0.15  # 15% - moderate bet
    large_trade_pct: float = 0.30   # 30% - aggressive bet
    
    # Continuous action range [-1, 1]
    # -1 = full NO, 0 = neutral, +1 = full YES
    max_position_pct: float = 0.5  # Max 50% of portfolio in one position
    
    # Reward configuration - with fund management incentives
    reward_scale: float = 0.01  # Scale down rewards to reasonable range
    reward_pnl_weight: float = 1.0  # Main driver is PnL
    reward_risk_penalty: float = 0.0  # Disabled - was hurting exploration
    reward_transaction_penalty: float = 0.001  # Minimal transaction penalty
    reward_holding_penalty: float = 0.0  # Disabled - was hurting learning
    reward_resolution_bonus: float = 1.0  # Bonus for correct direction at resolution
    reward_drawdown_penalty: float = 0.1  # Penalize large drawdowns
    reward_capital_preservation: float = 0.01  # Small bonus for preserving capital
    
    # Episode settings
    max_steps: int = 1000  # Shorter episodes for faster learning
    random_start: bool = True
    
    # Feature settings (MUST MATCH d3v binance-client.ts)
    sequence_length: int = 300  # 300 points = 30 seconds at 100ms (matches d3v historyLength)
    include_portfolio_state: bool = True
    
    # Replay configuration
    replay_config: ReplayConfig | None = None
    
    # Simulator configuration
    simulator_config: SimulatorConfig | None = None


class PolymarketEnv(gym.Env):
    """
    Gymnasium environment for Polymarket direction prediction markets.
    
    Designed for markets like "Will BTC be Up or Down in 15 minutes?"
    where YES = price goes up, NO = price goes down.
    
    **Includes fund management training**: position sizing, risk management.
    
    Observation Space (Box):
        - Feature sequence: (seq_len, num_features) flattened
        - Current features: (num_features,)
        - Portfolio state: position, unrealized_pnl, cash_ratio, exposure
        - Time features: time_to_resolution, progress
    
    Action Space (9 discrete actions for fund management):
        0: Hold (keep current position)
        1: Buy YES (small, 5%) - conservative UP bet
        2: Buy YES (medium, 15%) - moderate UP bet
        3: Buy YES (large, 30%) - aggressive UP bet
        4: Buy NO (small, 5%) - conservative DOWN bet
        5: Buy NO (medium, 15%) - moderate DOWN bet
        6: Buy NO (large, 30%) - aggressive DOWN bet
        7: Reduce position 50% - take profit / cut loss
        8: Close all positions - full exit
        
        Continuous mode: Box(-1, 1) where -1 = full NO, 0 = neutral, +1 = full YES
    
    Reward:
        PnL-based with drawdown penalties and capital preservation bonus.
    
    Example:
        env = PolymarketEnv(price_data, config)
        obs, info = env.reset()
        
        for _ in range(1000):
            action = model.predict(obs)  # 0-8 for discrete
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        price_data: Any,  # pd.DataFrame or dict of DataFrames
        config: EnvConfig | None = None,
        market_id: str = "default",
        resolution_at: Any = None,  # datetime
        created_at: Any = None,  # datetime
        outcome: bool | None = None,
        render_mode: str | None = None,
    ):
        """
        Initialize the environment.
        
        Args:
            price_data: DataFrame with timestamp, price, volume columns
            config: Environment configuration
            market_id: Market identifier
            resolution_at: Market resolution datetime
            created_at: Market creation datetime
            outcome: True/False outcome (for resolved markets)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Initialize replay engine
        replay_config = self.config.replay_config or ReplayConfig(
            max_steps=self.config.max_steps,
            random_start=self.config.random_start,
            feature_config=FeatureConfig(sequence_length=self.config.sequence_length),
        )
        
        self.replay = ReplayEngine(
            price_data,
            config=replay_config,
            market_id=market_id,
            resolution_at=resolution_at,
            created_at=created_at,
            outcome=outcome,
        )
        
        # Initialize market simulator
        simulator_config = self.config.simulator_config or SimulatorConfig()
        self.simulator = MarketSimulator(simulator_config)
        
        # Store metadata
        self.market_id = market_id
        self.outcome = outcome
        
        # Define observation space
        self._setup_observation_space()
        
        # Define action space
        self._setup_action_space()
        
        # Tracking
        self._episode_rewards: list[float] = []
        self._episode_trades: int = 0
        self._last_price: float = 0.5
        self._last_pnl: float = 0.0
        self._returns_history: list[float] = []  # For rolling Sharpe calculation
        self._initial_portfolio_value: float = 0.0
        
        logger.info(
            "PolymarketEnv initialized",
            market_id=market_id,
            action_type=self.config.action_type,
            obs_dim=self._obs_dim,
        )
    
    def _setup_observation_space(self) -> None:
        """
        Setup the observation space for PURE ML approach.
        
        Matches d3v binance-client.ts priceHistory format:
        - Raw normalized price history (seq_len points)
        - Volume history (seq_len points)  
        - Current price change (1)
        - Portfolio state (4)
        - Time features (2)
        """
        seq_len = self.config.sequence_length
        
        # Pure ML observation dimensions
        price_history_dim = seq_len  # Raw normalized prices
        volume_history_dim = seq_len  # Raw normalized volumes
        current_price_dim = 1  # Current price change from start
        portfolio_dim = 4 if self.config.include_portfolio_state else 0
        time_dim = 2  # time_to_resolution, progress
        
        self._obs_dim = price_history_dim + volume_history_dim + current_price_dim + portfolio_dim + time_dim
        self._feature_dim = 2  # Price + Volume (pure ML, not computed features)
        self._seq_len = seq_len
        
        # Define observation space bounds
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
    
    def _setup_action_space(self) -> None:
        """Setup the action space."""
        if self.config.action_type == "discrete":
            self.action_space = spaces.Discrete(self.config.discrete_actions)
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Optional reset options
            
        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)
        
        # Reset replay and simulator
        state = self.replay.reset(seed=seed)
        self.simulator.reset()
        
        # Reset tracking
        self._episode_rewards = []
        self._episode_trades = 0
        self._last_price = state.price
        self._last_pnl = 0.0
        self._returns_history = []
        self._initial_portfolio_value = self.simulator.portfolio.cash
        self._episode_start_price = state.price  # For pure ML observation normalization
        
        # Get initial observation
        obs = self._get_observation(state)
        info = self._get_info(state)
        
        return obs, info
    
    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current state
        current_state = self.replay.get_state()
        current_price = current_state.price
        
        # Execute action
        trade_cost = self._execute_action(action, current_price)
        
        # Advance simulation
        next_state = self.replay.step()
        next_price = next_state.price
        
        # Process any pending orders
        self.simulator.process_orders(
            next_price,
            self.market_id,
            self.replay.current_step,
        )
        
        # Calculate reward
        reward = self._calculate_reward(
            current_price,
            next_price,
            trade_cost,
            next_state,
        )
        self._episode_rewards.append(reward)
        
        # Check termination
        terminated = False
        truncated = False
        
        if self.replay.done:
            # Episode ended
            if self.outcome is not None:
                # Settle the market if we know the outcome
                settlement_price = 1.0 if self.outcome else 0.0
                self.simulator.settle_market(
                    self.market_id,
                    self.outcome,
                    settlement_price,
                )
            terminated = True
        
        # Get observation
        obs = self._get_observation(next_state)
        info = self._get_info(next_state)
        
        # Update tracking
        self._last_price = next_price
        self._last_pnl = self.simulator.portfolio.total_pnl
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int | np.ndarray, current_price: float) -> float:
        """
        Execute the given action using Polymarket YES/NO semantics.
        
        For direction prediction markets:
        - YES = betting price goes UP (buy YES shares)
        - NO = betting price goes DOWN (buy NO shares)
        
        In our simulation:
        - BUY (long) = YES position (profit if price goes up)
        - SELL (short) = NO position (profit if price goes down)
        
        Returns:
            Transaction cost incurred
        """
        portfolio_value = self.simulator.portfolio.total_value({self.market_id: current_price})
        position = self.simulator.portfolio.get_position(self.market_id)
        
        trade_cost = 0.0
        
        if self.config.action_type == "discrete":
            action_idx = int(action)
            
            if action_idx == 0:  # Hold - no action
                pass
            
            elif action_idx == 1:  # Buy YES (small, 5%) - conservative UP bet
                size = (portfolio_value * self.config.small_trade_pct) / current_price
                order = self.simulator.submit_order(self.market_id, OrderSide.BUY, size)
                if order:
                    trade_cost = size * current_price * 0.002
                    self._episode_trades += 1
            
            elif action_idx == 2:  # Buy YES (medium, 15%) - moderate UP bet
                size = (portfolio_value * self.config.medium_trade_pct) / current_price
                order = self.simulator.submit_order(self.market_id, OrderSide.BUY, size)
                if order:
                    trade_cost = size * current_price * 0.002
                    self._episode_trades += 1
            
            elif action_idx == 3:  # Buy YES (large, 30%) - aggressive UP bet
                size = (portfolio_value * self.config.large_trade_pct) / current_price
                order = self.simulator.submit_order(self.market_id, OrderSide.BUY, size)
                if order:
                    trade_cost = size * current_price * 0.002
                    self._episode_trades += 1
            
            elif action_idx == 4:  # Buy NO (small, 5%) - conservative DOWN bet
                size = (portfolio_value * self.config.small_trade_pct) / current_price
                order = self.simulator.submit_order(self.market_id, OrderSide.SELL, size)
                if order:
                    trade_cost = size * current_price * 0.002
                    self._episode_trades += 1
            
            elif action_idx == 5:  # Buy NO (medium, 15%) - moderate DOWN bet
                size = (portfolio_value * self.config.medium_trade_pct) / current_price
                order = self.simulator.submit_order(self.market_id, OrderSide.SELL, size)
                if order:
                    trade_cost = size * current_price * 0.002
                    self._episode_trades += 1
            
            elif action_idx == 6:  # Buy NO (large, 30%) - aggressive DOWN bet
                size = (portfolio_value * self.config.large_trade_pct) / current_price
                order = self.simulator.submit_order(self.market_id, OrderSide.SELL, size)
                if order:
                    trade_cost = size * current_price * 0.002
                    self._episode_trades += 1
            
            elif action_idx == 7:  # Reduce position by 50% - take profit / cut loss
                if abs(position.size) > 0.01:
                    reduce_size = abs(position.size) * 0.5
                    if position.size > 0:
                        order = self.simulator.submit_order(self.market_id, OrderSide.SELL, reduce_size)
                    else:
                        order = self.simulator.submit_order(self.market_id, OrderSide.BUY, reduce_size)
                    if order:
                        trade_cost = reduce_size * current_price * 0.002
                        self._episode_trades += 1
            
            elif action_idx == 8:  # Close all positions - full exit
                if abs(position.size) > 0.01:
                    order = self.simulator.submit_close_position(self.market_id)
                    if order:
                        trade_cost = abs(position.size) * current_price * 0.002
                        self._episode_trades += 1
        
        else:
            # Continuous action: [-1, 1] maps to full NO to full YES
            # -1 = max NO position, 0 = neutral, +1 = max YES position
            action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            target_position = action_val * self.config.max_position_pct * portfolio_value / current_price
            current_position = position.size
            delta = target_position - current_position
            
            if abs(delta) > 0.01:
                if delta > 0:
                    order = self.simulator.submit_order(self.market_id, OrderSide.BUY, abs(delta))
                else:
                    order = self.simulator.submit_order(self.market_id, OrderSide.SELL, abs(delta))
                
                if order:
                    trade_cost = abs(delta) * current_price * 0.002
                    self._episode_trades += 1
        
        return trade_cost
    
    def _calculate_reward(
        self,
        current_price: float,
        next_price: float,
        trade_cost: float,
        state: MarketState,
    ) -> float:
        """
        Calculate reward for the step with Sharpe-like risk adjustment.
        
        Based on RL trading research:
        - Primary: PnL change
        - Risk-adjusted: Rolling Sharpe bonus
        - Drawdown penalty
        - Transaction costs
        """
        # Get portfolio state
        portfolio = self.simulator.portfolio
        position = portfolio.get_position(self.market_id)
        current_value = portfolio.total_value({self.market_id: next_price})
        
        # 1. Calculate return for this step
        current_pnl = portfolio.total_pnl
        pnl_delta = current_pnl - self._last_pnl
        
        # Track returns for Sharpe calculation
        step_return = pnl_delta / (self._initial_portfolio_value + 1e-8)
        self._returns_history.append(step_return)
        
        # 2. PnL reward (main driver)
        pnl_reward = self.config.reward_pnl_weight * pnl_delta
        
        # 3. Rolling Sharpe bonus (rewards consistent positive returns)
        sharpe_bonus = 0.0
        if len(self._returns_history) >= 20:
            recent_returns = np.array(self._returns_history[-20:])
            mean_return = recent_returns.mean()
            std_return = recent_returns.std() + 1e-8
            rolling_sharpe = mean_return / std_return
            # Tanh to bound the bonus
            sharpe_bonus = 0.1 * np.tanh(rolling_sharpe)
        
        # 4. Transaction penalty (minimal)
        transaction_penalty = self.config.reward_transaction_penalty * trade_cost
        
        # 5. Drawdown penalty
        drawdown_penalty = 0.0
        if portfolio.max_drawdown > 0.1:
            drawdown_penalty = self.config.reward_drawdown_penalty * (portfolio.max_drawdown - 0.1)
        
        # 6. Correct direction bonus (simple: did price move in our favor?)
        direction_bonus = 0.0
        price_change = next_price - current_price
        if position.size > 0 and price_change > 0:
            direction_bonus = 0.01  # Long and price went up
        elif position.size < 0 and price_change < 0:
            direction_bonus = 0.01  # Short and price went down
        
        # Total reward
        reward = pnl_reward + sharpe_bonus + direction_bonus - transaction_penalty - drawdown_penalty
        reward *= self.config.reward_scale
        
        return float(reward)
    
    def _get_observation(self, state: MarketState) -> np.ndarray:
        """
        Build observation vector from current state.
        
        PURE ML APPROACH: Uses raw normalized price history
        matching d3v binance-client.ts priceHistory format.
        
        Observation format:
        - 300 normalized price points (30 seconds at 100ms in live)
        - Volume history (optional)
        - Portfolio state
        - Time features
        """
        obs_parts = []
        
        # 1. RAW PRICE HISTORY (normalized to % change from first price)
        # This matches d3v's priceHistory: 300 points of raw prices
        # We normalize by dividing by the first price to get relative changes
        price_seq = state.feature_sequence[:, 0]  # First column is typically price
        if len(price_seq) > 0 and price_seq[0] != 0:
            # Normalize: convert to % change from start of window
            normalized_prices = (price_seq / price_seq[0]) - 1.0
            # Scale to reasonable range for neural network
            normalized_prices = normalized_prices * 100  # % as value
        else:
            normalized_prices = np.zeros(len(price_seq), dtype=np.float32)
        obs_parts.append(normalized_prices)
        
        # 2. VOLUME HISTORY (normalized)
        if state.feature_sequence.shape[1] > 1:
            volume_seq = state.feature_sequence[:, 1] if state.feature_sequence.shape[1] > 1 else np.zeros(len(price_seq))
            if volume_seq.max() > 0:
                normalized_volume = volume_seq / (volume_seq.max() + 1e-8)
            else:
                normalized_volume = np.zeros(len(volume_seq), dtype=np.float32)
            obs_parts.append(normalized_volume)
        
        # 3. CURRENT PRICE CHANGE (since episode start)
        current_price = state.price
        start_price = self._initial_portfolio_value / 10  # Rough estimate
        if hasattr(self, '_episode_start_price') and self._episode_start_price > 0:
            start_price = self._episode_start_price
        price_change = (current_price / (start_price + 1e-8)) - 1.0
        obs_parts.append(np.array([price_change * 100], dtype=np.float32))
        
        # 4. PORTFOLIO STATE (position sizing context)
        if self.config.include_portfolio_state:
            port_state = self.simulator.get_state(self.market_id)
            obs_parts.append(np.array([
                port_state["position_normalized"],
                port_state["unrealized_pnl_normalized"],
                port_state["cash_ratio"],
                port_state["exposure"],
            ], dtype=np.float32))
        
        # 5. TIME FEATURES (how much time left in window)
        obs_parts.append(np.array([
            state.time_to_resolution,
            self.replay.progress,
        ], dtype=np.float32))
        
        # Concatenate all parts
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        # Clip to reasonable bounds
        obs = np.clip(obs, -10.0, 10.0)
        
        return obs
    
    def _get_info(self, state: MarketState) -> dict[str, Any]:
        """Build info dictionary."""
        portfolio = self.simulator.portfolio
        position = portfolio.get_position(self.market_id)
        
        return {
            "timestamp": state.timestamp.isoformat(),
            "price": state.price,
            "position": position.size,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
            "total_pnl": portfolio.total_pnl,
            "cash": portfolio.cash,
            "total_trades": portfolio.total_trades,
            "max_drawdown": portfolio.max_drawdown,
            "time_to_resolution": state.time_to_resolution,
            "episode_trades": self._episode_trades,
            "episode_reward_sum": sum(self._episode_rewards),
        }
    
    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Return a simple visualization
            # This would be expanded for actual visualization
            return np.zeros((400, 600, 3), dtype=np.uint8)
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass


def make_env(
    price_data: Any,
    config: EnvConfig | None = None,
    **kwargs: Any,
) -> PolymarketEnv:
    """
    Factory function to create a PolymarketEnv.
    
    Args:
        price_data: DataFrame with price data
        config: Environment configuration
        **kwargs: Additional arguments passed to PolymarketEnv
        
    Returns:
        Configured environment
    """
    return PolymarketEnv(price_data, config=config, **kwargs)


def make_vec_env(
    market_data: dict[str, Any],
    num_envs: int = 4,
    config: EnvConfig | None = None,
) -> Any:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        market_data: Dict mapping market_id to (price_df, metadata) tuples
        num_envs: Number of parallel environments
        config: Shared environment configuration
        
    Returns:
        Vectorized environment
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    market_ids = list(market_data.keys())
    
    def make_env_fn(idx: int) -> callable:
        def _init() -> PolymarketEnv:
            # Round-robin market selection
            market_id = market_ids[idx % len(market_ids)]
            price_df, metadata = market_data[market_id]
            
            return PolymarketEnv(
                price_df,
                config=config,
                market_id=market_id,
                resolution_at=metadata.get("resolution_at"),
                created_at=metadata.get("created_at"),
                outcome=metadata.get("outcome"),
            )
        return _init
    
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)
