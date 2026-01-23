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

    # === Exit Thresholds (now dynamic) ===
    exit_signal_base_threshold: float = 0.5  # Base threshold, adjusted dynamically
    exit_convergence_zone: float = 0.85  # Price above this = convergence zone
    exit_time_lock_fraction: float = 0.2  # Lock exits in final 20% of candle when winning
    min_confidence_to_trade: float = 0.40

    # === Risk Management ===
    max_drawdown_pct: float = 0.15  # 15% max drawdown before penalty
    stop_loss_pct: float = 0.10  # 10% stop loss
    take_profit_pct: float = 0.15  # 15% take profit

    # === Reward Shaping (NORMALIZED) ===
    # Rewards are now normalized to [-1, 1] range
    win_reward: float = 1.0
    loss_penalty: float = -1.0
    hold_reward: float = 0.15  # Increased to make waiting more viable
    sharpe_weight: float = 0.1  # Reduced for normalized rewards
    drawdown_penalty_weight: float = 0.1
    use_dsr_reward: bool = True  # Enable symmetric DSR for quality-weighted rewards
    confidence_bonus: float = 0.2  # Bonus for high-confidence correct trades
    # Note: DSR multipliers now use symmetric formula: 0.5 + quality * 0.5
    # This avoids amplifying losses which caused poor recovery behavior

    # === Churn Penalty ===
    churn_threshold: int = 3  # Penalize after this many trades per candle
    churn_penalty: float = 0.05  # Penalty per excess trade

    # === Time Urgency for Hold Rewards ===
    time_urgency_multiplier: float = 2.0  # Max multiplier for hold reward near settlement

    # === Settlement Reward Scaling ===
    settlement_magnitude_scale: float = 2.0  # Scale settlement reward by PnL magnitude

    # === Episode Structure ===
    steps_per_candle: int = 60  # Multiple decision points per candle

    # === Observation Enhancements ===
    returns_window: int = 20  # Window for Sharpe calculation
    momentum_window: int = 5  # Window for price momentum calculation


@dataclass
class Position:
    """Track a position with PnL history."""
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_step: int
    max_pnl: float = 0.0  # For trailing stop
    min_pnl: float = 0.0
    prev_pnl: float = 0.0  # For tracking PnL improvement
    entry_token_price: float = 0.5  # Token price at entry for momentum tracking


class DeepLOBDynamicEnv(gym.Env):
    """
    Dynamic trading environment with entry/exit actions.

    Action Space (4D):
    - direction: [-1, 1] - Short to Long
    - size: [0, 1] - Position size fraction
    - hold_prob: [0, 1] - Hold/wait probability
    - exit_signal: [0, 1] - Exit current position if > threshold

    Observation Space (26D):
    - DeepLOB probabilities (3)
    - Predicted class, confidence (2)
    - Market prices, spread (3)
    - Time features (2)
    - Position state (4): position, unrealized_pnl, max_pnl, drawdown
    - History (3): trades_this_candle, win_rate, balance_norm
    - Volatility regime (1)
    - Edge (2): edge_up, edge_down
    - NEW: Momentum features (2): yes_momentum, no_momentum
    - NEW: Convergence features (2): convergence_velocity, price_distance_to_settlement
    - NEW: Arbitrage signal (1): yes_no_sum_deviation
    - NEW: Time urgency (1): exponential urgency as settlement approaches
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[DynamicTradingConfig] = None,
    ):
        super().__init__()

        self.config = config or DynamicTradingConfig()

        # Observation space: 26 dimensions (was 20)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),
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

        # Price history for momentum calculation
        self._yes_price_history: List[float] = []
        self._no_price_history: List[float] = []

        logger.info("DeepLOBDynamicEnv initialized", obs_dim=26, action_dim=4)
    
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

        # Price history for momentum (reset)
        self._yes_price_history = [0.5]
        self._no_price_history = [0.5]

        # Volatility regime
        self._volatility = 0.0

        # Episode tracking
        self._episode_pnl = 0.0
        self._episode_max_balance = self.config.initial_balance
        self._episode_returns: List[float] = []

        # Outcome
        self._outcome: Optional[int] = None

        # Convergence target (set based on outcome in _generate_random_state)
        self._convergence_target: Optional[float] = None
    
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
        """Generate random state for training with realistic convergence."""
        # Generate 3-class probabilities
        alpha = self.np_random.uniform(0.5, 2.0, size=3)
        probs = self.np_random.dirichlet(alpha)
        self._prob_down, self._prob_hold, self._prob_up = probs

        # Market prices (sometimes mispriced)
        if self.np_random.random() < 0.7:
            self._market_yes_price = 0.5 + (self._prob_up - self._prob_down) * 0.3
        else:
            self._market_yes_price = self.np_random.uniform(0.3, 0.7)

        self._market_yes_price = np.clip(self._market_yes_price, 0.05, 0.95)
        self._market_no_price = 1.0 - self._market_yes_price
        self._spread = self.np_random.uniform(0.01, 0.04)

        # Initialize price history
        self._yes_price_history = [self._market_yes_price]
        self._no_price_history = [self._market_no_price]

        # Volatility regime
        self._volatility = self.np_random.uniform(0.0, 1.0)

        # Generate outcome - UNBIASED mapping from probabilities
        # FIX: Removed 0.8 multiplier that was causing UP bias in training
        outcome_rand = self.np_random.random()
        if outcome_rand < self._prob_down:
            self._outcome = 0  # Down
        elif outcome_rand < self._prob_down + self._prob_hold:
            self._outcome = 1  # Hold
        else:
            self._outcome = 2  # Up

        # Set convergence target based on outcome (for realistic price simulation)
        # YES token converges to 1.0 if UP, 0.0 if DOWN, stays ~0.5 if HOLD
        if self._outcome == 2:  # UP
            self._convergence_target = 1.0
        elif self._outcome == 0:  # DOWN
            self._convergence_target = 0.0
        else:  # HOLD
            self._convergence_target = 0.5
    
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

        # === REALISTIC INTRA-CANDLE PRICE DYNAMICS ===
        # Three phases: mean-reversion (early), trending (mid), convergence (late)
        # This makes waiting potentially valuable - prices oscillate early
        if self._convergence_target is not None:
            # Phase 1: Early candle (0-30%) - Mean-reverting around 0.5
            # Prices oscillate, creating opportunities for better entry timing
            if time_in_candle < 0.3:
                # Mean reversion toward 0.5 (uncertainty phase)
                mean_reversion = (0.5 - self._market_yes_price) * 0.08
                # Higher noise early - prices are uncertain
                noise_scale = 0.025 + self._volatility * 0.02
                noise = self.np_random.normal(0, noise_scale)
                # Weak pull toward outcome (market has some information)
                weak_trend = (self._convergence_target - self._market_yes_price) * 0.02
                price_change = mean_reversion + noise + weak_trend

            # Phase 2: Mid candle (30-70%) - Trending toward outcome
            elif time_in_candle < 0.7:
                # Moderate convergence toward outcome
                convergence_strength = 0.05 + (time_in_candle - 0.3) * 0.1
                target_pull = (self._convergence_target - self._market_yes_price) * convergence_strength
                # Moderate noise
                noise_scale = 0.015 * (1.0 - time_in_candle * 0.5)
                noise = self.np_random.normal(0, noise_scale)
                price_change = target_pull + noise

            # Phase 3: Late candle (70-100%) - Strong convergence
            else:
                # Strong pull toward settlement price
                convergence_strength = 0.1 + (time_in_candle - 0.7) * 0.4
                target_pull = (self._convergence_target - self._market_yes_price) * convergence_strength
                # Low noise near settlement
                noise_scale = 0.008 * (1.0 - time_in_candle)
                noise = self.np_random.normal(0, noise_scale)
                price_change = target_pull + noise

                # In final 10% of candle, accelerate convergence
                if time_remaining < 0.1:
                    acceleration = (0.1 - time_remaining) / 0.1 * 0.4
                    price_change += (self._convergence_target - self._market_yes_price) * acceleration
        else:
            # Fallback to random walk if no convergence target
            price_change = self.np_random.normal(0, 0.005)

        # Update prices - allow full range [0.01, 0.99] for realistic convergence
        self._market_yes_price = np.clip(self._market_yes_price + price_change, 0.01, 0.99)
        self._market_no_price = 1.0 - self._market_yes_price

        # Track price history for momentum calculation
        self._yes_price_history.append(self._market_yes_price)
        self._no_price_history.append(self._market_no_price)
        # Keep history bounded
        if len(self._yes_price_history) > 20:
            self._yes_price_history = self._yes_price_history[-20:]
            self._no_price_history = self._no_price_history[-20:]
        
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

            # === DYNAMIC EXIT THRESHOLD ===
            # Get current token price for our position
            if self._position.side == "long":
                current_token_price = self._market_yes_price
            else:
                current_token_price = self._market_no_price

            # Base threshold adjusted by:
            # 1. Price proximity to settlement (higher threshold in convergence zone)
            # 2. Time remaining (higher threshold near settlement if winning)
            dynamic_threshold = self.config.exit_signal_base_threshold

            # In convergence zone (price > 0.85 or < 0.15), raise exit threshold significantly
            if current_token_price > self.config.exit_convergence_zone:
                # Price converging to 1.0 - make it hard to exit
                price_factor = (current_token_price - self.config.exit_convergence_zone) / (1.0 - self.config.exit_convergence_zone)
                dynamic_threshold += 0.4 * price_factor  # Up to 0.9 threshold
            elif current_token_price < (1.0 - self.config.exit_convergence_zone):
                # Price converging to 0.0 - this is bad for our position, easier to exit
                dynamic_threshold -= 0.1

            # In final portion of candle, if we're profitable, lock exits
            if time_remaining < self.config.exit_time_lock_fraction and unrealized_pnl > 0:
                # Make it very hard to exit when winning near settlement
                time_lock_factor = (self.config.exit_time_lock_fraction - time_remaining) / self.config.exit_time_lock_fraction
                dynamic_threshold += 0.3 * time_lock_factor

            # Agent's exit signal with dynamic threshold
            if exit_signal > dynamic_threshold:
                should_exit = True
                exit_reason = "agent_signal"

            # Stop loss (always active)
            if unrealized_pnl < -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"

            # Take profit - but suppress in convergence zone near settlement
            in_convergence = current_token_price > self.config.exit_convergence_zone
            near_settlement = time_remaining < 0.2
            if unrealized_pnl > self.config.take_profit_pct and not (in_convergence and near_settlement):
                should_exit = True
                exit_reason = "take_profit"

            # Trailing stop (if max PnL > 50% of TP, exit if drops by 30% of max)
            # But disable in convergence zone
            if not in_convergence and self._position.max_pnl > self.config.take_profit_pct * 0.5:
                drawdown_from_max = self._position.max_pnl - unrealized_pnl
                if drawdown_from_max > self._position.max_pnl * 0.3:
                    should_exit = True
                    exit_reason = "trailing_stop"

            if should_exit:
                reward += self._execute_exit(exit_reason)
                self._trades_this_candle += 1
            else:
                # === HOLD REWARD WITH TIME URGENCY AND RECOVERY ===
                # Time urgency: reward holding more as settlement approaches
                # Exponential increase: 1.0 at start, up to time_urgency_multiplier at end
                time_urgency = 1.0 + (self.config.time_urgency_multiplier - 1.0) * (time_in_candle ** 2)

                if unrealized_pnl > 0:
                    # Base reward scaled by profit magnitude (capped at take_profit level)
                    profit_ratio = min(unrealized_pnl / self.config.take_profit_pct, 1.0)
                    scaled_reward = self.config.hold_reward * profit_ratio

                    # Apply time urgency multiplier
                    scaled_reward *= time_urgency

                    # Bonus for improving position (trending towards profitability)
                    pnl_improvement = unrealized_pnl - self._position.prev_pnl
                    if pnl_improvement > 0:
                        improvement_bonus = self.config.hold_reward * 0.5 * min(pnl_improvement * 10, 1.0)
                        scaled_reward += improvement_bonus * time_urgency

                    # Extra bonus for holding in convergence zone
                    if in_convergence:
                        convergence_bonus = self.config.hold_reward * 0.5 * time_urgency
                        scaled_reward += convergence_bonus

                    reward += scaled_reward

                elif unrealized_pnl < 0:
                    # === RECOVERY INCENTIVE FOR LOSING POSITIONS ===
                    # Reward for holding if price is moving toward recovery
                    pnl_improvement = unrealized_pnl - self._position.prev_pnl

                    if pnl_improvement > 0:
                        # Position is recovering - reward proportional to recovery rate
                        recovery_rate = min(pnl_improvement * 20, 1.0)  # Scale improvement
                        recovery_reward = self.config.hold_reward * 0.75 * recovery_rate * time_urgency
                        reward += recovery_reward

                    # Additional recovery signal: check if price is moving toward our side
                    if self._position.side == "long":
                        price_momentum = current_token_price - self._position.entry_token_price
                    else:
                        price_momentum = self._position.entry_token_price - current_token_price

                    # If momentum is positive (price moving our way), small reward for patience
                    if price_momentum > 0 and len(self._yes_price_history) >= 3:
                        recent_momentum = self._yes_price_history[-1] - self._yes_price_history[-3]
                        if self._position.side == "short":
                            recent_momentum = -recent_momentum
                        if recent_momentum > 0:
                            # Price trending toward recovery
                            momentum_bonus = self.config.hold_reward * 0.3 * min(recent_momentum * 50, 1.0)
                            reward += momentum_bonus

                else:
                    # Breakeven position - small reward for improving
                    improvement = unrealized_pnl - self._position.prev_pnl
                    if improvement > 0:
                        reward += self.config.hold_reward * 0.25 * min(improvement * 10, 1.0) * time_urgency

                # Update previous PnL for next step
                self._position.prev_pnl = unrealized_pnl

        # === Check for Entry or Hold ===
        if self._position is None:
            if hold_prob > 0.5:
                # Agent chooses to wait - reward based on entry improvement potential
                predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])

                # Base reward for waiting
                wait_reward = 0.02  # Increased from 0.01

                # Bonus if Hold is predicted (model agrees with waiting)
                if predicted_class == 1:
                    wait_reward += 0.02

                # Bonus for waiting during high volatility (more opportunity)
                if len(self._yes_price_history) >= 5:
                    recent_volatility = np.std(self._yes_price_history[-5:])
                    # Higher volatility = more potential for better entry
                    volatility_bonus = min(recent_volatility * 5, 0.05)
                    wait_reward += volatility_bonus

                # Bonus for waiting early in candle (more time for price discovery)
                if time_in_candle < 0.3:
                    early_bonus = 0.03 * (0.3 - time_in_candle) / 0.3
                    wait_reward += early_bonus

                # Bonus for waiting when price is unfavorable for our predicted direction
                if predicted_class == 2 and self._market_yes_price > 0.55:
                    # Want to go long but price is high - good to wait
                    wait_reward += 0.02
                elif predicted_class == 0 and self._market_no_price > 0.55:
                    # Want to go short but NO price is high - good to wait
                    wait_reward += 0.02

                reward += wait_reward

            elif self._trades_this_candle < self.config.max_trades_per_candle:
                # Agent wants to enter
                if abs(direction) > 0.1 and size > 0.05:
                    reward += self._execute_entry(direction, size)
                    self._trades_this_candle += 1

        # === CHURN PENALTY ===
        # Penalize excessive trading (entering/exiting too frequently)
        if self._trades_this_candle > self.config.churn_threshold:
            excess_trades = self._trades_this_candle - self.config.churn_threshold
            churn_penalty = -self.config.churn_penalty * excess_trades
            reward += churn_penalty

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
            entry_token_price=self._market_yes_price if side == "long" else self._market_no_price,
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
        
        # Apply DSR - SYMMETRIC quality weighting (no loss amplification)
        if self.config.use_dsr_reward:
            predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
            confidence = max(self._prob_down, self._prob_hold, self._prob_up)

            if (self._position.side == "long" and predicted_class == 2) or \
               (self._position.side == "short" and predicted_class == 0):
                quality = confidence
            else:
                quality = 0.3

            # SYMMETRIC multiplier: same scaling for wins and losses
            # High quality = higher reward for wins, lower penalty for losses
            # This encourages learning from mistakes rather than avoiding all risk
            multiplier = 0.5 + quality * 0.5  # Range: 0.5 to 1.0

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
        """Settle position at candle end with magnitude-scaled reward."""
        if self._position is None:
            return 0.0

        # Settlement price based on outcome
        if self._outcome == 2:
            settlement_price = 1.0
        elif self._outcome == 0:
            settlement_price = 0.0
        else:
            settlement_price = 0.5

        # Calculate PnL (in price units, e.g., 0.0 to 1.0)
        if self._position.side == "long":
            pnl = (settlement_price - self._position.entry_price) * self._position.size
        else:
            pnl = (1.0 - settlement_price - self._position.entry_price) * self._position.size

        # Calculate magnitude of the settlement move (how much we captured)
        if self._position.side == "long":
            price_move = settlement_price - self._position.entry_price
        else:
            price_move = (1.0 - settlement_price) - self._position.entry_price

        # Magnitude factor: larger moves get proportionally larger rewards
        # A move from 0.5 to 1.0 (0.5 profit) should reward more than 0.5 to 0.55
        magnitude_factor = min(abs(price_move) * self.config.settlement_magnitude_scale, 2.0)

        # Apply DSR if enabled - SYMMETRIC quality weighting (no loss amplification)
        if self.config.use_dsr_reward:
            predicted_class = np.argmax([self._prob_down, self._prob_hold, self._prob_up])
            confidence = max(self._prob_down, self._prob_hold, self._prob_up)

            if (self._position.side == "long" and predicted_class == 2) or \
               (self._position.side == "short" and predicted_class == 0):
                quality = confidence
            else:
                quality = 0.3

            # SYMMETRIC multiplier: same scaling for wins and losses
            multiplier = 0.5 + quality * 0.5  # Range: 0.5 to 1.0

            magnitude_factor *= multiplier

        self._trade_outcomes.append(pnl > 0)
        self._position = None

        # Settlement reward SCALED BY MAGNITUDE
        # This makes holding for big settlement wins more attractive than small early exits
        if pnl > 0:
            # Base reward * magnitude: a 50-cent win rewards 2x a 25-cent win
            reward = self.config.win_reward * (0.5 + magnitude_factor)
        else:
            reward = self.config.loss_penalty * (0.5 + magnitude_factor)

        return np.clip(reward, -2.5, 2.5)  # Allow higher rewards for big settlements
    
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
        """Build 26-dimensional observation with momentum and convergence features."""
        time_in_candle = self._candle_step / self.config.steps_per_candle
        time_remaining = 1.0 - time_in_candle
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

        # === NEW FEATURES ===

        # 1. Token momentum (rate of change over momentum_window steps)
        window = self.config.momentum_window
        if len(self._yes_price_history) >= window:
            yes_momentum = (self._yes_price_history[-1] - self._yes_price_history[-window]) / window
            no_momentum = (self._no_price_history[-1] - self._no_price_history[-window]) / window
        else:
            yes_momentum = 0.0
            no_momentum = 0.0

        # 2. Convergence velocity: how fast price is approaching 0 or 1
        # Positive = converging to 1, Negative = converging to 0
        # Higher magnitude = faster convergence
        if self._market_yes_price > 0.5:
            # Price above 0.5, check convergence toward 1.0
            convergence_velocity = yes_momentum * 10  # Scale for visibility
        else:
            # Price below 0.5, check convergence toward 0.0
            convergence_velocity = -yes_momentum * 10

        # 3. Price distance to settlement (how close to 0 or 1)
        # 0.5 = maximum uncertainty, 0.0 = at settlement price
        price_distance_to_settlement = 0.5 - abs(self._market_yes_price - 0.5)

        # 4. YES/NO sum deviation (arbitrage signal)
        # Should be ~0 in efficient market, non-zero indicates opportunity
        yes_no_sum_deviation = (self._market_yes_price + self._market_no_price) - 1.0

        # 5. Time urgency: exponential increase as settlement approaches
        # 1.0 at start, increases toward settlement
        time_urgency = 1.0 + (self.config.time_urgency_multiplier - 1.0) * (time_in_candle ** 2)

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
            # === NEW: Momentum (2) ===
            yes_momentum * 100,  # Scale for better learning
            no_momentum * 100,
            # === NEW: Convergence (2) ===
            convergence_velocity,
            price_distance_to_settlement,
            # === NEW: Arbitrage (1) ===
            yes_no_sum_deviation * 10,  # Scale deviation
            # === NEW: Time urgency (1) ===
            time_urgency / self.config.time_urgency_multiplier,  # Normalize to [0.5, 1]
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
