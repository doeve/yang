"""
Enhanced Multi-Asset Candle Prediction Environment.

Improvements over base MultiAssetEnv:
1. Order Flow Features (volume delta, trade imbalance, large trade detection)
2. Multi-Timeframe Features (1h, 4h price trends)
3. Skip Action (model can choose not to bet on uncertain candles)
4. Timing Reward (encourages waiting for confirmation before betting)
5. Risk-Adjusted Rewards (intelligent position sizing rewarded)
6. Increased history window (500 instead of 300)
"""

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import structlog
from gymnasium import spaces

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedMultiAssetConfig:
    """Configuration for enhanced multi-asset environment."""
    
    candle_minutes: int = 15
    price_history_length: int = 500  # Increased from 300
    max_position_size: float = 0.5
    random_start: bool = True
    
    # Timing rewards
    min_bet_time_pct: float = 0.3  # Earliest time to bet (30% into candle)
    timing_reward_scale: float = 0.5  # Bonus for waiting
    
    # Skip action
    enable_skip: bool = True
    skip_threshold: float = 0.5  # Action > threshold = skip
    
    # Order flow
    include_order_flow: bool = True
    large_trade_threshold_pct: float = 0.1  # Top 10% = large trade
    
    # Multi-timeframe
    include_multi_timeframe: bool = True
    
    # Risk-adjusted rewards
    use_risk_adjusted_rewards: bool = True
    sharpe_window: int = 20  # Rolling window for Sharpe calculation
    
    # Assets
    include_dxy: bool = True
    include_eurusd: bool = True


class EnhancedMultiAssetEnv(gym.Env):
    """
    Enhanced multi-asset candle prediction environment.
    
    Observation Space (variable dims based on config):
    - BTC price history (500) - normalized to candle open
    - BTC volume history (500) - normalized
    - Order flow features (6) - volume delta, imbalance, large trades, etc.
    - Multi-timeframe trends (4) - 1h/4h price and volume momentum
    - DXY returns (2) - 1h, 4h
    - EUR/USD returns (2) - 1h, 4h
    - Context features (7) - timing, volatility, etc.
    
    Action Space:
    - direction: -1 (short) to +1 (long)
    - size: 0 to 1 (position size)
    - skip: 0 to 1 (probability of skipping this candle)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        btc_data: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
        config: Optional[EnhancedMultiAssetConfig] = None,
    ):
        super().__init__()
        
        self.config = config or EnhancedMultiAssetConfig()
        self.has_dxy = dxy_data is not None and self.config.include_dxy
        self.has_eurusd = eurusd_data is not None and self.config.include_eurusd
        
        self._setup_data(btc_data, dxy_data, eurusd_data)
        self._setup_spaces()
        
        # Episode state
        self._current_candle_idx = 0
        self._current_step = 0
        self._position_taken = False
        self._position_direction = 0.0
        self._position_size = 0.0
        self._candle_open = 0.0
        self._balance = 1000.0
        self._reward_history: list[float] = []  # For Sharpe calculation
        
        logger.info(
            "EnhancedMultiAssetEnv initialized",
            total_candles=len(self._candles),
            obs_dim=self._obs_dim,
            has_order_flow=self.config.include_order_flow,
            has_multi_tf=self.config.include_multi_timeframe,
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.config.random_start:
            min_idx = self.config.price_history_length + 100  # Extra buffer for multi-TF
            max_idx = len(self._candles) - 1
            if max_idx > min_idx:
                self._current_candle_idx = self.np_random.integers(min_idx, max_idx)
            else:
                self._current_candle_idx = min_idx
        else:
            self._current_candle_idx = self.config.price_history_length + 100
        
        candle = self._candles[self._current_candle_idx]
        self._current_step = candle["start_idx"]
        self._candle_end_idx = candle["end_idx"]
        self._candle_open = candle["open"]
        self._candle_close = candle["close"]
        self._candle_return = candle["return"]
        
        self._position_taken = False
        self._position_direction = 0.0
        self._position_size = 0.0
        self._entry_time_pct = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))
        skip_prob = float(action[2]) if len(action) > 2 else 0.0
        
        reward = 0.0
        terminated = False
        
        # Calculate timing
        candle_length = self._candle_end_idx - self._candles[self._current_candle_idx]["start_idx"]
        current_position_in_candle = self._current_step - self._candles[self._current_candle_idx]["start_idx"]
        time_pct = current_position_in_candle / max(candle_length, 1)
        
        idx = min(self._current_step, len(self._btc_prices) - 1)
        current_price = self._btc_prices[idx]
        
        # Check for skip action
        if self.config.enable_skip and skip_prob > self.config.skip_threshold:
            # Model chose to skip this candle
            terminated = True
            # Small reward for skipping uncertain situations (if move was small)
            actual_move = abs(self._candle_return) * 100
            if actual_move < 0.1:  # The candle moved less than 0.1%
                reward = 0.05  # Good decision to skip
            else:
                reward = 0.0  # Neutral - missed opportunity but not penalized
            
        elif not self._position_taken and abs(direction) > 0.1 and size > 0.05:
            # Check if betting too early
            if time_pct < self.config.min_bet_time_pct:
                # Allow bet but no timing bonus
                timing_bonus = 0.0
            else:
                # Timing bonus for waiting
                timing_bonus = (time_pct - self.config.min_bet_time_pct) * self.config.timing_reward_scale
            
            self._position_taken = True
            self._position_direction = 1.0 if direction > 0 else -1.0
            self._position_size = size * self.config.max_position_size
            self._entry_time_pct = time_pct
            
            terminated = True
            
            # Calculate move from entry to close
            move_pct = (self._candle_close - current_price) / current_price * 100
            
            # === RISK-ADJUSTED REWARD ===
            if self.config.use_risk_adjusted_rewards:
                reward = self._compute_risk_adjusted_reward(
                    move_pct, self._position_direction, self._position_size, timing_bonus
                )
            else:
                # Original asymmetric reward
                if self._position_direction > 0:  # BET UP
                    if move_pct > 0:
                        reward = move_pct * self._position_size + timing_bonus
                    else:
                        reward = -1.0 * (self._position_size ** 2) * abs(move_pct) * 2.0
                else:  # BET DOWN
                    if move_pct < 0:
                        reward = abs(move_pct) * self._position_size + timing_bonus
                    else:
                        reward = -1.0 * (self._position_size ** 2) * abs(move_pct) * 2.0
        else:
            # Continue waiting
            self._current_step += 1
            if self._current_step >= self._candle_end_idx:
                terminated = True
                # Penalty for indecision when there was a clear move
                actual_move = abs(self._candle_return) * 100
                if actual_move > 0.2:  # Missed a significant move
                    reward = -0.1 * actual_move  # Penalty proportional to missed move
                else:
                    reward = 0.01  # Small reward for correctly identifying uncertain candle
        
        self._reward_history.append(reward)
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _compute_risk_adjusted_reward(
        self, 
        move_pct: float, 
        direction: float, 
        position_size: float,
        timing_bonus: float,
    ) -> float:
        """
        Compute risk-adjusted reward using Kelly-inspired logic.
        
        Rewards:
        1. Higher reward for correct direction
        2. Bonus for appropriate sizing (not over-betting on small moves)
        3. Bonus for timing (waiting for confirmation)
        4. Penalty scaled by position size squared for wrong bets
        """
        is_correct = (direction > 0 and move_pct > 0) or (direction < 0 and move_pct < 0)
        abs_move = abs(move_pct)
        
        if is_correct:
            # === WIN ===
            base_reward = abs_move * position_size
            
            # Kelly-optimal sizing bonus
            # Ideal: size proportional to edge
            # If abs_move is small but size is small = good
            # If abs_move is large and size is large = good
            # If abs_move is small but size is large = suboptimal
            edge_estimate = abs_move / 100  # Rough edge
            optimal_size = min(edge_estimate * 5, 0.5)  # Cap at 50%
            sizing_accuracy = 1.0 - abs(position_size - optimal_size)
            sizing_bonus = sizing_accuracy * 0.2
            
            reward = base_reward + timing_bonus + sizing_bonus
            
        else:
            # === LOSS ===
            # Quadratic penalty scaled by position size
            # Larger positions on wrong bets = much larger penalty
            base_penalty = abs_move * (position_size ** 2) * 2.0
            
            # Extra penalty for betting large on small uncertain moves
            if abs_move < 0.1 and position_size > 0.3:
                overbet_penalty = position_size * 0.5
                base_penalty += overbet_penalty
            
            reward = -base_penalty
        
        return reward
    
    def _setup_data(
        self, 
        btc_data: pd.DataFrame, 
        dxy_data: Optional[pd.DataFrame],
        eurusd_data: Optional[pd.DataFrame],
    ) -> None:
        """Process and prepare all data for the environment."""
        # Process BTC
        btc = btc_data.copy().sort_values("timestamp").reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(btc["timestamp"]):
            btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
        
        self._btc_timestamps = btc["timestamp"].values
        self._btc_prices = btc["price"].values.astype(np.float64)
        self._btc_volumes = btc["volume"].values.astype(np.float64) if "volume" in btc else np.ones(len(btc))
        
        # === ORDER FLOW FEATURES ===
        if self.config.include_order_flow:
            self._compute_order_flow_features(btc)
        
        # === MULTI-TIMEFRAME FEATURES ===
        if self.config.include_multi_timeframe:
            self._compute_multi_timeframe_features(btc)
        
        # === FOREX DATA ===
        def prepare_forex(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            
            prices = df["price"]
            ret_1h = prices.pct_change(1).fillna(0)
            ret_4h = prices.pct_change(4).fillna(0)
            
            s_1h = pd.Series(ret_1h.values, index=df["timestamp"]).sort_index()
            s_4h = pd.Series(ret_4h.values, index=df["timestamp"]).sort_index()
            
            btc_index = pd.DatetimeIndex(btc["timestamp"])
            a_1h = s_1h.reindex(btc_index, method="ffill").fillna(0).values.astype(np.float32)
            a_4h = s_4h.reindex(btc_index, method="ffill").fillna(0).values.astype(np.float32)
            
            return a_1h, a_4h
        
        if self.has_dxy and dxy_data is not None:
            self._dxy_ret_1h, self._dxy_ret_4h = prepare_forex(dxy_data)
        else:
            self._dxy_ret_1h = None
            self._dxy_ret_4h = None
        
        if self.has_eurusd and eurusd_data is not None:
            self._eurusd_ret_1h, self._eurusd_ret_4h = prepare_forex(eurusd_data)
        else:
            self._eurusd_ret_1h = None
            self._eurusd_ret_4h = None
        
        # Create candles
        self._create_candles(btc)
        self._compute_atr()
    
    def _compute_order_flow_features(self, btc: pd.DataFrame) -> None:
        """
        Compute order flow features:
        1. Volume Delta (cumulative buy - sell volume proxy)
        2. Trade Imbalance (buy volume / total volume)
        3. Large Trade Count (trades in top percentile)
        4. Volume Momentum (current vs rolling average)
        5. Price-Volume Correlation
        6. VWAP Distance
        """
        volumes = self._btc_volumes
        prices = self._btc_prices
        n = len(prices)
        
        # Volume Delta (proxy: use price direction to estimate buy/sell)
        price_changes = np.diff(prices, prepend=prices[0])
        buy_volume = np.where(price_changes > 0, volumes, 0)
        sell_volume = np.where(price_changes < 0, volumes, 0)
        
        # Cumulative volume delta over rolling window
        window = 100
        self._volume_delta = np.zeros(n, dtype=np.float32)
        for i in range(window, n):
            self._volume_delta[i] = (buy_volume[i-window:i].sum() - sell_volume[i-window:i].sum())
        
        # Normalize
        vd_std = np.std(self._volume_delta[self._volume_delta != 0]) + 1e-8
        self._volume_delta /= vd_std
        
        # Trade Imbalance (buy / total)
        self._trade_imbalance = np.zeros(n, dtype=np.float32)
        for i in range(window, n):
            total = buy_volume[i-window:i].sum() + sell_volume[i-window:i].sum()
            if total > 0:
                self._trade_imbalance[i] = buy_volume[i-window:i].sum() / total
            else:
                self._trade_imbalance[i] = 0.5
        
        # Large Trade Detection (rolling percentile)
        self._large_trade_ratio = np.zeros(n, dtype=np.float32)
        threshold_pct = self.config.large_trade_threshold_pct
        for i in range(window, n):
            vol_window = volumes[i-window:i]
            threshold = np.percentile(vol_window, (1 - threshold_pct) * 100)
            self._large_trade_ratio[i] = (vol_window > threshold).sum() / window
        
        # Volume Momentum (current vs 20-period MA)
        vol_ma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
        self._volume_momentum = (volumes / (vol_ma + 1e-8) - 1.0).astype(np.float32)
        np.clip(self._volume_momentum, -5, 5, out=self._volume_momentum)
        
        # Price-Volume Correlation (rolling)
        self._pv_correlation = np.zeros(n, dtype=np.float32)
        for i in range(window, n):
            p = prices[i-window:i]
            v = volumes[i-window:i]
            if np.std(p) > 0 and np.std(v) > 0:
                self._pv_correlation[i] = np.corrcoef(p, v)[0, 1]
        
        # VWAP Distance
        cum_vol = np.cumsum(volumes)
        cum_pv = np.cumsum(prices * volumes)
        vwap = cum_pv / (cum_vol + 1e-8)
        self._vwap_distance = ((prices - vwap) / (vwap + 1e-8) * 100).astype(np.float32)
        np.clip(self._vwap_distance, -5, 5, out=self._vwap_distance)
        
        logger.info("Order flow features computed")
    
    def _compute_multi_timeframe_features(self, btc: pd.DataFrame) -> None:
        """
        Compute multi-timeframe features:
        1. 1-hour price momentum (% change)
        2. 4-hour price momentum
        3. 1-hour volume momentum
        4. 4-hour volume momentum
        """
        prices = self._btc_prices
        volumes = self._btc_volumes
        n = len(prices)
        
        # Approximate indices for 1h and 4h lookback
        # Assuming ~100ms data: 1h = 36000 ticks, 4h = 144000 ticks
        # Assuming ~1s data: 1h = 3600 ticks, 4h = 14400 ticks
        # We'll estimate based on timestamp delta
        if len(btc) > 1:
            avg_delta = (btc["timestamp"].iloc[-1] - btc["timestamp"].iloc[0]).total_seconds() / len(btc)
            ticks_per_hour = int(3600 / max(avg_delta, 0.001))
        else:
            ticks_per_hour = 3600  # Default assumption
        
        lookback_1h = min(ticks_per_hour, n - 1)
        lookback_4h = min(ticks_per_hour * 4, n - 1)
        
        # 1H and 4H Price Momentum
        self._momentum_1h = np.zeros(n, dtype=np.float32)
        self._momentum_4h = np.zeros(n, dtype=np.float32)
        
        for i in range(lookback_4h, n):
            self._momentum_1h[i] = (prices[i] / prices[i - lookback_1h] - 1) * 100
            self._momentum_4h[i] = (prices[i] / prices[i - lookback_4h] - 1) * 100
        
        np.clip(self._momentum_1h, -10, 10, out=self._momentum_1h)
        np.clip(self._momentum_4h, -20, 20, out=self._momentum_4h)
        
        # 1H and 4H Volume Momentum
        self._vol_momentum_1h = np.zeros(n, dtype=np.float32)
        self._vol_momentum_4h = np.zeros(n, dtype=np.float32)
        
        for i in range(lookback_4h, n):
            vol_now = volumes[max(0, i-100):i].mean()
            vol_1h_ago = volumes[max(0, i-lookback_1h-100):max(1, i-lookback_1h)].mean()
            vol_4h_ago = volumes[max(0, i-lookback_4h-100):max(1, i-lookback_4h)].mean()
            
            self._vol_momentum_1h[i] = (vol_now / (vol_1h_ago + 1e-8) - 1)
            self._vol_momentum_4h[i] = (vol_now / (vol_4h_ago + 1e-8) - 1)
        
        np.clip(self._vol_momentum_1h, -5, 5, out=self._vol_momentum_1h)
        np.clip(self._vol_momentum_4h, -5, 5, out=self._vol_momentum_4h)
        
        logger.info("Multi-timeframe features computed", lookback_1h=lookback_1h, lookback_4h=lookback_4h)
    
    def _create_candles(self, btc: pd.DataFrame) -> None:
        """Create candle boundaries from BTC data."""
        candle_seconds = self.config.candle_minutes * 60
        btc["candle_idx"] = (btc["timestamp"].astype(np.int64) // 10**9 // candle_seconds).astype(int)
        
        candles = []
        for candle_id, group in btc.groupby("candle_idx"):
            if len(group) < 10:
                continue
            
            candles.append({
                "start_idx": group.index[0],
                "end_idx": group.index[-1],
                "open": group["price"].iloc[0],
                "close": group["price"].iloc[-1],
                "high": group["price"].max(),
                "low": group["price"].min(),
                "return": (group["price"].iloc[-1] - group["price"].iloc[0]) / group["price"].iloc[0],
                "volume": group["volume"].sum() if "volume" in group else 0,
            })
        
        self._candles = candles
    
    def _compute_atr(self, period: int = 14) -> None:
        """Compute Average True Range."""
        changes = np.abs(np.diff(self._btc_prices))
        self._atr = np.zeros(len(self._btc_prices))
        if len(changes) > period:
            for i in range(period, len(self._btc_prices)):
                self._atr[i] = np.mean(changes[max(0, i-period):i])
        self._atr[:period] = self._atr[period] if period < len(self._atr) else 1.0
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        seq_len = self.config.price_history_length
        
        # BTC: prices + volumes (500 + 500)
        obs_dim = seq_len * 2
        
        # Order flow features (6)
        if self.config.include_order_flow:
            obs_dim += 6
        
        # Multi-timeframe features (4)
        if self.config.include_multi_timeframe:
            obs_dim += 4
        
        # Forex (DXY: 2, EUR/USD: 2)
        if self.has_dxy:
            obs_dim += 2
        if self.has_eurusd:
            obs_dim += 2
        
        # Context features (7)
        # current_vs_open, time_remaining, time_in_candle, hour_sin, hour_cos, atr, volatility_regime
        obs_dim += 7
        
        self._obs_dim = obs_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        
        # 3D action space: [direction, size, skip]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector with all features."""
        idx = min(self._current_step, len(self._btc_prices) - 1)
        seq_len = self.config.price_history_length
        
        obs_parts = []
        
        # === BTC Prices (normalized to candle open) ===
        if idx < seq_len:
            btc_prices = np.full(seq_len, self._candle_open)
            if idx > 0:
                btc_prices[seq_len - idx:] = self._btc_prices[:idx]
        else:
            btc_prices = self._btc_prices[idx - seq_len:idx]
        
        btc_norm = (btc_prices / self._candle_open - 1.0) * 100
        obs_parts.append(btc_norm.astype(np.float32))
        
        # === BTC Volumes (normalized) ===
        if idx < seq_len:
            volumes = np.zeros(seq_len)
            if idx > 0:
                volumes[seq_len - idx:] = self._btc_volumes[:idx]
        else:
            volumes = self._btc_volumes[idx - seq_len:idx]
        
        vol_max = volumes.max() if volumes.max() > 0 else 1.0
        obs_parts.append((volumes / vol_max).astype(np.float32))
        
        # === Order Flow Features (6) ===
        if self.config.include_order_flow:
            order_flow = np.array([
                self._volume_delta[idx],
                self._trade_imbalance[idx],
                self._large_trade_ratio[idx],
                self._volume_momentum[idx],
                self._pv_correlation[idx],
                self._vwap_distance[idx],
            ], dtype=np.float32)
            obs_parts.append(order_flow)
        
        # === Multi-Timeframe Features (4) ===
        if self.config.include_multi_timeframe:
            mtf = np.array([
                self._momentum_1h[idx],
                self._momentum_4h[idx],
                self._vol_momentum_1h[idx],
                self._vol_momentum_4h[idx],
            ], dtype=np.float32)
            obs_parts.append(mtf)
        
        # === Forex Features ===
        if self._dxy_ret_1h is not None:
            obs_parts.append(np.array([
                self._dxy_ret_1h[idx] * 100,
                self._dxy_ret_4h[idx] * 100,
            ], dtype=np.float32))
        
        if self._eurusd_ret_1h is not None:
            obs_parts.append(np.array([
                self._eurusd_ret_1h[idx] * 100,
                self._eurusd_ret_4h[idx] * 100,
            ], dtype=np.float32))
        
        # === Context Features (7) ===
        current_price = self._btc_prices[idx]
        current_vs_open = (current_price / self._candle_open - 1.0) * 100
        
        candle = self._candles[self._current_candle_idx]
        candle_length = candle["end_idx"] - candle["start_idx"]
        time_in_candle = (idx - candle["start_idx"]) / max(candle_length, 1)
        time_remaining = 1.0 - time_in_candle
        
        timestamp = pd.Timestamp(self._btc_timestamps[idx])
        hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
        
        atr = self._atr[idx] / (self._candle_open + 1e-8) * 100
        
        # Volatility regime (normalized ATR percentile)
        atr_window = self._atr[max(0, idx-1000):idx]
        if len(atr_window) > 0:
            vol_regime = (self._atr[idx] - atr_window.mean()) / (atr_window.std() + 1e-8)
        else:
            vol_regime = 0.0
        
        context = np.array([
            current_vs_open,
            time_remaining,
            time_in_candle,
            hour_sin,
            hour_cos,
            atr,
            np.clip(vol_regime, -3, 3),
        ], dtype=np.float32)
        obs_parts.append(context)
        
        # Concatenate and clip
        obs = np.concatenate(obs_parts)
        return np.clip(obs, -100.0, 100.0).astype(np.float32)
    
    def _get_info(self) -> dict[str, Any]:
        """Build info dictionary."""
        return {
            "candle_idx": self._current_candle_idx,
            "balance": self._balance,
            "position_taken": self._position_taken,
            "position_direction": "UP" if self._position_direction > 0 else "DOWN" if self._position_direction < 0 else None,
            "position_size": self._position_size,
            "candle_return": self._candle_return,
            "entry_time_pct": self._entry_time_pct if self._position_taken else None,
        }


def make_enhanced_multi_asset_vec_env(
    btc_data: pd.DataFrame,
    dxy_data: Optional[pd.DataFrame] = None,
    eurusd_data: Optional[pd.DataFrame] = None,
    num_envs: int = 4,
    config: Optional[EnhancedMultiAssetConfig] = None,
) -> Any:
    """Create vectorized enhanced multi-asset environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        cfg = config or EnhancedMultiAssetConfig()
        return EnhancedMultiAssetEnv(btc_data, dxy_data, eurusd_data, config=cfg)
    
    return DummyVecEnv([make_env for _ in range(num_envs)])
