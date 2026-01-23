"""
Token-centric feature engineering for Polymarket trading.

This module extracts features primarily from token prices (YES/NO),
using BTC data only as secondary guidance signals.

Key insight: We're trading tokens, not predicting BTC. Token prices
already embed market information and have guaranteed convergence to 0/1.

Feature Categories:
1. Price State: Current YES/NO prices, spread, mid-price
2. Momentum: Price velocity at multiple timescales
3. Convergence: Rate and direction of approach to 0 or 1
4. Time Value: Time to settlement, theta proxy
5. Liquidity: Volume, spread dynamics
6. Arbitrage: YES + NO deviation from 1.0
7. BTC Guidance: Current return, volatility (secondary)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TokenFeatureConfig:
    """Configuration for token feature extraction."""

    # Momentum windows (in steps, typically 1 step = 1 second)
    momentum_windows: Tuple[int, ...] = (5, 15, 60, 300)  # 5s, 15s, 1m, 5m

    # Volatility windows
    volatility_windows: Tuple[int, ...] = (30, 60, 300)  # 30s, 1m, 5m

    # Convergence calculation window
    convergence_window: int = 30  # 30 seconds for velocity calculation

    # Candle duration in seconds
    candle_seconds: int = 15 * 60  # 15 minutes

    # Price bounds
    min_price: float = 0.01
    max_price: float = 0.99

    # Feature normalization
    clip_value: float = 5.0  # Clip normalized features


class TokenFeatureBuilder:
    """
    Build features from Polymarket token prices.

    Features are designed for:
    1. Edge detection (is token mispriced?)
    2. Timing optimization (when to enter/exit?)
    3. Convergence prediction (which way will price move?)
    """

    def __init__(self, config: Optional[TokenFeatureConfig] = None):
        self.config = config or TokenFeatureConfig()

        # Feature dimension calculation
        self._calculate_feature_dim()

        logger.info(
            "TokenFeatureBuilder initialized",
            feature_dim=self.feature_dim,
            momentum_windows=self.config.momentum_windows,
        )

    def _calculate_feature_dim(self):
        """Calculate total feature dimension."""
        dim = 0

        # Price state features (7)
        dim += 7  # yes_price, no_price, spread, mid_price, yes_distance_0, yes_distance_1, price_sum_deviation

        # Momentum features (4 windows × 2 tokens = 8)
        dim += len(self.config.momentum_windows) * 2

        # Acceleration features (4 windows × 2 tokens = 8)
        dim += len(self.config.momentum_windows) * 2

        # Volatility features (3 windows × 2 = 6)
        dim += len(self.config.volatility_windows) * 2

        # Convergence features (6)
        dim += 6  # velocity, acceleration, direction, distance_to_target, convergence_strength, reversal_risk

        # Time features (4)
        dim += 4  # time_remaining, time_elapsed, theta_proxy, urgency

        # Relative value features (4)
        dim += 4  # edge_proxy, mispricing_signal, mean_reversion_signal, trend_strength

        # BTC guidance features (6)
        dim += 6  # btc_return, btc_momentum, btc_volatility, btc_direction, correlation_proxy, divergence

        self.feature_dim = dim

        # Feature names for debugging
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        """Build list of feature names."""
        names = []

        # Price state
        names.extend([
            'yes_price', 'no_price', 'spread', 'mid_price',
            'yes_distance_0', 'yes_distance_1', 'price_sum_deviation'
        ])

        # Momentum
        for w in self.config.momentum_windows:
            names.extend([f'yes_momentum_{w}s', f'no_momentum_{w}s'])

        # Acceleration
        for w in self.config.momentum_windows:
            names.extend([f'yes_accel_{w}s', f'no_accel_{w}s'])

        # Volatility
        for w in self.config.volatility_windows:
            names.extend([f'yes_volatility_{w}s', f'no_volatility_{w}s'])

        # Convergence
        names.extend([
            'convergence_velocity', 'convergence_acceleration', 'convergence_direction',
            'distance_to_target', 'convergence_strength', 'reversal_risk'
        ])

        # Time
        names.extend(['time_remaining', 'time_elapsed', 'theta_proxy', 'urgency'])

        # Relative value
        names.extend(['edge_proxy', 'mispricing_signal', 'mean_reversion_signal', 'trend_strength'])

        # BTC guidance
        names.extend([
            'btc_return', 'btc_momentum', 'btc_volatility',
            'btc_direction', 'correlation_proxy', 'divergence'
        ])

        return names

    def compute_features(
        self,
        yes_prices: np.ndarray,
        no_prices: np.ndarray,
        time_remaining: float,
        btc_prices: Optional[np.ndarray] = None,
        btc_open: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute all features from price history.

        Args:
            yes_prices: Array of YES token prices (most recent last)
            no_prices: Array of NO token prices (most recent last)
            time_remaining: Fraction of candle remaining (1.0 = start, 0.0 = settlement)
            btc_prices: Optional array of BTC prices for guidance
            btc_open: Optional BTC candle open price

        Returns:
            Feature vector of shape (feature_dim,)
        """
        features = []

        # Ensure we have enough history
        min_history = max(self.config.momentum_windows) + 10
        if len(yes_prices) < min_history:
            # Pad with first value
            pad_len = min_history - len(yes_prices)
            yes_prices = np.concatenate([np.full(pad_len, yes_prices[0]), yes_prices])
            no_prices = np.concatenate([np.full(pad_len, no_prices[0]), no_prices])

        # Current prices
        yes_price = yes_prices[-1]
        no_price = no_prices[-1]

        # === PRICE STATE FEATURES ===
        spread = abs(yes_price + no_price - 1.0)  # Should be ~0 if efficient
        mid_price = (yes_price + no_price) / 2.0
        yes_distance_0 = yes_price  # Distance from 0 (NO wins)
        yes_distance_1 = 1.0 - yes_price  # Distance from 1 (YES wins)
        price_sum_deviation = yes_price + no_price - 1.0  # Arbitrage signal

        features.extend([
            yes_price, no_price, spread, mid_price,
            yes_distance_0, yes_distance_1, price_sum_deviation
        ])

        # === MOMENTUM FEATURES ===
        for window in self.config.momentum_windows:
            if len(yes_prices) > window:
                yes_momentum = (yes_prices[-1] - yes_prices[-window-1]) / (yes_prices[-window-1] + 1e-8)
                no_momentum = (no_prices[-1] - no_prices[-window-1]) / (no_prices[-window-1] + 1e-8)
            else:
                yes_momentum = 0.0
                no_momentum = 0.0
            features.extend([yes_momentum * 100, no_momentum * 100])  # Scale for visibility

        # === ACCELERATION FEATURES ===
        for window in self.config.momentum_windows:
            half_window = window // 2
            if len(yes_prices) > window + half_window:
                # Recent momentum
                yes_mom_recent = yes_prices[-1] - yes_prices[-half_window-1]
                no_mom_recent = no_prices[-1] - no_prices[-half_window-1]
                # Previous momentum
                yes_mom_prev = yes_prices[-half_window-1] - yes_prices[-window-1]
                no_mom_prev = no_prices[-half_window-1] - no_prices[-window-1]
                # Acceleration = change in momentum
                yes_accel = yes_mom_recent - yes_mom_prev
                no_accel = no_mom_recent - no_mom_prev
            else:
                yes_accel = 0.0
                no_accel = 0.0
            features.extend([yes_accel * 100, no_accel * 100])

        # === VOLATILITY FEATURES ===
        for window in self.config.volatility_windows:
            if len(yes_prices) > window:
                yes_returns = np.diff(yes_prices[-window-1:]) / (yes_prices[-window-1:-1] + 1e-8)
                no_returns = np.diff(no_prices[-window-1:]) / (no_prices[-window-1:-1] + 1e-8)
                yes_vol = np.std(yes_returns) if len(yes_returns) > 1 else 0.0
                no_vol = np.std(no_returns) if len(no_returns) > 1 else 0.0
            else:
                yes_vol = 0.0
                no_vol = 0.0
            features.extend([yes_vol * 100, no_vol * 100])

        # === CONVERGENCE FEATURES ===
        conv_window = min(self.config.convergence_window, len(yes_prices) - 1)
        if conv_window > 5:
            # Convergence velocity (rate of approach to 0 or 1)
            price_changes = np.diff(yes_prices[-conv_window-1:])
            velocity = np.mean(price_changes)
            acceleration = np.mean(np.diff(price_changes)) if len(price_changes) > 1 else 0.0

            # Direction: positive = toward 1, negative = toward 0
            direction = 1.0 if velocity > 0 else -1.0 if velocity < 0 else 0.0

            # Distance to likely target
            if yes_price > 0.5:
                distance_to_target = 1.0 - yes_price  # Converging to 1
            else:
                distance_to_target = yes_price  # Converging to 0

            # Convergence strength (how fast relative to distance)
            convergence_strength = abs(velocity) / (distance_to_target + 0.01)

            # Reversal risk (price moving against trend)
            trend_sign = 1 if yes_price > 0.5 else -1
            reversal_risk = max(0, -trend_sign * velocity * 10)
        else:
            velocity = 0.0
            acceleration = 0.0
            direction = 0.0
            distance_to_target = abs(yes_price - 0.5)
            convergence_strength = 0.0
            reversal_risk = 0.0

        features.extend([
            velocity * 100, acceleration * 1000, direction,
            distance_to_target, convergence_strength, reversal_risk
        ])

        # === TIME FEATURES ===
        time_elapsed = 1.0 - time_remaining

        # Theta proxy: expected price decay rate
        # Higher theta when price is far from 0.5 and time is running out
        theta_proxy = (1.0 - abs(yes_price - 0.5) * 2) * (1.0 - time_remaining)

        # Urgency: exponential increase as settlement approaches
        urgency = np.exp(2.0 * time_elapsed) - 1.0  # 0 at start, ~6.4 at end
        urgency = min(urgency, 10.0)  # Cap

        features.extend([time_remaining, time_elapsed, theta_proxy, urgency / 10.0])

        # === RELATIVE VALUE FEATURES ===
        # Edge proxy: how far from 0.5 (extreme prices = strong conviction)
        edge_proxy = abs(yes_price - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

        # Mispricing signal: momentum opposite to price level
        # If price > 0.5 but momentum negative, possible mispricing
        if len(yes_prices) > 30:
            recent_momentum = yes_prices[-1] - yes_prices[-30]
            price_level = yes_price - 0.5
            mispricing_signal = -price_level * recent_momentum * 10  # Positive = potential mispricing
        else:
            mispricing_signal = 0.0

        # Mean reversion signal: distance from 0.5 (extremes tend to revert early in candle)
        mean_reversion_signal = (0.5 - yes_price) * time_remaining

        # Trend strength: consistency of direction
        if len(yes_prices) > 30:
            price_diffs = np.diff(yes_prices[-30:])
            trend_strength = np.mean(price_diffs > 0) - 0.5  # -0.5 to 0.5
        else:
            trend_strength = 0.0

        features.extend([edge_proxy, mispricing_signal, mean_reversion_signal, trend_strength * 2])

        # === BTC GUIDANCE FEATURES ===
        if btc_prices is not None and len(btc_prices) > 0 and btc_open is not None:
            btc_current = btc_prices[-1]
            btc_return = (btc_current - btc_open) / btc_open * 100  # Percentage

            # BTC momentum
            if len(btc_prices) > 60:
                btc_momentum = (btc_prices[-1] - btc_prices[-60]) / btc_prices[-60] * 100
            else:
                btc_momentum = 0.0

            # BTC volatility
            if len(btc_prices) > 30:
                btc_returns = np.diff(btc_prices[-31:]) / btc_prices[-31:-1]
                btc_volatility = np.std(btc_returns) * 100
            else:
                btc_volatility = 0.0

            # BTC direction signal
            btc_direction = 1.0 if btc_return > 0.05 else -1.0 if btc_return < -0.05 else 0.0

            # Correlation proxy: does YES price track BTC?
            if len(btc_prices) > 30 and len(yes_prices) > 30:
                btc_norm = (btc_prices[-30:] - btc_prices[-30]) / (btc_prices[-30] + 1e-8)
                yes_norm = (yes_prices[-30:] - yes_prices[-30]) / (yes_prices[-30] + 1e-8)
                if np.std(btc_norm) > 0 and np.std(yes_norm) > 0:
                    correlation_proxy = np.corrcoef(btc_norm, yes_norm)[0, 1]
                else:
                    correlation_proxy = 0.0
            else:
                correlation_proxy = 0.0

            # Divergence: BTC says one thing, price says another
            btc_implied = 0.5 + btc_return / 10  # Simple mapping
            btc_implied = np.clip(btc_implied, 0.1, 0.9)
            divergence = yes_price - btc_implied
        else:
            btc_return = 0.0
            btc_momentum = 0.0
            btc_volatility = 0.0
            btc_direction = 0.0
            correlation_proxy = 0.0
            divergence = 0.0

        features.extend([
            btc_return, btc_momentum, btc_volatility,
            btc_direction, correlation_proxy, divergence
        ])

        # Convert to numpy and clip
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -self.config.clip_value, self.config.clip_value)

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=self.config.clip_value, neginf=-self.config.clip_value)

        return features

    def compute_features_from_df(
        self,
        token_df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        time_remaining: float = 0.5,
    ) -> np.ndarray:
        """
        Compute features from DataFrames.

        Args:
            token_df: DataFrame with 'yes_price', 'no_price' columns
            btc_df: Optional DataFrame with 'price' column for BTC
            time_remaining: Fraction of candle remaining

        Returns:
            Feature vector
        """
        yes_prices = token_df['yes_price'].values
        no_prices = token_df['no_price'].values

        btc_prices = None
        btc_open = None
        if btc_df is not None and 'price' in btc_df.columns:
            btc_prices = btc_df['price'].values
            btc_open = btc_prices[0] if len(btc_prices) > 0 else None

        return self.compute_features(
            yes_prices=yes_prices,
            no_prices=no_prices,
            time_remaining=time_remaining,
            btc_prices=btc_prices,
            btc_open=btc_open,
        )

    def compute_batch_features(
        self,
        token_data: pd.DataFrame,
        btc_data: Optional[pd.DataFrame] = None,
        sequence_length: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute features for all candles in historical data.

        Args:
            token_data: DataFrame with timestamp, yes_price, no_price, outcome
            btc_data: Optional DataFrame with timestamp, price for BTC
            sequence_length: Number of steps per sequence

        Returns:
            Tuple of (features, labels) where features is (n_samples, feature_dim)
            and labels is (n_samples,) with 0/1 outcomes
        """
        logger.info("Computing batch features for training...")

        # Group by candle
        if 'candle_idx' not in token_data.columns:
            token_data = token_data.copy()
            token_data['timestamp_seconds'] = token_data['timestamp'].astype(np.int64) // 10**9
            token_data['candle_idx'] = token_data['timestamp_seconds'] // self.config.candle_seconds

        candle_groups = token_data.groupby('candle_idx')

        features_list = []
        labels_list = []

        for candle_idx, candle_df in candle_groups:
            if len(candle_df) < 60:  # Skip short candles
                continue

            # Get outcome (should be same for whole candle)
            if 'outcome' in candle_df.columns:
                outcome = candle_df['outcome'].iloc[-1]
            else:
                continue  # Skip if no outcome

            # Sample multiple points within the candle
            n_samples = min(5, len(candle_df) // 60)  # Up to 5 samples per candle
            sample_indices = np.linspace(60, len(candle_df) - 1, n_samples, dtype=int)

            for idx in sample_indices:
                # Time remaining at this point
                time_in_candle = idx / len(candle_df)
                time_remaining = 1.0 - time_in_candle

                # Get price history up to this point
                yes_prices = candle_df['yes_price'].values[:idx+1]
                no_prices = candle_df['no_price'].values[:idx+1]

                # Get BTC data if available
                btc_prices = None
                btc_open = None
                if btc_data is not None:
                    # Align BTC data to this candle
                    candle_start = candle_df['timestamp'].iloc[0]
                    candle_btc = btc_data[
                        (btc_data['timestamp'] >= candle_start) &
                        (btc_data['timestamp'] <= candle_df['timestamp'].iloc[idx])
                    ]
                    if len(candle_btc) > 0:
                        btc_prices = candle_btc['price'].values
                        btc_open = btc_prices[0]

                # Compute features
                features = self.compute_features(
                    yes_prices=yes_prices,
                    no_prices=no_prices,
                    time_remaining=time_remaining,
                    btc_prices=btc_prices,
                    btc_open=btc_open,
                )

                features_list.append(features)
                labels_list.append(outcome)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)

        logger.info(
            "Batch features computed",
            n_samples=len(X),
            feature_dim=self.feature_dim,
            label_distribution=f"YES={np.sum(y==1)}, NO={np.sum(y==0)}",
        )

        return X, y


def create_token_feature_builder(config: Optional[TokenFeatureConfig] = None) -> TokenFeatureBuilder:
    """Factory function to create TokenFeatureBuilder."""
    return TokenFeatureBuilder(config)
