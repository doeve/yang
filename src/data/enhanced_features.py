"""
Enhanced Feature Engineering for Market Prediction.

This module adds features that the model needs to LEARN the following principles
(without hardcoding them):

1. TREND AWARENESS: Understand price direction and consistency
2. TIME AWARENESS: Know when entering is risky vs safe
3. CONVERGENCE: Detect when prices are approaching certainty
4. ENTRY QUALITY: Signal when NOW is a good entry point
5. HOLD VALUE: Estimate value of continuing to hold

Key Design:
- Features are normalized and bounded
- No hardcoded thresholds - model learns optimal values
- Rich enough that model can learn all trading principles
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedFeatureConfig:
    """Configuration for enhanced features."""

    # Momentum windows (in ticks, typically 5 seconds each)
    momentum_windows: Tuple[int, ...] = (5, 15, 30, 60)

    # Volatility windows
    volatility_windows: Tuple[int, ...] = (15, 30, 60)

    # Trend analysis windows
    trend_windows: Tuple[int, ...] = (10, 30, 60)

    # EMA smoothing
    ema_alpha: float = 0.3

    # Feature bounds
    clip_value: float = 5.0


class EnhancedFeatureBuilder:
    """
    Build enhanced features for market prediction.

    Output dimension: 74 features
    - Price State: 8 features
    - Momentum: 8 features (4 windows x 2 directions)
    - Volatility: 6 features
    - Trend Analysis: 10 features
    - Convergence: 8 features
    - Time Features: 6 features
    - Entry Quality: 8 features
    - BTC Guidance: 10 features
    - BTC Volume: 3 features
    - Cross-Signal: 7 features
    """

    def __init__(self, config: Optional[EnhancedFeatureConfig] = None):
        self.config = config or EnhancedFeatureConfig()
        self.feature_dim = 74
        self._build_feature_names()
        logger.info(
            "EnhancedFeatureBuilder initialized",
            feature_dim=self.feature_dim,
        )

    def _build_feature_names(self) -> List[str]:
        """Build feature names for debugging."""
        self.feature_names = [
            # Price State (8)
            'yes_price', 'no_price', 'spread', 'mid_price',
            'distance_to_0', 'distance_to_1', 'price_extremity', 'price_imbalance',

            # Momentum (8)
            *[f'yes_momentum_{w}' for w in self.config.momentum_windows],
            *[f'no_momentum_{w}' for w in self.config.momentum_windows],

            # Volatility (6)
            *[f'volatility_{w}' for w in self.config.volatility_windows],
            'volatility_ratio', 'volatility_trend', 'volatility_zscore',

            # Trend Analysis (10)
            *[f'trend_strength_{w}' for w in self.config.trend_windows],
            *[f'trend_consistency_{w}' for w in self.config.trend_windows],
            'trend_acceleration', 'trend_reversal_risk', 'trend_age', 'trend_maturity',

            # Convergence (8)
            'convergence_velocity', 'convergence_acceleration',
            'target_probability', 'time_adjusted_momentum',
            'convergence_confidence', 'convergence_phase',
            'resistance_distance', 'support_distance',

            # Time Features (6)
            'time_remaining', 'time_remaining_sqrt', 'time_remaining_log',
            'time_pressure', 'phase_indicator', 'settlement_urgency',

            # Entry Quality (8)
            'entry_quality_yes', 'entry_quality_no',
            'relative_value_yes', 'relative_value_no',
            'timing_signal', 'momentum_alignment',
            'volatility_regime', 'trend_entry_score',

            # BTC Guidance (10)
            'btc_return', 'btc_momentum', 'btc_volatility',
            'btc_direction', 'btc_correlation', 'btc_divergence',
            'btc_trend_strength', 'btc_regime', 'btc_interaction', 'btc_lead_signal',

            # BTC Volume (3)
            'btc_volume_ratio', 'btc_volume_trend', 'btc_volume_price_corr',

            # Cross-Signal (7)
            'momentum_volatility_ratio', 'trend_time_interaction',
            'convergence_quality', 'signal_agreement', 'noise_ratio',
            'regime_clarity', 'action_confidence',
        ]

        return self.feature_names

    def _ema(self, prices: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average."""
        if len(prices) == 0:
            return prices
        ema = np.zeros_like(prices, dtype=np.float64)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safe division with default value."""
        if abs(b) < 1e-8:
            return default
        return a / b

    def compute_features(
        self,
        yes_prices: np.ndarray,
        no_prices: np.ndarray,
        time_remaining: float,
        btc_prices: Optional[np.ndarray] = None,
        btc_open: Optional[float] = None,
        btc_volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute enhanced feature vector.

        Args:
            yes_prices: Historical YES prices (most recent last)
            no_prices: Historical NO prices
            time_remaining: Fraction of candle remaining (1.0 = start)
            btc_prices: Optional BTC price history
            btc_open: BTC candle open price
            btc_volumes: Optional BTC volume history

        Returns:
            Feature vector (74 dimensions)
        """
        features = []

        # Ensure minimum history
        min_history = max(self.config.momentum_windows) + 10
        if len(yes_prices) < min_history:
            pad_len = min_history - len(yes_prices)
            yes_prices = np.concatenate([np.full(pad_len, yes_prices[0]), yes_prices])
            no_prices = np.concatenate([np.full(pad_len, no_prices[0]), no_prices])

        yes_price = yes_prices[-1]
        no_price = no_prices[-1]

        # ==================== PRICE STATE (8) ====================
        spread = abs(yes_price + no_price - 1.0)
        mid_price = (yes_price + no_price) / 2.0
        distance_to_0 = yes_price
        distance_to_1 = 1.0 - yes_price
        price_extremity = 2.0 * abs(yes_price - 0.5)  # 0 at 0.5, 1 at extremes
        price_imbalance = yes_price - 0.5  # Direction from center

        features.extend([
            yes_price, no_price, spread, mid_price,
            distance_to_0, distance_to_1, price_extremity, price_imbalance
        ])

        # ==================== MOMENTUM (8) ====================
        for window in self.config.momentum_windows:
            if len(yes_prices) > window:
                yes_ema = self._ema(yes_prices[-window-1:], self.config.ema_alpha)
                no_ema = self._ema(no_prices[-window-1:], self.config.ema_alpha)
                yes_mom = self._safe_divide(yes_ema[-1] - yes_ema[0], yes_ema[0], 0.0)
                no_mom = self._safe_divide(no_ema[-1] - no_ema[0], no_ema[0], 0.0)
            else:
                yes_mom, no_mom = 0.0, 0.0
            features.append(yes_mom * 10)  # Scale for visibility

        for window in self.config.momentum_windows:
            if len(no_prices) > window:
                no_ema = self._ema(no_prices[-window-1:], self.config.ema_alpha)
                no_mom = self._safe_divide(no_ema[-1] - no_ema[0], no_ema[0], 0.0)
            else:
                no_mom = 0.0
            features.append(no_mom * 10)

        # ==================== VOLATILITY (6) ====================
        volatilities = []
        for window in self.config.volatility_windows:
            if len(yes_prices) > window:
                returns = np.diff(yes_prices[-window-1:]) / (yes_prices[-window-1:-1] + 1e-8)
                vol = np.std(returns) if len(returns) > 1 else 0.0
            else:
                vol = 0.0
            volatilities.append(vol)
            features.append(vol * 100)

        # Volatility ratio (short-term / long-term) - indicates regime change
        vol_ratio = self._safe_divide(volatilities[0], volatilities[-1] + 1e-8, 1.0)
        features.append(np.clip(vol_ratio, 0.1, 10.0))

        # Volatility trend
        if len(volatilities) >= 2:
            vol_trend = volatilities[-1] - volatilities[0]
        else:
            vol_trend = 0.0
        features.append(vol_trend * 100)

        # Volatility z-score
        if len(yes_prices) > 60:
            recent_vol = np.std(np.diff(yes_prices[-15:]))
            historical_vol = np.std(np.diff(yes_prices[-60:]))
            vol_zscore = self._safe_divide(recent_vol - historical_vol, historical_vol, 0.0)
        else:
            vol_zscore = 0.0
        features.append(np.clip(vol_zscore, -3.0, 3.0))

        # ==================== TREND ANALYSIS (12) ====================
        for window in self.config.trend_windows:
            if len(yes_prices) > window:
                price_diffs = np.diff(yes_prices[-window-1:])
                # Trend strength: net movement / total movement
                net_move = abs(yes_prices[-1] - yes_prices[-window-1])
                total_move = np.sum(np.abs(price_diffs)) + 1e-8
                trend_strength = net_move / total_move
            else:
                trend_strength = 0.0
            features.append(trend_strength)

        for window in self.config.trend_windows:
            if len(yes_prices) > window:
                price_diffs = np.diff(yes_prices[-window-1:])
                # Trend consistency: fraction of moves in same direction
                positive_moves = np.sum(price_diffs > 0)
                consistency = abs(positive_moves / len(price_diffs) - 0.5) * 2
            else:
                consistency = 0.0
            features.append(consistency)

        # Trend acceleration
        if len(yes_prices) > 30:
            mom_recent = yes_prices[-1] - yes_prices[-10]
            mom_earlier = yes_prices[-10] - yes_prices[-30]
            trend_accel = mom_recent - mom_earlier
        else:
            trend_accel = 0.0
        features.append(trend_accel * 10)

        # Trend reversal risk
        if len(yes_prices) > 30:
            current_dir = np.sign(yes_prices[-1] - yes_prices[-10])
            recent_dir = np.sign(yes_prices[-10] - yes_prices[-30])
            reversal_risk = float(current_dir != recent_dir)
        else:
            reversal_risk = 0.0
        features.append(reversal_risk)

        # Trend age (how long in current direction)
        if len(yes_prices) > 5:
            current_dir = np.sign(yes_prices[-1] - yes_prices[-2])
            age = 1
            for i in range(2, min(60, len(yes_prices))):
                if np.sign(yes_prices[-i] - yes_prices[-i-1]) == current_dir:
                    age += 1
                else:
                    break
            trend_age = min(age / 30.0, 1.0)
        else:
            trend_age = 0.0
        features.append(trend_age)

        # Trend maturity (combination of age and strength)
        trend_maturity = trend_age * trend_strength if 'trend_strength' in dir() else 0.0
        features.append(trend_maturity)

        # ==================== CONVERGENCE (8) ====================
        if len(yes_prices) > 15:
            # Velocity toward extremes
            convergence_velocity = (yes_prices[-1] - yes_prices[-15]) / 15
            features.append(convergence_velocity * 100)

            # Acceleration
            if len(yes_prices) > 30:
                vel_recent = yes_prices[-1] - yes_prices[-15]
                vel_earlier = yes_prices[-15] - yes_prices[-30]
                conv_accel = vel_recent - vel_earlier
            else:
                conv_accel = 0.0
            features.append(conv_accel * 100)
        else:
            features.extend([0.0, 0.0])

        # Target probability (which extreme is likely)
        if yes_price > 0.5:
            target_prob = 2 * (yes_price - 0.5)  # 0 at 0.5, 1 at 1.0
        else:
            target_prob = -2 * (0.5 - yes_price)  # -1 at 0.0, 0 at 0.5
        features.append(target_prob)

        # Time-adjusted momentum (momentum matters more late in candle)
        time_elapsed = 1.0 - time_remaining
        if len(yes_prices) > 15:
            momentum_30 = yes_prices[-1] - yes_prices[-min(30, len(yes_prices)-1)]
            time_adj_mom = momentum_30 * (1.0 + time_elapsed)
        else:
            time_adj_mom = 0.0
        features.append(time_adj_mom)

        # Convergence confidence (how sure are we about the target)
        conv_conf = price_extremity * (1.0 - time_remaining) + trend_strength
        features.append(min(conv_conf, 2.0))

        # Convergence phase (0=early/uncertain, 1=mid/trending, 2=late/converged)
        if time_remaining > 0.7:
            phase = 0.0
        elif time_remaining > 0.3:
            phase = 1.0
        else:
            phase = 2.0
        features.append(phase / 2.0)

        # Resistance/support distance
        if len(yes_prices) > 30:
            max_price = np.max(yes_prices[-30:])
            min_price = np.min(yes_prices[-30:])
            resistance_dist = max_price - yes_price
            support_dist = yes_price - min_price
        else:
            resistance_dist = 1.0 - yes_price
            support_dist = yes_price
        features.extend([resistance_dist, support_dist])

        # ==================== TIME FEATURES (6) ====================
        features.append(time_remaining)
        features.append(np.sqrt(time_remaining))  # Non-linear time scale
        features.append(np.log(time_remaining + 0.01))  # Log time (captures urgency)

        # Time pressure (exponential as time runs out)
        time_pressure = np.exp(3.0 * time_elapsed) - 1.0
        features.append(min(time_pressure / 10.0, 1.0))

        # Phase indicator (discrete phase)
        if time_remaining > 0.7:
            phase_ind = 0.0  # Early: uncertainty high
        elif time_remaining > 0.4:
            phase_ind = 0.5  # Mid: trend forming
        else:
            phase_ind = 1.0  # Late: convergence
        features.append(phase_ind)

        # Settlement urgency (combined time + position)
        urgency = (1.0 - time_remaining) ** 2 * (1.0 + price_extremity)
        features.append(min(urgency, 2.0))

        # ==================== ENTRY QUALITY (8) ====================
        # Entry quality for YES (higher = better entry point)
        # Good entry: price < fair value, momentum positive, not too late
        if len(yes_prices) > 30:
            price_discount_yes = 0.5 - yes_price  # Positive if YES is cheap
            mom_30 = yes_prices[-1] - yes_prices[-30]
            entry_quality_yes = (
                price_discount_yes * 2 +  # Price component
                (mom_30 if mom_30 > 0 else 0) * 5 +  # Momentum component
                time_remaining * 0.5  # Time component (earlier is better)
            )
        else:
            entry_quality_yes = 0.0
        features.append(np.clip(entry_quality_yes, -2.0, 2.0))

        # Entry quality for NO
        if len(no_prices) > 30:
            price_discount_no = 0.5 - no_price
            mom_30_no = no_prices[-1] - no_prices[-30]
            entry_quality_no = (
                price_discount_no * 2 +
                (mom_30_no if mom_30_no > 0 else 0) * 5 +
                time_remaining * 0.5
            )
        else:
            entry_quality_no = 0.0
        features.append(np.clip(entry_quality_no, -2.0, 2.0))

        # Relative value (vs expected price at settlement)
        # If price is 0.7 and trend is strongly up, relative value for YES is poor
        relative_value_yes = 1.0 - yes_price - (target_prob if target_prob > 0 else 0)
        relative_value_no = no_price - (abs(target_prob) if target_prob < 0 else 0)
        features.extend([
            np.clip(relative_value_yes, -1.0, 1.0),
            np.clip(relative_value_no, -1.0, 1.0)
        ])

        # Timing signal (is NOW a good time to act?)
        # Good timing: low volatility, clear trend, not extreme price
        timing_signal = (1.0 - vol_ratio) * trend_strength * (1.0 - price_extremity)
        features.append(np.clip(timing_signal, -1.0, 1.0))

        # Momentum alignment (does momentum align with potential position?)
        if len(yes_prices) > 15:
            mom_sign = np.sign(yes_prices[-1] - yes_prices[-15])
            pos_sign = np.sign(yes_price - 0.5)
            mom_alignment = mom_sign * pos_sign  # +1 if aligned, -1 if opposed
        else:
            mom_alignment = 0.0
        features.append(mom_alignment)

        # Volatility regime (0=low, 1=high)
        if len(volatilities) > 0:
            vol_regime = min(volatilities[-1] * 50, 1.0)
        else:
            vol_regime = 0.5
        features.append(vol_regime)

        # Trend entry score (composite)
        trend_entry = trend_strength * trend_consistency * (1.0 - reversal_risk) if 'trend_consistency' in dir() else 0.0
        features.append(np.clip(trend_entry, 0.0, 1.0))

        # ==================== BTC GUIDANCE (10) ====================
        if btc_prices is not None and len(btc_prices) > 0 and btc_open is not None:
            btc_current = btc_prices[-1]
            btc_return = (btc_current - btc_open) / btc_open * 100

            # BTC momentum
            if len(btc_prices) > 60:
                btc_ema = self._ema(btc_prices[-61:], self.config.ema_alpha)
                btc_momentum = (btc_ema[-1] - btc_ema[0]) / btc_ema[0] * 100
            else:
                btc_momentum = 0.0

            # BTC volatility
            if len(btc_prices) > 30:
                btc_returns = np.diff(btc_prices[-31:]) / btc_prices[-31:-1]
                btc_volatility = np.std(btc_returns) * 100
            else:
                btc_volatility = 0.0

            # BTC direction (smooth)
            btc_direction = np.tanh(btc_return * 2)

            # BTC-token correlation
            if len(btc_prices) > 30 and len(yes_prices) > 30:
                try:
                    corr, _ = stats.spearmanr(btc_prices[-30:], yes_prices[-30:])
                    btc_correlation = 0.0 if np.isnan(corr) else corr
                except:
                    btc_correlation = 0.0
            else:
                btc_correlation = 0.0

            # Divergence (token vs BTC expectation)
            btc_implied = 1.0 / (1.0 + np.exp(-btc_return * 0.5))
            btc_divergence = yes_price - btc_implied

            # BTC trend strength
            if len(btc_prices) > 30:
                btc_diffs = np.diff(btc_prices[-31:])
                btc_trend = abs(np.sum(btc_diffs)) / (np.sum(np.abs(btc_diffs)) + 1e-8)
            else:
                btc_trend = 0.0

            # BTC regime
            if len(btc_prices) > 60:
                short_vol = np.std(np.diff(btc_prices[-15:])) if len(btc_prices) > 15 else 0.001
                long_vol = np.std(np.diff(btc_prices[-60:]))
                btc_regime = np.tanh((long_vol / (short_vol + 1e-8) - 2) * 0.5)
            else:
                btc_regime = 0.0

            # BTC-token interaction
            btc_interaction = btc_direction * (yes_price - 0.5) * 2

            # BTC lead signal (does BTC move predict token move?)
            if len(btc_prices) > 10 and len(yes_prices) > 10:
                btc_move_5 = btc_prices[-1] - btc_prices[-5]
                yes_move_5 = yes_prices[-1] - yes_prices[-5]
                btc_lead = np.sign(btc_move_5) * np.sign(yes_move_5)
            else:
                btc_lead = 0.0

            features.extend([
                np.clip(btc_return, -10, 10),
                np.clip(btc_momentum, -10, 10),
                np.clip(btc_volatility, 0, 5),
                btc_direction,
                btc_correlation,
                np.clip(btc_divergence, -1, 1),
                btc_trend,
                btc_regime,
                btc_interaction,
                btc_lead,
            ])
        else:
            features.extend([0.0] * 10)

        # ==================== BTC VOLUME (3) ====================
        if btc_volumes is not None and len(btc_volumes) > 10:
            # Volume spike detection: recent avg / overall avg
            recent_vol_avg = np.mean(btc_volumes[-10:])
            overall_vol_avg = np.mean(btc_volumes) + 1e-8
            btc_vol_ratio = recent_vol_avg / overall_vol_avg
            features.append(np.clip(btc_vol_ratio, 0.0, 5.0))

            # Volume trend: slope of recent 30-tick volume
            vol_window = min(30, len(btc_volumes))
            if vol_window >= 5:
                x_range = np.arange(vol_window)
                slope = stats.linregress(x_range, btc_volumes[-vol_window:]).slope
                vol_norm = overall_vol_avg
                btc_vol_trend = slope / vol_norm
                features.append(np.clip(btc_vol_trend, -5.0, 5.0))
            else:
                features.append(0.0)

            # Volume-price correlation
            if btc_prices is not None and len(btc_prices) >= 10:
                corr_window = min(30, len(btc_volumes), len(btc_prices))
                if corr_window >= 5:
                    abs_vol_changes = np.abs(np.diff(btc_volumes[-corr_window:]))
                    abs_price_changes = np.abs(np.diff(btc_prices[-corr_window:]))
                    if len(abs_vol_changes) >= 3:
                        try:
                            corr, _ = stats.spearmanr(abs_vol_changes, abs_price_changes)
                            btc_vol_price_corr = 0.0 if np.isnan(corr) else corr
                        except Exception:
                            btc_vol_price_corr = 0.0
                    else:
                        btc_vol_price_corr = 0.0
                else:
                    btc_vol_price_corr = 0.0
                features.append(btc_vol_price_corr)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])

        # ==================== CROSS-SIGNAL (7) ====================
        # Momentum/volatility ratio
        if volatilities and len(self.config.momentum_windows) > 0:
            avg_mom = abs(features[8])  # First momentum feature
            avg_vol = volatilities[-1] if volatilities[-1] > 0 else 0.01
            mom_vol_ratio = avg_mom / avg_vol
        else:
            mom_vol_ratio = 0.0
        features.append(np.clip(mom_vol_ratio, 0, 10))

        # Trend-time interaction
        trend_time = trend_strength * time_elapsed if 'trend_strength' in dir() else 0.0
        features.append(trend_time)

        # Convergence quality
        conv_quality = conv_conf * (1.0 - vol_regime) if 'conv_conf' in dir() else 0.0
        features.append(np.clip(conv_quality, 0, 2))

        # Signal agreement (do multiple signals agree?)
        signals = [
            np.sign(features[8]) if len(features) > 8 else 0,  # Momentum
            np.sign(target_prob),  # Target direction
            np.sign(btc_direction) if btc_prices is not None else 0,  # BTC
        ]
        signal_agree = abs(sum(signals)) / 3.0
        features.append(signal_agree)

        # Noise ratio (how much of movement is noise vs signal)
        if len(volatilities) > 0 and 'net_move' in dir():
            noise_ratio = volatilities[-1] / (abs(net_move) + 1e-8) if net_move != 0 else 1.0
        else:
            noise_ratio = 0.5
        features.append(np.clip(noise_ratio, 0, 5))

        # Regime clarity (is the regime clear or ambiguous?)
        regime_clarity = (1.0 - vol_ratio) * trend_strength * signal_agree
        features.append(np.clip(regime_clarity, 0, 1))

        # Action confidence (overall confidence in taking action)
        action_conf = (
            (1.0 - vol_ratio) * 0.3 +
            trend_strength * 0.3 +
            conv_conf * 0.2 +
            signal_agree * 0.2
        )
        features.append(np.clip(action_conf, 0, 1))

        # ==================== FINALIZE ====================
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -self.config.clip_value, self.config.clip_value)
        features = np.nan_to_num(features, nan=0.0, posinf=self.config.clip_value, neginf=-self.config.clip_value)

        assert len(features) == self.feature_dim, f"Expected {self.feature_dim} features, got {len(features)}"

        return features


def create_enhanced_feature_builder(config: Optional[EnhancedFeatureConfig] = None) -> EnhancedFeatureBuilder:
    """Factory function."""
    return EnhancedFeatureBuilder(config)
