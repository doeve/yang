"""
KAMA Spectrum Feature Engineering for Time-Aware Attention Model.

This module generates:
1. KAMA Spectrum: Multiple Kaufman Adaptive Moving Averages at different periods
2. Time Context: time_remaining, time_decay, time_pressure for 15-minute markets
3. Strike Context: distance_to_strike, convergence rate
4. Volatility Context: ATR, volatility ratios

The attention mechanism uses context features to dynamically weight KAMA features,
learning to focus on fast KAMAs near expiration and slow KAMAs early in the candle.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KAMASpectrumConfig:
    """Configuration for KAMA spectrum and context features."""
    
    # KAMA periods (sorted fast → slow)
    # For 15-minute markets with ~1 tick per second:
    # - Fast (5-13): capture 5-13 second momentum
    # - Medium (16-30): capture 16-30 second trends  
    # - Slow (40-80): capture overall direction
    kama_periods: Tuple[int, ...] = (5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80)
    
    # KAMA smoothing constants (Kaufman defaults)
    fast_sc: float = 2.0   # Fast smoothing constant
    slow_sc: float = 30.0  # Slow smoothing constant
    
    # Volatility windows for context
    atr_windows: Tuple[int, ...] = (10, 20)
    vol_windows: Tuple[int, ...] = (10, 30)
    
    # Feature clipping
    clip_value: float = 5.0


class KAMAFeatureBuilder:
    """
    Build KAMA spectrum and time-aware context features.
    
    Output dimensions:
    - KAMA Spectrum: num_kamas * 2 features (deviation + slope per KAMA)
    - Context: 12 features (time: 5, strike: 4, volatility: 3)
    """
    
    def __init__(self, config: Optional[KAMASpectrumConfig] = None):
        self.config = config or KAMASpectrumConfig()
        self.num_kamas = len(self.config.kama_periods)
        self.kama_feature_dim = self.num_kamas * 2  # deviation + slope per KAMA
        self.context_feature_dim = 12  # time(5) + strike(4) + vol(3)
        self.total_feature_dim = self.kama_feature_dim + self.context_feature_dim
        
        self._build_feature_names()
        logger.info(
            "KAMAFeatureBuilder initialized",
            num_kamas=self.num_kamas,
            kama_feature_dim=self.kama_feature_dim,
            context_feature_dim=self.context_feature_dim,
            total_feature_dim=self.total_feature_dim,
        )
    
    def _build_feature_names(self) -> List[str]:
        """Build feature names for debugging/analysis."""
        self.kama_feature_names = []
        for period in self.config.kama_periods:
            self.kama_feature_names.extend([
                f'kama_{period}_deviation',
                f'kama_{period}_slope',
            ])
        
        self.context_feature_names = [
            # Time context (5)
            'time_remaining_pct',
            'time_decay',
            'time_pressure',
            'time_phase',
            'urgency',
            # Strike context (4)
            'distance_to_strike',
            'abs_distance_to_strike',
            'near_strike',
            'convergence_rate',
            # Volatility context (3)
            'atr_normalized',
            'vol_ratio',
            'vol_regime',
        ]
        
        self.all_feature_names = self.kama_feature_names + self.context_feature_names
        return self.all_feature_names
    
    def compute_kama(
        self,
        prices: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """
        Compute Kaufman's Adaptive Moving Average.
        
        KAMA adapts to market efficiency:
        - High efficiency (trending): faster response
        - Low efficiency (choppy): slower response
        
        Formula:
            ER = abs(price_change) / sum(abs(price_changes))  # Efficiency Ratio
            SC = (ER * (fast_sc - slow_sc) + slow_sc)^2       # Smoothing Constant
            KAMA = KAMA_prev + SC * (Price - KAMA_prev)
        
        Args:
            prices: Price array (most recent last)
            period: Lookback period for efficiency ratio
            
        Returns:
            KAMA array (same length as prices)
        """
        if len(prices) < period + 1:
            # Not enough data, return SMA-like behavior
            return self._simple_ema(prices, alpha=2.0 / (period + 1))
        
        fast_sc = 2.0 / (self.config.fast_sc + 1)
        slow_sc = 2.0 / (self.config.slow_sc + 1)
        
        kama = np.zeros_like(prices, dtype=np.float64)
        kama[0] = prices[0]
        
        for i in range(1, len(prices)):
            if i < period:
                # Not enough history, use simple EMA
                alpha = 2.0 / (i + 2)
                kama[i] = alpha * prices[i] + (1 - alpha) * kama[i-1]
            else:
                # Compute Efficiency Ratio
                price_change = abs(prices[i] - prices[i - period])
                volatility = np.sum(np.abs(np.diff(prices[i - period:i + 1])))
                
                if volatility > 1e-10:
                    er = price_change / volatility
                else:
                    er = 0.0
                
                # Compute Smoothing Constant
                sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                
                # Update KAMA
                kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
        
        return kama
    
    def _simple_ema(self, prices: np.ndarray, alpha: float) -> np.ndarray:
        """Simple exponential moving average."""
        if len(prices) == 0:
            return prices
        ema = np.zeros_like(prices, dtype=np.float64)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _compute_atr(self, prices: np.ndarray, window: int) -> float:
        """
        Compute Average True Range (simplified for single price series).
        Uses absolute returns as proxy for true range.
        """
        if len(prices) < window + 1:
            return 0.0
        
        returns = np.abs(np.diff(prices[-window-1:]))
        return np.mean(returns)
    
    def compute_kama_spectrum(
        self,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute KAMA spectrum features.
        
        For each KAMA period, computes:
        1. Deviation: (price - KAMA) / price  # How far price is from KAMA
        2. Slope: KAMA change rate  # Trend direction/strength
        
        Args:
            prices: Price array (most recent last)
            
        Returns:
            KAMA spectrum features (num_kamas * 2,)
        """
        current_price = prices[-1]
        features = []
        
        for period in self.config.kama_periods:
            kama = self.compute_kama(prices, period)
            kama_current = kama[-1]
            
            # Deviation: how far price is from this KAMA
            if current_price > 1e-8:
                deviation = (current_price - kama_current) / current_price
            else:
                deviation = 0.0
            
            # Slope: KAMA trend direction (normalized)
            if len(kama) >= 5:
                slope = (kama[-1] - kama[-5]) / (kama[-5] + 1e-8)
            else:
                slope = 0.0
            
            features.extend([
                np.clip(deviation, -1.0, 1.0),
                np.clip(slope * 10, -1.0, 1.0),  # Scale for visibility
            ])
        
        return np.array(features, dtype=np.float32)
    
    def compute_context_features(
        self,
        yes_prices: np.ndarray,
        time_remaining: float,
    ) -> np.ndarray:
        """
        Compute time-aware context features.
        
        These features drive the attention mechanism to weight KAMAs appropriately
        based on time, strike distance, and volatility.
        
        Args:
            yes_prices: YES token price history
            time_remaining: Fraction of candle remaining (1.0 = start, 0.0 = end)
            
        Returns:
            Context features (12,)
        """
        current_price = yes_prices[-1]
        features = []
        
        # ==================== TIME CONTEXT (5) ====================
        
        # 1. time_remaining_pct: Raw time remaining [0, 1]
        time_remaining_pct = np.clip(time_remaining, 0.0, 1.0)
        features.append(time_remaining_pct)
        
        # 2. time_decay: Exponential decay near end
        # High value early, decays rapidly near expiration
        time_decay = np.exp(-3.0 * (1.0 - time_remaining))
        features.append(time_decay)
        
        # 3. time_pressure: Quadratic pressure increase
        # Low early, increases quadratically near end
        time_elapsed = 1.0 - time_remaining
        time_pressure = time_elapsed ** 2
        features.append(time_pressure)
        
        # 4. time_phase: Discrete phase indicator
        # 0 = early (>60% remaining), 0.5 = mid, 1.0 = late (<30% remaining)
        if time_remaining > 0.6:
            time_phase = 0.0
        elif time_remaining > 0.3:
            time_phase = 0.5
        else:
            time_phase = 1.0
        features.append(time_phase)
        
        # ==================== STRIKE CONTEXT (4) ====================
        
        # 5. distance_to_strike: Signed distance from 0.50 [-0.5, +0.5]
        distance_to_strike = current_price - 0.5
        features.append(distance_to_strike)
        
        # 6. abs_distance_to_strike: Magnitude [0, 0.5]
        abs_distance = abs(distance_to_strike)
        features.append(abs_distance)
        
        # 7. near_strike: Binary indicator for uncertainty zone
        near_strike = 1.0 if abs_distance < 0.10 else 0.0
        features.append(near_strike)
        
        # 8. convergence_rate: Price velocity toward extremes
        if len(yes_prices) >= 10:
            price_change = yes_prices[-1] - yes_prices[-10]
            convergence_rate = price_change * np.sign(distance_to_strike)
        else:
            convergence_rate = 0.0
        features.append(np.clip(convergence_rate * 10, -1.0, 1.0))
        
        # 9. urgency: Combined time pressure and distance
        # Higher urgency = more time pressure AND far from strike
        urgency = time_pressure * (1.0 + abs_distance * 2)
        features.append(np.clip(urgency, 0.0, 2.0))
        
        # ==================== VOLATILITY CONTEXT (3) ====================
        
        # 10. atr_normalized: Normalized ATR
        atr_short = self._compute_atr(yes_prices, self.config.atr_windows[0])
        # Normalize by typical price range for binary market
        atr_normalized = atr_short / 0.1  # 10% is high volatility for binary
        features.append(np.clip(atr_normalized, 0.0, 2.0))
        
        # 11. vol_ratio: Short-term / long-term volatility
        if len(yes_prices) >= self.config.vol_windows[1]:
            short_vol = np.std(np.diff(yes_prices[-self.config.vol_windows[0]:]))
            long_vol = np.std(np.diff(yes_prices[-self.config.vol_windows[1]:]))
            vol_ratio = short_vol / (long_vol + 1e-8)
        else:
            vol_ratio = 1.0
        features.append(np.clip(vol_ratio, 0.1, 5.0))
        
        # 12. vol_regime: Discrete volatility regime
        # 0 = low, 0.5 = normal, 1.0 = high
        if atr_normalized < 0.3:
            vol_regime = 0.0
        elif atr_normalized < 0.7:
            vol_regime = 0.5
        else:
            vol_regime = 1.0
        features.append(vol_regime)
        
        return np.array(features, dtype=np.float32)
    
    def compute_features(
        self,
        yes_prices: np.ndarray,
        time_remaining: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute complete feature set: KAMA spectrum + context.
        
        Args:
            yes_prices: YES token price history (most recent last)
            time_remaining: Fraction of candle remaining [0, 1]
            
        Returns:
            Tuple of (kama_spectrum, context_features)
            - kama_spectrum: (num_kamas * 2,) KAMA-based features
            - context_features: (12,) time-aware context for attention
        """
        # Ensure minimum history
        min_history = max(self.config.kama_periods) + 10
        if len(yes_prices) < min_history:
            pad_len = min_history - len(yes_prices)
            yes_prices = np.concatenate([
                np.full(pad_len, yes_prices[0]),
                yes_prices
            ])
        
        kama_spectrum = self.compute_kama_spectrum(yes_prices)
        context_features = self.compute_context_features(yes_prices, time_remaining)
        
        # Clip and sanitize
        kama_spectrum = np.clip(kama_spectrum, -self.config.clip_value, self.config.clip_value)
        context_features = np.clip(context_features, -self.config.clip_value, self.config.clip_value)
        
        kama_spectrum = np.nan_to_num(kama_spectrum, nan=0.0)
        context_features = np.nan_to_num(context_features, nan=0.0)
        
        return kama_spectrum, context_features
    
    def compute_combined_features(
        self,
        yes_prices: np.ndarray,
        time_remaining: float,
    ) -> np.ndarray:
        """
        Compute combined feature vector (for compatibility).
        
        Returns concatenated [kama_spectrum, context_features].
        """
        kama_spectrum, context_features = self.compute_features(yes_prices, time_remaining)
        return np.concatenate([kama_spectrum, context_features])


def create_kama_feature_builder(
    config: Optional[KAMASpectrumConfig] = None
) -> KAMAFeatureBuilder:
    """Factory function for KAMAFeatureBuilder."""
    return KAMAFeatureBuilder(config)
