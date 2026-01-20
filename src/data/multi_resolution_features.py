"""
Multi-resolution feature engineering for probability model.

Builds feature tensors at multiple time resolutions (15s, 1m, 5m, 15m)
for LSTM probability prediction. Designed for 15-minute crypto Polymarket
prediction markets.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MultiResolutionConfig:
    """Configuration for multi-resolution feature builder."""
    
    # Base data resolution (seconds)
    base_resolution_seconds: int = 1
    
    # Target resolutions to compute features for
    target_resolutions_seconds: list[int] = field(
        default_factory=lambda: [15, 60, 300, 900]  # 15s, 1m, 5m, 15m
    )
    
    # Lookback windows for each resolution (in that resolution's units)
    lookback_windows: list[int] = field(
        default_factory=lambda: [60, 60, 36, 12]  # 15min, 1h, 3h, 3h
    )
    
    # Feature computation parameters
    volatility_window: int = 20
    ema_fast_period: int = 5
    ema_slow_period: int = 20
    
    # Forex feature weights (lower than BTC)
    forex_weight: float = 0.3


class MultiResolutionFeatureBuilder:
    """
    Builds feature tensors at multiple time resolutions.
    
    For each resolution, computes:
    - Log returns
    - Rolling volatility (std of returns)
    - Momentum (EMA fast - EMA slow slope)
    
    Plus cross-asset features:
    - BTC-DXY divergence
    - BTC-DXY correlation flag
    - Sign disagreement flags
    
    Output shape: (sequence_length, feature_dim)
    where feature_dim = 24 by default.
    """
    
    def __init__(self, config: Optional[MultiResolutionConfig] = None):
        self.config = config or MultiResolutionConfig()
        
        # Calculate feature dimension
        # Per resolution: 3 features (return, vol, momentum)
        # Resolutions: 4
        # Time features: 3 (time_remaining, hour_sin, hour_cos)
        # DXY: 3 (return, momentum, vol)
        # EURUSD: 3 (return, momentum, vol)
        # Relative: 3 (divergence, correlation, sign_disagree)
        self.btc_features_per_res = 3
        self.num_resolutions = len(self.config.target_resolutions_seconds)
        self.time_features = 3
        self.dxy_features = 3
        self.eurusd_features = 3
        self.relative_features = 3
        
        self.feature_dim = (
            self.btc_features_per_res * self.num_resolutions +  # BTC
            self.time_features +
            self.dxy_features +
            self.eurusd_features +
            self.relative_features
        )
        
        logger.info(
            "MultiResolutionFeatureBuilder initialized",
            feature_dim=self.feature_dim,
            resolutions=self.config.target_resolutions_seconds,
        )
    
    def _resample_to_resolution(
        self,
        data: pd.DataFrame,
        resolution_seconds: int,
        price_col: str = "price",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """Resample data to target resolution."""
        if "timestamp" not in data.columns:
            raise ValueError("Data must have 'timestamp' column")
        
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        df = df.set_index("timestamp")
        
        # Resample
        rule = f"{resolution_seconds}s"
        resampled = df.resample(rule).agg({
            price_col: ["first", "last", "max", "min"],
            volume_col: "sum" if volume_col in df.columns else "first",
        })
        
        # Flatten multi-level columns
        resampled.columns = ["open", "close", "high", "low", "volume"]
        resampled = resampled.dropna().reset_index()
        
        return resampled
    
    def _compute_features_for_resolution(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute log returns, volatility, and momentum for a price series.
        
        Returns:
            Tuple of (log_returns, volatility, momentum)
        """
        n = len(prices)
        
        # Log returns
        log_returns = np.zeros(n, dtype=np.float32)
        log_returns[1:] = np.log(prices[1:] / (prices[:-1] + 1e-10))
        
        # Clip extreme returns
        log_returns = np.clip(log_returns, -0.1, 0.1)
        
        # Rolling volatility
        volatility = np.zeros(n, dtype=np.float32)
        window = self.config.volatility_window
        for i in range(window, n):
            volatility[i] = np.std(log_returns[i-window:i]) * 100  # Scale up
        volatility[:window] = volatility[window] if window < n else 0
        
        # Momentum: EMA fast - EMA slow
        ema_fast = self._compute_ema(prices, self.config.ema_fast_period)
        ema_slow = self._compute_ema(prices, self.config.ema_slow_period)
        
        # Normalize momentum to percentage
        momentum = (ema_fast - ema_slow) / (ema_slow + 1e-10) * 100
        momentum = np.clip(momentum, -5, 5).astype(np.float32)
        
        return log_returns, volatility, momentum
    
    def _compute_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute exponential moving average."""
        ema = np.zeros_like(data, dtype=np.float64)
        alpha = 2.0 / (period + 1)
        
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def build_features(
        self,
        btc_data: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
        candle_end_timestamp: Optional[pd.Timestamp] = None,
        candle_minutes: int = 15,
    ) -> np.ndarray:
        """
        Build multi-resolution feature tensor.
        
        Args:
            btc_data: BTC price/volume data with 'timestamp', 'price', 'volume'
            dxy_data: Optional DXY data with 'timestamp', 'price'
            eurusd_data: Optional EUR/USD data with 'timestamp', 'price'
            candle_end_timestamp: End time of current candle (for time features)
            candle_minutes: Duration of candle in minutes
            
        Returns:
            Feature array of shape (sequence_length, feature_dim)
        """
        features_list = []
        
        # Process each resolution
        btc_features_all = []
        for res_seconds in self.config.target_resolutions_seconds:
            resampled = self._resample_to_resolution(
                btc_data, res_seconds, "price", "volume"
            )
            
            prices = resampled["close"].values
            volumes = resampled["volume"].values
            
            log_ret, vol, mom = self._compute_features_for_resolution(prices, volumes)
            
            # Take most recent value for each feature
            btc_features_all.extend([
                log_ret[-1] * 100,  # Scale log returns
                vol[-1],
                mom[-1],
            ])
        
        features_list.extend(btc_features_all)
        
        # Time features
        if candle_end_timestamp is not None:
            current_time = btc_data["timestamp"].iloc[-1]
            if not isinstance(current_time, pd.Timestamp):
                current_time = pd.Timestamp(current_time)
            if not isinstance(candle_end_timestamp, pd.Timestamp):
                candle_end_timestamp = pd.Timestamp(candle_end_timestamp)
            
            time_remaining = (candle_end_timestamp - current_time).total_seconds()
            time_remaining = max(0, time_remaining) / (candle_minutes * 60)
            
            hour = current_time.hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
        else:
            time_remaining = 0.5
            hour_sin = 0.0
            hour_cos = 1.0
        
        features_list.extend([time_remaining, hour_sin, hour_cos])
        
        # DXY features
        if dxy_data is not None and len(dxy_data) > 0:
            dxy_prices = dxy_data["price"].values
            dxy_ret, dxy_vol, dxy_mom = self._compute_features_for_resolution(
                dxy_prices, np.ones(len(dxy_prices))
            )
            features_list.extend([
                dxy_ret[-1] * 100 * self.config.forex_weight,
                dxy_mom[-1] * self.config.forex_weight,
                dxy_vol[-1] * self.config.forex_weight,
            ])
        else:
            features_list.extend([0.0, 0.0, 0.0])
        
        # EUR/USD features
        if eurusd_data is not None and len(eurusd_data) > 0:
            eurusd_prices = eurusd_data["price"].values
            eur_ret, eur_vol, eur_mom = self._compute_features_for_resolution(
                eurusd_prices, np.ones(len(eurusd_prices))
            )
            features_list.extend([
                eur_ret[-1] * 100 * self.config.forex_weight,
                eur_mom[-1] * self.config.forex_weight,
                eur_vol[-1] * self.config.forex_weight,
            ])
        else:
            features_list.extend([0.0, 0.0, 0.0])
        
        # Relative pressure features (BTC vs DXY)
        btc_return = btc_features_all[0] if len(btc_features_all) > 0 else 0
        dxy_return = features_list[15] if len(features_list) > 15 else 0  # After time features
        
        # Divergence: BTC up + DXY down = bullish confirmation
        divergence = btc_return - dxy_return
        
        # Sign disagreement: warning signal
        btc_sign = np.sign(btc_return)
        dxy_sign = np.sign(dxy_return) if dxy_data is not None else 0
        sign_disagree = float(btc_sign != -dxy_sign)  # They should be inverse
        
        # Correlation flag (simplified)
        correlation_flag = 1.0 if btc_sign == -dxy_sign else -1.0
        
        features_list.extend([
            np.clip(divergence, -5, 5),
            correlation_flag,
            sign_disagree,
        ])
        
        return np.array(features_list, dtype=np.float32)
    
    def build_sequence(
        self,
        btc_data: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
        sequence_length: int = 180,
        candle_end_timestamp: Optional[pd.Timestamp] = None,
        candle_minutes: int = 15,
    ) -> np.ndarray:
        """
        Build a sequence of features for LSTM input.
        
        Returns shape (sequence_length, feature_dim).
        """
        # For each timestep in the sequence, compute features
        # using data up to that point
        
        n = len(btc_data)
        if n < sequence_length:
            # Pad with zeros
            pad_length = sequence_length - n
            features_seq = np.zeros((sequence_length, self.feature_dim), dtype=np.float32)
            
            for i in range(n):
                end_idx = i + 1
                btc_slice = btc_data.iloc[:end_idx]
                
                dxy_slice = None
                if dxy_data is not None and len(dxy_data) > 0:
                    dxy_slice = dxy_data.iloc[:min(end_idx, len(dxy_data))]
                
                eurusd_slice = None
                if eurusd_data is not None and len(eurusd_data) > 0:
                    eurusd_slice = eurusd_data.iloc[:min(end_idx, len(eurusd_data))]
                
                features_seq[pad_length + i] = self.build_features(
                    btc_slice, dxy_slice, eurusd_slice,
                    candle_end_timestamp, candle_minutes
                )
            
            return features_seq
        
        # Full sequence
        features_seq = np.zeros((sequence_length, self.feature_dim), dtype=np.float32)
        start_idx = n - sequence_length
        
        for i in range(sequence_length):
            end_idx = start_idx + i + 1
            btc_slice = btc_data.iloc[:end_idx]
            
            dxy_slice = None
            if dxy_data is not None and len(dxy_data) > 0:
                dxy_end = min(end_idx, len(dxy_data))
                dxy_slice = dxy_data.iloc[:dxy_end]
            
            eurusd_slice = None
            if eurusd_data is not None and len(eurusd_data) > 0:
                eur_end = min(end_idx, len(eurusd_data))
                eurusd_slice = eurusd_data.iloc[:eur_end]
            
            features_seq[i] = self.build_features(
                btc_slice, dxy_slice, eurusd_slice,
                candle_end_timestamp, candle_minutes
            )
        
        return features_seq


def create_training_dataset(
    btc_data: pd.DataFrame,
    dxy_data: Optional[pd.DataFrame] = None,
    eurusd_data: Optional[pd.DataFrame] = None,
    candle_minutes: int = 15,
    sequence_length: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset from historical data.
    
    Returns:
        X: Feature sequences, shape (num_samples, sequence_length, feature_dim)
        y: Binary outcomes (1 = candle closed up, 0 = down)
    """
    builder = MultiResolutionFeatureBuilder()
    
    # Group by candles
    btc = btc_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(btc["timestamp"]):
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    
    candle_seconds = candle_minutes * 60
    btc["candle_idx"] = (btc["timestamp"].astype(np.int64) // 10**9 // candle_seconds).astype(int)
    
    X_list = []
    y_list = []
    
    for candle_id, group in btc.groupby("candle_idx"):
        if len(group) < 10:
            continue
        
        # Get data up to candle start
        candle_start_idx = group.index[0]
        if candle_start_idx < sequence_length:
            continue
        
        # Features: use data leading up to the candle
        feature_data = btc.iloc[candle_start_idx - sequence_length:candle_start_idx]
        
        # Outcome: did price go up or down?
        candle_open = group["price"].iloc[0]
        candle_close = group["price"].iloc[-1]
        outcome = 1 if candle_close > candle_open else 0
        
        # Build features
        candle_end = group["timestamp"].iloc[-1]
        features = builder.build_sequence(
            feature_data,
            dxy_data=dxy_data,
            eurusd_data=eurusd_data,
            sequence_length=sequence_length,
            candle_end_timestamp=candle_end,
            candle_minutes=candle_minutes,
        )
        
        X_list.append(features)
        y_list.append(outcome)
    
    logger.info(
        "Created training dataset",
        num_samples=len(X_list),
        feature_dim=builder.feature_dim,
        sequence_length=sequence_length,
    )
    
    return np.array(X_list), np.array(y_list)
