"""
Feature preprocessing and engineering for ML models.

Computes technical indicators and normalizes features for stable training:
- Technical indicators: RSI, MACD, Bollinger Bands, ATR
- Rolling statistics: volatility, momentum, mean reversion signals
- Time-based features: time to resolution, market age
- Normalization and standardization
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Lookback windows (in number of steps)
    short_window: int = 60  # 1 minute at 1s resolution
    medium_window: int = 300  # 5 minutes
    long_window: int = 900  # 15 minutes
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    
    # Normalization
    normalize_method: Literal["zscore", "minmax", "robust"] = "zscore"
    zscore_window: int = 3600  # 1 hour rolling window for z-score
    
    # Output sequence length for temporal models
    sequence_length: int = 60


class FeaturePreprocessor:
    """
    Preprocesses raw price data into features for ML models.
    
    Computes a rich set of features designed to capture:
    - Price dynamics at multiple time scales
    - Volatility and momentum signals
    - Mean reversion indicators
    - Time-aware features for prediction markets
    
    Example:
        preprocessor = FeaturePreprocessor()
        features = preprocessor.transform(df, resolution_at=resolution_time)
    """
    
    # Feature names for documentation and model input
    FEATURE_NAMES = [
        "price_normalized",
        "return_1m",
        "return_5m",
        "return_15m",
        "volatility_1m",
        "volatility_5m",
        "volatility_15m",
        "rsi",
        "macd_histogram",
        "bollinger_position",
        "momentum",
        "mean_reversion_signal",
        "time_to_resolution",
        "market_age",
        "volume_ratio",
    ]
    
    def __init__(self, config: FeatureConfig | None = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Feature configuration, uses defaults if not provided
        """
        self.config = config or FeatureConfig()
        
        # Statistics for normalization (computed during fit)
        self._price_mean: float | None = None
        self._price_std: float | None = None
        self._feature_stats: dict[str, tuple[float, float]] = {}
    
    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """
        Compute normalization statistics from training data.
        
        Args:
            df: DataFrame with 'timestamp', 'price', 'volume' columns
            
        Returns:
            self for chaining
        """
        # Price statistics
        self._price_mean = df["price"].mean()
        self._price_std = df["price"].std()
        
        # Compute features to get statistics
        features_df = self._compute_raw_features(df)
        
        # Store statistics for each feature
        for col in features_df.columns:
            if col not in ["timestamp"]:
                self._feature_stats[col] = (
                    features_df[col].mean(),
                    features_df[col].std() + 1e-8,  # Avoid division by zero
                )
        
        logger.info(
            "Fitted preprocessor",
            price_mean=self._price_mean,
            price_std=self._price_std,
            num_features=len(self._feature_stats),
        )
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Transform raw price data into normalized features.
        
        Args:
            df: DataFrame with 'timestamp', 'price', 'volume' columns
            resolution_at: Market resolution time (for time features)
            created_at: Market creation time (for market age feature)
            
        Returns:
            DataFrame with normalized features
        """
        # Compute raw features
        features_df = self._compute_raw_features(df, resolution_at, created_at)
        
        # Normalize features
        normalized_df = self._normalize_features(features_df)
        
        return normalized_df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df)
        return self.transform(df, resolution_at, created_at)
    
    def _compute_raw_features(
        self,
        df: pd.DataFrame,
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> pd.DataFrame:
        """Compute raw (unnormalized) features."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # =================================================================
        # Price-based features
        # =================================================================
        
        # Normalized price (0-1 for binary markets)
        df["price_normalized"] = df["price"].clip(0, 1)
        
        # Returns at different time scales
        df["return_1m"] = df["price"].pct_change(self.config.short_window).fillna(0)
        df["return_5m"] = df["price"].pct_change(self.config.medium_window).fillna(0)
        df["return_15m"] = df["price"].pct_change(self.config.long_window).fillna(0)
        
        # Volatility (rolling std of returns)
        log_returns = np.log(df["price"] / df["price"].shift(1)).fillna(0)
        df["volatility_1m"] = log_returns.rolling(self.config.short_window).std().fillna(0)
        df["volatility_5m"] = log_returns.rolling(self.config.medium_window).std().fillna(0)
        df["volatility_15m"] = log_returns.rolling(self.config.long_window).std().fillna(0)
        
        # =================================================================
        # Technical indicators
        # =================================================================
        
        # RSI
        df["rsi"] = self._compute_rsi(df["price"], self.config.rsi_period)
        
        # MACD
        df["macd_histogram"] = self._compute_macd(
            df["price"],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        
        # Bollinger Bands position (-1 to 1, where 0 is at middle band)
        df["bollinger_position"] = self._compute_bollinger_position(
            df["price"],
            self.config.bollinger_period,
            self.config.bollinger_std,
        )
        
        # =================================================================
        # Momentum and mean reversion
        # =================================================================
        
        # Momentum (price acceleration)
        df["momentum"] = df["return_1m"] - df["return_1m"].shift(self.config.short_window)
        df["momentum"] = df["momentum"].fillna(0)
        
        # Mean reversion signal (deviation from moving average)
        ma = df["price"].rolling(self.config.long_window).mean()
        std = df["price"].rolling(self.config.long_window).std()
        df["mean_reversion_signal"] = ((df["price"] - ma) / (std + 1e-8)).fillna(0)
        
        # =================================================================
        # Volume features
        # =================================================================
        
        # Volume ratio (current vs rolling average)
        if "volume" in df.columns and df["volume"].sum() > 0:
            avg_volume = df["volume"].rolling(self.config.long_window).mean()
            df["volume_ratio"] = (df["volume"] / (avg_volume + 1e-8)).fillna(1)
        else:
            df["volume_ratio"] = 1.0
        
        # =================================================================
        # Time-based features
        # =================================================================
        
        # Time to resolution (normalized 0-1, 1 = far from resolution)
        if resolution_at:
            resolution_ts = resolution_at.timestamp()
            current_ts = df["timestamp"].apply(lambda x: x.timestamp())
            time_remaining = (resolution_ts - current_ts).clip(lower=0)
            
            # Normalize by total market duration
            if created_at:
                total_duration = resolution_ts - created_at.timestamp()
                df["time_to_resolution"] = time_remaining / (total_duration + 1)
            else:
                # Use 30 days as default total duration
                df["time_to_resolution"] = time_remaining / (30 * 24 * 3600)
            
            df["time_to_resolution"] = df["time_to_resolution"].clip(0, 1)
        else:
            df["time_to_resolution"] = 0.5  # Default to middle
        
        # Market age (normalized 0-1)
        if created_at:
            created_ts = created_at.timestamp()
            current_ts = df["timestamp"].apply(lambda x: x.timestamp())
            age = current_ts - created_ts
            # Normalize by 30 days
            df["market_age"] = (age / (30 * 24 * 3600)).clip(0, 1)
        else:
            df["market_age"] = 0.5
        
        # Select only feature columns
        feature_cols = [
            "timestamp",
            "price_normalized",
            "return_1m",
            "return_5m",
            "return_15m",
            "volatility_1m",
            "volatility_5m",
            "volatility_15m",
            "rsi",
            "macd_histogram",
            "bollinger_position",
            "momentum",
            "mean_reversion_signal",
            "volume_ratio",
            "time_to_resolution",
            "market_age",
        ]
        
        return df[feature_cols]
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using configured method."""
        df = df.copy()
        
        for col in df.columns:
            if col == "timestamp":
                continue
            
            if self.config.normalize_method == "zscore":
                if col in self._feature_stats:
                    mean, std = self._feature_stats[col]
                else:
                    mean = df[col].mean()
                    std = df[col].std() + 1e-8
                
                df[col] = (df[col] - mean) / std
                
            elif self.config.normalize_method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                
            elif self.config.normalize_method == "robust":
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25) + 1e-8
                df[col] = (df[col] - median) / iqr
        
        # Clip extreme values
        for col in df.columns:
            if col != "timestamp":
                df[col] = df[col].clip(-10, 10)
        
        return df
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to 0-1
        return (rsi / 100).fillna(0.5)
    
    @staticmethod
    def _compute_macd(
        prices: pd.Series,
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> pd.Series:
        """Compute MACD histogram."""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        histogram = macd_line - signal_line
        
        return histogram.fillna(0)
    
    @staticmethod
    def _compute_bollinger_position(
        prices: pd.Series,
        period: int,
        num_std: float,
    ) -> pd.Series:
        """Compute position within Bollinger Bands (-1 to 1)."""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = ma + num_std * std
        lower = ma - num_std * std
        
        # Position relative to bands
        position = (prices - lower) / (upper - lower + 1e-8)
        
        # Convert to -1 to 1 range
        position = (position * 2 - 1).clip(-1, 1)
        
        return position.fillna(0)
    
    def get_sequence(
        self,
        features_df: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        """
        Extract a sequence of features for temporal models.
        
        Args:
            features_df: Preprocessed features DataFrame
            index: Current time index
            
        Returns:
            numpy array of shape (sequence_length, num_features)
        """
        seq_len = self.config.sequence_length
        
        # Handle edge case at start
        if index < seq_len:
            # Pad with first observation
            padding = seq_len - index
            sequence = features_df.iloc[:index + 1].drop(columns=["timestamp"]).values
            if len(sequence) > 0:
                pad_value = sequence[0]
                sequence = np.vstack([
                    np.tile(pad_value, (padding, 1)),
                    sequence,
                ])
            else:
                sequence = np.zeros((seq_len, len(self.FEATURE_NAMES)))
        else:
            sequence = features_df.iloc[index - seq_len + 1:index + 1].drop(
                columns=["timestamp"]
            ).values
        
        return sequence.astype(np.float32)
    
    def get_feature_dim(self) -> int:
        """Get the number of features (excluding timestamp)."""
        return len(self.FEATURE_NAMES)


def create_training_dataset(
    df: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    resolution_at: datetime | None = None,
    created_at: datetime | None = None,
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a training dataset from raw price data.
    
    Returns:
        Tuple of (sequences, prices) where:
        - sequences: (N, seq_len, features) array
        - prices: (N,) array of corresponding prices
    """
    features_df = preprocessor.fit_transform(df, resolution_at, created_at)
    
    sequences = []
    prices = []
    
    for i in range(sequence_length, len(features_df)):
        seq = preprocessor.get_sequence(features_df, i)
        sequences.append(seq)
        prices.append(df.iloc[i]["price"])
    
    return np.stack(sequences), np.array(prices)
