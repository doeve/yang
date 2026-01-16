"""
Pure ML preprocessor for tick-level data.

MATCHES D3V FORMAT EXACTLY:
- Raw normalized prices (no computed indicators)
- Volume/order flow data
- Time-based features only

This replaces the feature-engineered preprocessor for pure ML training.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PureMLConfig:
    """Configuration for pure ML preprocessing.
    
    Matches d3v binance-client.ts:
    - historyLength = 300 (30 seconds at 100ms)
    - tradeWindowMs = 60_000 (1 minute order flow)
    """
    
    # Sequence length (matches d3v historyLength)
    sequence_length: int = 300
    
    # Normalization
    normalize_prices: bool = True
    normalize_method: Literal["pct_change", "zscore", "minmax"] = "pct_change"
    
    # Resampling (aggregate ticks to fixed intervals)
    resample_interval_ms: int = 100  # 100ms intervals like d3v


class PureMLPreprocessor:
    """
    Pure ML preprocessor for tick-level training data.
    
    EXACTLY matches what d3v binance-client.ts provides:
    - Raw price history (300 points)
    - Trade-based volume/order flow signals
    - NO computed indicators (RSI, MACD, etc.)
    
    The neural network learns patterns from raw data.
    """
    
    def __init__(self, config: PureMLConfig | None = None):
        self.config = config or PureMLConfig()
        self._fitted = False
        self._price_stats: tuple[float, float] | None = None
    
    def resample_ticks_to_intervals(
        self,
        df: pd.DataFrame,
        interval_ms: int | None = None,
    ) -> pd.DataFrame:
        """
        Resample tick data to fixed intervals.
        
        Matches d3v's effective 100ms granularity by:
        - Taking last price within each interval
        - Summing volume within each interval
        - Calculating buy pressure per interval
        
        Args:
            df: Tick data with timestamp, price, quantity, is_buyer_maker
            interval_ms: Interval in milliseconds (default: config value)
            
        Returns:
            Resampled DataFrame with regular intervals
        """
        interval_ms = interval_ms or self.config.resample_interval_ms
        
        df = df.copy()
        df = df.sort_values("timestamp")
        
        # Set timestamp as index for resampling
        df = df.set_index("timestamp")
        
        # Resample to fixed intervals
        resampled = df.resample(f"{interval_ms}ms").agg({
            "price": "last",  # Last price in interval
            "quantity": "sum",  # Total volume
        }).dropna()
        
        # Calculate buy pressure if we have is_buyer_maker
        if "is_buyer_maker" in df.columns:
            # Group by interval and calculate buy volume ratio
            df["buy_value"] = df["price"] * df["quantity"] * (~df["is_buyer_maker"]).astype(float)
            df["sell_value"] = df["price"] * df["quantity"] * df["is_buyer_maker"].astype(float)
            
            buy_volume = df["buy_value"].resample(f"{interval_ms}ms").sum()
            sell_volume = df["sell_value"].resample(f"{interval_ms}ms").sum()
            
            total = buy_volume + sell_volume
            resampled["buy_pressure"] = (buy_volume / (total + 1e-8)).fillna(0.5)
        else:
            resampled["buy_pressure"] = 0.5
        
        resampled = resampled.reset_index()
        resampled = resampled.rename(columns={"quantity": "volume"})
        
        logger.info(
            "Resampled ticks to intervals",
            original_ticks=len(df),
            resampled_points=len(resampled),
            interval_ms=interval_ms,
        )
        
        return resampled
    
    def fit(self, df: pd.DataFrame) -> "PureMLPreprocessor":
        """
        Compute normalization statistics from training data.
        
        Args:
            df: DataFrame with 'price' column
        """
        self._price_stats = (df["price"].mean(), df["price"].std() + 1e-8)
        self._fitted = True
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Transform raw data to PURE ML format.
        
        Output columns:
        - timestamp
        - price_normalized: Raw price (normalized)
        - volume_normalized: Volume (normalized)
        - buy_pressure: [0, 1] ratio from order flow
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # Normalize price (% change from first)
        if self.config.normalize_method == "pct_change":
            first_price = df["price"].iloc[0]
            df["price_normalized"] = ((df["price"] / first_price) - 1.0) * 100
        elif self.config.normalize_method == "zscore":
            if self._price_stats:
                mean, std = self._price_stats
            else:
                mean, std = df["price"].mean(), df["price"].std() + 1e-8
            df["price_normalized"] = (df["price"] - mean) / std
        else:  # minmax
            min_p, max_p = df["price"].min(), df["price"].max()
            df["price_normalized"] = (df["price"] - min_p) / (max_p - min_p + 1e-8)
        
        # Normalize volume
        if "volume" in df.columns and df["volume"].sum() > 0:
            max_vol = df["volume"].rolling(min(100, len(df))).max().fillna(df["volume"].max())
            df["volume_normalized"] = df["volume"] / (max_vol + 1e-8)
        else:
            df["volume_normalized"] = 0.0
        
        # Buy pressure (if available)
        if "buy_pressure" not in df.columns:
            df["buy_pressure"] = 0.5
        
        # Time features
        if resolution_at:
            resolution_ts = resolution_at.timestamp()
            current_ts = df["timestamp"].apply(lambda x: x.timestamp())
            time_remaining = (resolution_ts - current_ts).clip(lower=0)
            
            if created_at:
                total_duration = resolution_ts - created_at.timestamp()
                df["time_to_resolution"] = time_remaining / (total_duration + 1)
            else:
                df["time_to_resolution"] = time_remaining / (15 * 60)  # 15 min default
        else:
            df["time_to_resolution"] = 0.5
        
        # Clip normalized values
        df["price_normalized"] = df["price_normalized"].clip(-10, 10)
        df["volume_normalized"] = df["volume_normalized"].clip(0, 10)
        
        return df[["timestamp", "price_normalized", "volume_normalized", "buy_pressure", "time_to_resolution"]]
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df)
        return self.transform(df, resolution_at, created_at)
    
    def get_sequence(
        self,
        features_df: pd.DataFrame,
        index: int,
    ) -> np.ndarray:
        """
        Extract a sequence for the model.
        
        MATCHES D3V FORMAT:
        - 300 points of [price, volume/order_flow] 
        
        Args:
            features_df: Preprocessed features DataFrame
            index: Current time index
            
        Returns:
            numpy array of shape (sequence_length, 2)
            - Column 0: normalized prices
            - Column 1: volume/order flow signal
        """
        seq_len = self.config.sequence_length
        
        # Select columns (price + volume/order flow)
        data_cols = ["price_normalized", "volume_normalized"]
        if "buy_pressure" in features_df.columns:
            data_cols = ["price_normalized", "buy_pressure"]
        
        # Handle edge case at start
        if index < seq_len:
            padding = seq_len - index
            sequence = features_df.iloc[:index + 1][data_cols].values
            if len(sequence) > 0:
                pad_value = sequence[0]
                sequence = np.vstack([
                    np.tile(pad_value, (padding, 1)),
                    sequence,
                ])
            else:
                sequence = np.zeros((seq_len, len(data_cols)))
        else:
            sequence = features_df.iloc[index - seq_len + 1:index + 1][data_cols].values
        
        return sequence.astype(np.float32)
    
    def get_feature_dim(self) -> int:
        """Get the number of features per timestep."""
        return 2  # price + volume/order_flow (PURE ML - just raw data)


def convert_aggtrades_to_training_format(
    df: pd.DataFrame,
    resample_ms: int = 100,
) -> pd.DataFrame:
    """
    Convert aggtrades DataFrame to d3v-compatible training format.
    
    Args:
        df: DataFrame from BinanceCollector.fetch_aggtrades_historical()
            Columns: timestamp, price, quantity, is_buyer_maker, trade_id
        resample_ms: Interval to resample to (default 100ms like d3v)
        
    Returns:
        DataFrame with columns: timestamp, price, volume, buy_pressure
    """
    preprocessor = PureMLPreprocessor(
        PureMLConfig(resample_interval_ms=resample_ms)
    )
    
    return preprocessor.resample_ticks_to_intervals(df, resample_ms)
