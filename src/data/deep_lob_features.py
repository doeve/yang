"""
DeepLOB-style feature engineering based on proven academic research.

Implements:
- Order Flow Imbalance (OFI) - stationary features from order book changes
- Trade Flow Imbalance (TFI) - aggressor-based trade direction
- LOB Asymmetry - bid/ask depth imbalance
- 3-class labeling with threshold (Up/Down/Hold)

Based on:
- Kolm et al. 2021: "Deep order flow imbalance: Extracting alpha"
- Zhang et al. 2019: "DeepLOB: Deep CNNs for Limit Order Books"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DeepLOBConfig:
    """Configuration for DeepLOB feature engineering."""
    
    # Labeling
    prediction_horizon: int = 10  # Steps ahead to predict
    alpha_threshold: float = 0.001  # 0.1% threshold for Up/Down
    
    # Feature windows
    ofi_windows: list = None  # Order flow imbalance windows
    tfi_window: int = 60  # Trade flow imbalance window
    
    # Normalization
    norm_window: int = 3600  # 1 hour sliding window for z-score
    
    def __post_init__(self):
        if self.ofi_windows is None:
            self.ofi_windows = [10, 30, 60, 300]  # Multi-scale OFI


class DeepLOBFeatureBuilder:
    """
    Feature engineering based on DeepLOB / Deep Order Flow Imbalance research.
    
    Key insight: Stationary features (OFI, TFI) outperform raw prices.
    
    Features (per sample):
    1. Order Flow Imbalance at multiple scales (8)
    2. Trade Flow Imbalance (4)
    3. Buy Pressure features (4)
    4. Price momentum (4)
    5. Volatility features (4)
    6. LOB Asymmetry proxy (4)
    7. Volume profile (4)
    8. Time features (2)
    
    Total: 34 features
    """
    
    def __init__(self, config: Optional[DeepLOBConfig] = None):
        self.config = config or DeepLOBConfig()
        self.feature_dim = 34
        
        logger.info(
            "DeepLOBFeatureBuilder initialized",
            feature_dim=self.feature_dim,
            alpha=self.config.alpha_threshold,
        )
    
    def compute_ofi(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        buy_pressure: np.ndarray,
        windows: list[int],
    ) -> np.ndarray:
        """
        Compute Order Flow Imbalance at multiple windows.
        
        OFI = cumulative(buy_volume - sell_volume)
        
        This is stationary and predictive, unlike raw prices.
        """
        n = len(prices)
        ofi_features = np.zeros((n, len(windows) * 2), dtype=np.float32)
        
        # Volume delta (buy - sell based on buy_pressure)
        buy_vol = volumes * buy_pressure
        sell_vol = volumes * (1 - buy_pressure)
        volume_delta = buy_vol - sell_vol
        
        for i, w in enumerate(windows):
            # Cumulative OFI over window
            ofi = pd.Series(volume_delta).rolling(w, min_periods=1).sum().values
            ofi_normalized = ofi / (pd.Series(volumes).rolling(w, min_periods=1).sum().values + 1e-10)
            
            # OFI change (acceleration)
            ofi_change = np.gradient(ofi_normalized)
            
            ofi_features[:, i*2] = np.clip(ofi_normalized, -1, 1)
            ofi_features[:, i*2 + 1] = np.clip(ofi_change * 100, -1, 1)
        
        return ofi_features
    
    def compute_tfi_from_trades(
        self,
        trades_df: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        window_seconds: int = 60,
    ) -> np.ndarray:
        """
        Compute Trade Flow Imbalance from aggregated trades.
        
        TFI = (buy_volume - sell_volume) / total_volume
        
        Uses is_buyer_maker to determine aggressor side.
        """
        n = len(timestamps)
        tfi_features = np.zeros((n, 4), dtype=np.float32)
        
        if trades_df is None or len(trades_df) == 0:
            return tfi_features
        
        # Ensure timestamp is datetime
        if 'timestamp' in trades_df.columns:
            trades = trades_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
                trades['timestamp'] = pd.to_datetime(trades['timestamp'], utc=True)
            trades = trades.set_index('timestamp')
        else:
            return tfi_features
        
        # Resample to match main data
        for i, ts in enumerate(timestamps):
            window_start = ts - pd.Timedelta(seconds=window_seconds)
            window_trades = trades.loc[window_start:ts]
            
            if len(window_trades) == 0:
                continue
            
            # Seller-initiated trades (is_buyer_maker=True means seller aggressed)
            if 'is_buyer_maker' in window_trades.columns:
                sell_vol = window_trades[window_trades['is_buyer_maker'] == True]['quantity'].sum()
                buy_vol = window_trades[window_trades['is_buyer_maker'] == False]['quantity'].sum()
            else:
                buy_vol = sell_vol = window_trades['quantity'].sum() / 2
            
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                tfi_features[i, 0] = (buy_vol - sell_vol) / total_vol  # TFI ratio
                tfi_features[i, 1] = window_trades['quantity'].mean()  # Avg trade size
                tfi_features[i, 2] = len(window_trades)  # Trade count
                tfi_features[i, 3] = window_trades['quantity'].std() if len(window_trades) > 1 else 0
        
        # Normalize
        for j in range(4):
            if tfi_features[:, j].std() > 0:
                tfi_features[:, j] = (tfi_features[:, j] - tfi_features[:, j].mean()) / (tfi_features[:, j].std() + 1e-10)
        
        return np.clip(tfi_features, -3, 3)
    
    def compute_buy_pressure_features(
        self,
        buy_pressure: np.ndarray,
    ) -> np.ndarray:
        """
        Compute features from buy_pressure column (proxy for OFI).
        """
        n = len(buy_pressure)
        bp_features = np.zeros((n, 4), dtype=np.float32)
        
        # Raw centered buy pressure
        bp_features[:, 0] = buy_pressure - 0.5
        
        # Rolling averages
        bp_features[:, 1] = pd.Series(buy_pressure).rolling(60, min_periods=1).mean().values - 0.5
        bp_features[:, 2] = pd.Series(buy_pressure).rolling(300, min_periods=1).mean().values - 0.5
        
        # Buy pressure momentum
        bp_features[:, 3] = np.gradient(pd.Series(buy_pressure).rolling(30, min_periods=1).mean().values)
        
        return np.clip(bp_features, -1, 1)
    
    def compute_price_momentum(
        self,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute price momentum features (returns at multiple horizons).
        """
        n = len(prices)
        momentum = np.zeros((n, 4), dtype=np.float32)
        
        horizons = [10, 60, 300, 900]
        for i, h in enumerate(horizons):
            if h < n:
                ret = (prices[h:] - prices[:-h]) / (prices[:-h] + 1e-10)
                momentum[h:, i] = np.clip(ret * 100, -5, 5)  # Percentage
        
        return momentum
    
    def compute_volatility_features(
        self,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute volatility features at multiple windows.
        """
        n = len(prices)
        vol_features = np.zeros((n, 4), dtype=np.float32)
        
        # Log returns
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(prices[1:] / (prices[:-1] + 1e-10))
        
        windows = [30, 60, 300, 900]
        for i, w in enumerate(windows):
            vol_features[:, i] = pd.Series(log_ret).rolling(w, min_periods=1).std().values * 100
        
        return np.clip(vol_features, 0, 10)
    
    def compute_lob_asymmetry_proxy(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        buy_pressure: np.ndarray,
    ) -> np.ndarray:
        """
        Proxy for LOB asymmetry using available data.
        
        Without actual LOB data, we approximate using:
        - Volume-weighted price impact
        - Buy/sell volume ratio
        """
        n = len(prices)
        asym_features = np.zeros((n, 4), dtype=np.float32)
        
        # Volume imbalance (bid proxy)
        buy_vol = volumes * buy_pressure
        sell_vol = volumes * (1 - buy_pressure)
        
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)
        asym_features[:, 0] = pd.Series(imbalance).rolling(60, min_periods=1).mean().values
        
        # Price-volume correlation (depth proxy)
        price_change = np.abs(np.diff(prices, prepend=prices[0]))
        vol_impact = price_change / (volumes + 1e-10)
        asym_features[:, 1] = pd.Series(vol_impact).rolling(60, min_periods=1).mean().values
        
        # Volume relative to recent average (liquidity proxy)
        vol_ma = pd.Series(volumes).rolling(300, min_periods=1).mean().values
        asym_features[:, 2] = np.log1p(volumes / (vol_ma + 1e-10))
        
        # Spread proxy (high-low range when available)
        asym_features[:, 3] = np.gradient(price_change)
        
        return np.clip(asym_features, -3, 3)
    
    def compute_volume_profile(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """
        Volume profile features.
        """
        n = len(prices)
        vol_features = np.zeros((n, 4), dtype=np.float32)
        
        # Volume momentum
        vol_ma_short = pd.Series(volumes).rolling(30, min_periods=1).mean().values
        vol_ma_long = pd.Series(volumes).rolling(300, min_periods=1).mean().values
        vol_features[:, 0] = (vol_ma_short - vol_ma_long) / (vol_ma_long + 1e-10)
        
        # Volume spikes
        vol_std = pd.Series(volumes).rolling(300, min_periods=1).std().values
        vol_features[:, 1] = (volumes - vol_ma_long) / (vol_std + 1e-10)
        
        # VWAP distance
        pv = prices * volumes
        vwap = pd.Series(pv).rolling(300, min_periods=1).sum().values / (pd.Series(volumes).rolling(300, min_periods=1).sum().values + 1e-10)
        vol_features[:, 2] = (prices - vwap) / (vwap + 1e-10) * 100
        
        # Volume trend
        vol_features[:, 3] = np.gradient(vol_ma_short)
        
        return np.clip(vol_features, -5, 5)
    
    def compute_time_features(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Time-based features (cyclical encoding).
        """
        n = len(timestamps)
        time_features = np.zeros((n, 2), dtype=np.float32)
        
        # Use .dt accessor for pandas Series
        if hasattr(timestamps, 'dt'):
            hours = timestamps.dt.hour.values
        else:
            hours = timestamps.hour
            
        time_features[:, 0] = np.sin(2 * np.pi * hours / 24)
        time_features[:, 1] = np.cos(2 * np.pi * hours / 24)
        
        return time_features

    
    def precompute_all_features(
        self,
        btc_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Precompute all features for the dataset.
        
        Args:
            btc_data: Must have: timestamp, price, volume, buy_pressure
            trades_data: Optional aggregated trades with is_buyer_maker
            
        Returns:
            Feature array of shape (n_samples, 34)
        """
        n = len(btc_data)
        
        # Extract data
        prices = btc_data['price'].values.astype(np.float64)
        volumes = btc_data['volume'].values.astype(np.float64)
        buy_pressure = btc_data.get('buy_pressure', pd.Series(np.ones(n) * 0.5)).values.astype(np.float64)
        
        if not pd.api.types.is_datetime64_any_dtype(btc_data['timestamp']):
            timestamps = pd.to_datetime(btc_data['timestamp'], utc=True)
        else:
            timestamps = btc_data['timestamp']
        
        # Compute all feature groups
        logger.info("Computing OFI features...")
        ofi = self.compute_ofi(prices, volumes, buy_pressure, self.config.ofi_windows)
        
        logger.info("Computing buy pressure features...")
        bp = self.compute_buy_pressure_features(buy_pressure)
        
        logger.info("Computing price momentum...")
        momentum = self.compute_price_momentum(prices)
        
        logger.info("Computing volatility features...")
        volatility = self.compute_volatility_features(prices)
        
        logger.info("Computing LOB asymmetry proxy...")
        asymmetry = self.compute_lob_asymmetry_proxy(prices, volumes, buy_pressure)
        
        logger.info("Computing volume profile...")
        vol_profile = self.compute_volume_profile(prices, volumes)
        
        logger.info("Computing time features...")
        time_feats = self.compute_time_features(timestamps)
        
        # TFI from trades if available
        if trades_data is not None and len(trades_data) > 0:
            logger.info("Computing TFI from trades...")
            tfi = self.compute_tfi_from_trades(trades_data, timestamps)
        else:
            tfi = np.zeros((n, 4), dtype=np.float32)
        
        # Concatenate all features
        features = np.concatenate([
            ofi,          # 8 features
            tfi,          # 4 features
            bp,           # 4 features
            momentum,     # 4 features
            volatility,   # 4 features
            asymmetry,    # 4 features
            vol_profile,  # 4 features
            time_feats,   # 2 features
        ], axis=1)
        
        assert features.shape[1] == self.feature_dim, f"Expected {self.feature_dim} features, got {features.shape[1]}"
        
        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.astype(np.float32)
    
    def create_3class_labels(
        self,
        prices: np.ndarray,
        horizon: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create 3-class labels: Down (0), Hold (1), Up (2)

        Key insight: Don't predict tiny moves - use threshold α.
        """
        horizon = horizon or self.config.prediction_horizon
        alpha = alpha or self.config.alpha_threshold

        n = len(prices)
        labels = np.ones(n, dtype=np.int64)  # Default: Hold

        # Future returns
        future_returns = np.zeros(n)
        future_returns[:-horizon] = (prices[horizon:] - prices[:-horizon]) / (prices[:-horizon] + 1e-10)

        # Classify
        labels[future_returns > alpha] = 2   # Up
        labels[future_returns < -alpha] = 0  # Down
        # Everything else stays 1 (Hold)

        # Mark last `horizon` samples as invalid
        labels[-horizon:] = -1

        logger.info(
            "Created 3-class labels",
            up_pct=f"{(labels == 2).sum() / (labels >= 0).sum() * 100:.1f}%",
            hold_pct=f"{(labels == 1).sum() / (labels >= 0).sum() * 100:.1f}%",
            down_pct=f"{(labels == 0).sum() / (labels >= 0).sum() * 100:.1f}%",
        )

        return labels

    def create_candle_labels(
        self,
        btc_data: pd.DataFrame,
        candle_minutes: int = 15,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create candle-level labels based on candle close vs open.

        This is the CORRECT labeling for Polymarket 15-min candle prediction.
        Labels represent: "Will this candle close above or below its open?"

        Args:
            btc_data: DataFrame with timestamp, price columns
            candle_minutes: Candle duration (default 15 for Polymarket)
            alpha: Threshold for Up/Down classification (default 0.0003 = 0.03%)

        Returns:
            Tuple of (labels array aligned to btc_data, candle_indices)
        """
        alpha = alpha or 0.0003  # 0.03% threshold for 15-min candles (tighter than 10s)

        btc = btc_data.copy()

        # Convert timestamp to seconds for candle grouping
        if not pd.api.types.is_datetime64_any_dtype(btc['timestamp']):
            btc['timestamp'] = pd.to_datetime(btc['timestamp'], utc=True)

        btc['timestamp_seconds'] = btc['timestamp'].astype(np.int64) // 10**9
        btc['candle_idx'] = btc['timestamp_seconds'] // (candle_minutes * 60)

        # Get candle open and close prices
        candle_stats = btc.groupby('candle_idx').agg(
            open_price=('price', 'first'),
            close_price=('price', 'last'),
            first_idx=('price', lambda x: x.index[0]),
            last_idx=('price', lambda x: x.index[-1]),
        )

        # Calculate candle returns
        candle_stats['return'] = (
            (candle_stats['close_price'] - candle_stats['open_price'])
            / candle_stats['open_price']
        )

        # Classify candles
        candle_stats['label'] = 1  # Default: Hold
        candle_stats.loc[candle_stats['return'] > alpha, 'label'] = 2   # Up
        candle_stats.loc[candle_stats['return'] < -alpha, 'label'] = 0  # Down

        # Map candle labels back to each data point
        n = len(btc)
        labels = np.ones(n, dtype=np.int64) * -1  # Default invalid

        for candle_idx, row in candle_stats.iterrows():
            # All points in this candle get the candle's label
            mask = btc['candle_idx'] == candle_idx
            labels[mask] = int(row['label'])

        # Mark the last candle as invalid (incomplete)
        last_candle = btc['candle_idx'].max()
        labels[btc['candle_idx'] == last_candle] = -1

        valid_mask = labels >= 0
        logger.info(
            "Created candle-level labels",
            candle_minutes=candle_minutes,
            total_candles=len(candle_stats),
            up_pct=f"{(labels[valid_mask] == 2).sum() / valid_mask.sum() * 100:.1f}%",
            hold_pct=f"{(labels[valid_mask] == 1).sum() / valid_mask.sum() * 100:.1f}%",
            down_pct=f"{(labels[valid_mask] == 0).sum() / valid_mask.sum() * 100:.1f}%",
        )

        return labels, btc['candle_idx'].values


def create_training_dataset_v2(
    btc_data: pd.DataFrame,
    trades_data: Optional[pd.DataFrame] = None,
    candle_minutes: int = 15,
    sequence_length: int = 120,
    prediction_horizon: int = 10,
    alpha_threshold: float = 0.001,
    use_candle_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset with DeepLOB-style features and 3-class labels.

    Args:
        btc_data: BTC price data with timestamp, price, volume, buy_pressure
        trades_data: Optional aggregated trades data
        candle_minutes: Candle duration for grouping (default 15)
        sequence_length: Number of timesteps per sequence (default 120)
        prediction_horizon: Steps ahead for short-term labels (only used if use_candle_labels=False)
        alpha_threshold: Threshold for Up/Down classification
        use_candle_labels: If True, use candle close vs open labels (RECOMMENDED).
                          If False, use short-term return labels (legacy behavior).

    Returns:
        X: Feature sequences, shape (num_samples, sequence_length, 34)
        y: 3-class labels (0=Down, 1=Hold, 2=Up)
    """
    logger.info(
        "Creating DeepLOB-style training dataset...",
        use_candle_labels=use_candle_labels,
        candle_minutes=candle_minutes,
    )

    config = DeepLOBConfig(
        prediction_horizon=prediction_horizon,
        alpha_threshold=alpha_threshold,
    )
    builder = DeepLOBFeatureBuilder(config)

    # Ensure timestamp is datetime
    btc = btc_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(btc['timestamp']):
        btc['timestamp'] = pd.to_datetime(btc['timestamp'], utc=True)

    logger.info(f"Precomputing features for {len(btc):,} data points...")

    # Precompute features
    all_features = builder.precompute_all_features(btc, trades_data)

    # Create labels based on mode
    if use_candle_labels:
        # CANDLE-LEVEL LABELS: Predict candle close vs open
        # This is the correct semantic for Polymarket 15-min candle prediction
        candle_alpha = 0.0003  # 0.03% threshold for candle-level
        all_labels, candle_indices = builder.create_candle_labels(
            btc, candle_minutes=candle_minutes, alpha=candle_alpha
        )
    else:
        # LEGACY: Short-term return labels (10-second horizon)
        # Warning: These don't match Polymarket candle semantics
        logger.warning(
            "Using short-term labels - these may not match Polymarket candle semantics!"
        )
        prices = btc['price'].values
        all_labels = builder.create_3class_labels(prices, prediction_horizon, alpha_threshold)

    logger.info("Grouping by candles...")

    # Group by candles
    candle_seconds = candle_minutes * 60
    btc['candle_idx'] = (btc['timestamp'].astype(np.int64) // 10**9 // candle_seconds).astype(int)

    # Get candle start indices
    candle_groups = btc.groupby('candle_idx')
    candle_starts = candle_groups.apply(lambda x: x.index[0], include_groups=False).values

    logger.info(f"Found {len(candle_starts)} candles. Extracting sequences...")

    # Extract sequences
    X_list = []
    y_list = []

    min_idx = sequence_length + 100

    for start_idx in candle_starts:
        if start_idx < min_idx:
            continue

        seq_start = start_idx - sequence_length
        seq_end = start_idx

        if seq_start < 0 or seq_end > len(all_features):
            continue

        # Get label at prediction point
        label = all_labels[start_idx]
        if label < 0:  # Invalid (near end of data)
            continue

        features = all_features[seq_start:seq_end]

        if len(features) == sequence_length:
            X_list.append(features)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(
        "Created training dataset",
        num_samples=len(X),
        feature_dim=builder.feature_dim,
        sequence_length=sequence_length,
        use_candle_labels=use_candle_labels,
        class_distribution=f"Down:{(y==0).sum()}, Hold:{(y==1).sum()}, Up:{(y==2).sum()}",
    )

    return X, y
