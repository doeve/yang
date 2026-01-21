"""
Enhanced multi-resolution feature engineering for probability model.

V2: Optimized with proper financial features for crypto prediction.
Includes order flow, multi-timeframe momentum, volatility regimes, and FX correlation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedFeatureConfig:
    """Configuration for enhanced feature builder."""
    
    # Resolutions (in seconds)
    resolutions: list[int] = field(
        default_factory=lambda: [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h
    )
    
    # Lookback windows (in each resolution's units)
    momentum_windows: list[int] = field(
        default_factory=lambda: [5, 10, 20, 60]  # Short, medium, long, trend
    )
    
    # Volatility windows
    volatility_windows: list[int] = field(default_factory=lambda: [10, 20, 60])
    
    # Order flow settings
    large_trade_threshold: float = 0.01  # Top 1% of trades
    volume_spike_threshold: float = 2.0  # 2x average volume
    
    # Forex weight (they're secondary signals)
    forex_weight: float = 0.5


class EnhancedFeatureBuilder:
    """
    Enhanced feature engineering with predictive financial features.
    
    Feature groups:
    1. PRICE ACTION (12 features)
       - Multi-timeframe returns (1m, 5m, 15m, 1h)
       - Multi-timeframe momentum (EMA slopes)
       - Price vs VWAP
       
    2. VOLATILITY (8 features)
       - Rolling volatility at multiple windows
       - ATR ratios
       - Volatility acceleration
       - High-low range
       
    3. ORDER FLOW (10 features)
       - Buy pressure (buyer-initiated volume %)
       - Volume delta (buy - sell)
       - Trade imbalance
       - Large trade detection
       - Volume momentum
       
    4. MICROSTRUCTURE (6 features)
       - Price acceleration
       - Volume profile skew
       - VWAP distance
       - Price efficiency
       
    5. FX CORRELATION (6 features)
       - DXY momentum
       - EUR/USD momentum
       - BTC-DXY divergence
       - Correlation regime
       
    6. TIME CONTEXT (4 features)
       - Time remaining in candle
       - Hour of day (cyclic)
       - Day of week effect
       
    Total: 46 features
    """
    
    def __init__(self, config: Optional[EnhancedFeatureConfig] = None):
        self.config = config or EnhancedFeatureConfig()
        
        # Feature dimensions
        self.price_features = 12
        self.volatility_features = 8
        self.orderflow_features = 10
        self.microstructure_features = 6
        self.fx_features = 6
        self.time_features = 4
        
        self.feature_dim = (
            self.price_features +
            self.volatility_features +
            self.orderflow_features +
            self.microstructure_features +
            self.fx_features +
            self.time_features
        )
        
        logger.info(
            "EnhancedFeatureBuilder initialized",
            feature_dim=self.feature_dim,
        )
    
    def _compute_returns(self, prices: np.ndarray, windows: list[int]) -> np.ndarray:
        """Compute returns at multiple windows."""
        n = len(prices)
        returns = np.zeros((n, len(windows)), dtype=np.float32)
        
        for i, w in enumerate(windows):
            if w < n:
                returns[w:, i] = (prices[w:] - prices[:-w]) / (prices[:-w] + 1e-10)
        
        return np.clip(returns, -0.1, 0.1) * 100  # Percentage
    
    def _compute_ema(self, data: np.ndarray, span: int) -> np.ndarray:
        """Fast EMA computation."""
        return pd.Series(data).ewm(span=span, adjust=False).mean().values
    
    def _compute_volatility(self, returns: np.ndarray, windows: list[int]) -> np.ndarray:
        """Compute rolling volatility at multiple windows."""
        n = len(returns)
        vol = np.zeros((n, len(windows)), dtype=np.float32)
        
        for i, w in enumerate(windows):
            vol[:, i] = pd.Series(returns).rolling(w, min_periods=1).std().fillna(0).values
        
        return vol * 100  # Scale up
    
    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
        """Compute Average True Range."""
        n = len(close)
        tr = np.zeros(n, dtype=np.float32)
        
        # True Range
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        return pd.Series(tr).rolling(window, min_periods=1).mean().values
    
    def _compute_vwap(self, prices: np.ndarray, volumes: np.ndarray, window: int = 60) -> np.ndarray:
        """Compute VWAP."""
        pv = prices * volumes
        cum_pv = pd.Series(pv).rolling(window, min_periods=1).sum().values
        cum_v = pd.Series(volumes).rolling(window, min_periods=1).sum().values
        return cum_pv / (cum_v + 1e-10)
    
    def precompute_all_features(
        self,
        btc_data: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Precompute all features for the entire dataset.
        
        Args:
            btc_data: Must have columns: timestamp, price, volume, and optionally 
                      buy_pressure, high, low, open, close
            dxy_data: Optional DXY hourly data
            eurusd_data: Optional EUR/USD hourly data
            
        Returns:
            Feature array of shape (n_samples, feature_dim)
        """
        n = len(btc_data)
        features = np.zeros((n, self.feature_dim), dtype=np.float32)
        
        # Extract price data
        if 'close' in btc_data.columns:
            prices = btc_data['close'].values.astype(np.float64)
            high = btc_data.get('high', btc_data['close']).values.astype(np.float64)
            low = btc_data.get('low', btc_data['close']).values.astype(np.float64)
        else:
            prices = btc_data['price'].values.astype(np.float64)
            high = prices
            low = prices
        
        volumes = btc_data.get('volume', pd.Series(np.ones(n))).values.astype(np.float64)
        buy_pressure = btc_data.get('buy_pressure', pd.Series(np.ones(n) * 0.5)).values.astype(np.float64)
        
        col = 0
        
        # ===== 1. PRICE ACTION (12 features) =====
        # Multi-timeframe returns
        returns_1m = self._compute_returns(prices, [60])[:, 0]
        returns_5m = self._compute_returns(prices, [300])[:, 0]
        returns_15m = self._compute_returns(prices, [900])[:, 0]
        returns_1h = self._compute_returns(prices, [3600])[:, 0]
        
        features[:, col:col+4] = np.column_stack([returns_1m, returns_5m, returns_15m, returns_1h])
        col += 4
        
        # Momentum (EMA slope)
        ema_fast = self._compute_ema(prices, 10)
        ema_slow = self._compute_ema(prices, 50)
        ema_trend = self._compute_ema(prices, 200)
        
        momentum_short = (ema_fast - ema_slow) / (ema_slow + 1e-10) * 100
        momentum_long = (ema_slow - ema_trend) / (ema_trend + 1e-10) * 100
        momentum_acceleration = np.gradient(momentum_short)
        
        features[:, col:col+3] = np.column_stack([
            np.clip(momentum_short, -5, 5),
            np.clip(momentum_long, -5, 5),
            np.clip(momentum_acceleration, -1, 1),
        ])
        col += 3
        
        # Price vs VWAP
        vwap = self._compute_vwap(prices, volumes, 60)
        vwap_distance = (prices - vwap) / (vwap + 1e-10) * 100
        
        vwap_1h = self._compute_vwap(prices, volumes, 3600)
        vwap_distance_1h = (prices - vwap_1h) / (vwap_1h + 1e-10) * 100
        
        features[:, col:col+2] = np.column_stack([
            np.clip(vwap_distance, -5, 5),
            np.clip(vwap_distance_1h, -5, 5),
        ])
        col += 2
        
        # Price efficiency (trend strength)
        price_change = np.abs(prices - np.roll(prices, 60))
        price_path = pd.Series(np.abs(np.diff(prices, prepend=prices[0]))).rolling(60, min_periods=1).sum().values
        efficiency = price_change / (price_path + 1e-10)
        
        features[:, col:col+1] = np.clip(efficiency, 0, 2).reshape(-1, 1)
        col += 1
        
        # RSI-like feature
        returns = np.diff(prices, prepend=prices[0])
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean().values
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
        
        features[:, col:col+2] = np.column_stack([
            (rsi - 50) / 50,  # Normalize to [-1, 1]
            np.gradient(rsi) / 10,  # RSI momentum
        ])
        col += 2
        
        # ===== 2. VOLATILITY (8 features) =====
        log_returns = np.log(prices[1:] / prices[:-1])
        log_returns = np.concatenate([[0], log_returns])
        
        vol_10 = pd.Series(log_returns).rolling(10, min_periods=1).std().values * 100
        vol_60 = pd.Series(log_returns).rolling(60, min_periods=1).std().values * 100
        vol_300 = pd.Series(log_returns).rolling(300, min_periods=1).std().values * 100
        
        features[:, col:col+3] = np.column_stack([vol_10, vol_60, vol_300])
        col += 3
        
        # Volatility ratio (short vs long)
        vol_ratio = vol_10 / (vol_60 + 1e-10)
        vol_regime = np.where(vol_ratio > 1.5, 1, np.where(vol_ratio < 0.5, -1, 0))
        
        features[:, col:col+2] = np.column_stack([
            np.clip(vol_ratio, 0, 3),
            vol_regime,
        ])
        col += 2
        
        # ATR features
        atr_14 = self._compute_atr(high, low, prices, 14)
        atr_ratio = atr_14 / (prices + 1e-10) * 100  # ATR as % of price
        
        features[:, col:col+1] = np.clip(atr_ratio, 0, 5).reshape(-1, 1)
        col += 1
        
        # Range features
        range_60 = pd.Series(high).rolling(60, min_periods=1).max().values - pd.Series(low).rolling(60, min_periods=1).min().values
        range_pct = range_60 / (prices + 1e-10) * 100
        
        features[:, col:col+2] = np.column_stack([
            np.clip(range_pct, 0, 10),
            np.gradient(vol_60),  # Volatility acceleration
        ])
        col += 2
        
        # ===== 3. ORDER FLOW (10 features) =====
        # Buy pressure (direct from data or compute)
        bp = buy_pressure
        bp_ma = pd.Series(bp).rolling(60, min_periods=1).mean().values
        bp_deviation = bp - bp_ma
        
        features[:, col:col+3] = np.column_stack([
            bp - 0.5,  # Centered around 0
            bp_ma - 0.5,
            bp_deviation,
        ])
        col += 3
        
        # Volume delta
        buy_vol = volumes * bp
        sell_vol = volumes * (1 - bp)
        volume_delta = (buy_vol - sell_vol) / (volumes + 1e-10)
        volume_delta_ma = pd.Series(volume_delta).rolling(60, min_periods=1).mean().values
        
        features[:, col:col+2] = np.column_stack([
            volume_delta,
            volume_delta_ma,
        ])
        col += 2
        
        # Volume momentum
        vol_ma_short = pd.Series(volumes).rolling(10, min_periods=1).mean().values
        vol_ma_long = pd.Series(volumes).rolling(60, min_periods=1).mean().values
        volume_momentum = (vol_ma_short - vol_ma_long) / (vol_ma_long + 1e-10)
        
        features[:, col:col+1] = np.clip(volume_momentum, -3, 3).reshape(-1, 1)
        col += 1
        
        # Volume spikes
        vol_zscore = (volumes - vol_ma_long) / (pd.Series(volumes).rolling(60, min_periods=1).std().values + 1e-10)
        is_spike = (vol_zscore > 2).astype(np.float32)
        
        features[:, col:col+2] = np.column_stack([
            np.clip(vol_zscore, -3, 5),
            is_spike,
        ])
        col += 2
        
        # Large trade detection (approximated from volume)
        vol_pct = volumes / (pd.Series(volumes).rolling(1000, min_periods=1).quantile(0.99).values + 1e-10)
        is_large_trade = (vol_pct > 1).astype(np.float32)
        
        features[:, col:col+2] = np.column_stack([
            np.clip(vol_pct, 0, 3),
            is_large_trade,
        ])
        col += 2
        
        # ===== 4. MICROSTRUCTURE (6 features) =====
        # Price acceleration
        price_vel = np.gradient(prices)
        price_acc = np.gradient(price_vel)
        
        features[:, col:col+2] = np.column_stack([
            np.clip(price_vel / (prices + 1e-10) * 10000, -10, 10),
            np.clip(price_acc / (prices + 1e-10) * 10000, -10, 10),
        ])
        col += 2
        
        # Price-volume correlation
        pv_corr = pd.Series(price_vel).rolling(60, min_periods=10).corr(pd.Series(volumes)).fillna(0).values
        
        features[:, col:col+1] = np.clip(pv_corr, -1, 1).reshape(-1, 1)
        col += 1
        
        # Trend consistency
        returns_sign = np.sign(np.diff(prices, prepend=prices[0]))
        trend_consistency = pd.Series(returns_sign).rolling(20, min_periods=1).mean().values
        
        features[:, col:col+1] = trend_consistency.reshape(-1, 1)
        col += 1
        
        # Mean reversion signal
        price_zscore = (prices - pd.Series(prices).rolling(60, min_periods=1).mean().values) / (pd.Series(prices).rolling(60, min_periods=1).std().values + 1e-10)
        
        features[:, col:col+2] = np.column_stack([
            np.clip(price_zscore, -3, 3),
            np.gradient(price_zscore),
        ])
        col += 2
        
        # ===== 5. FX CORRELATION (6 features) =====
        if dxy_data is not None and len(dxy_data) > 0:
            dxy_prices = dxy_data['close'].values if 'close' in dxy_data.columns else dxy_data['price'].values
            dxy_returns = np.diff(dxy_prices, prepend=dxy_prices[0]) / (dxy_prices + 1e-10)
            dxy_momentum = self._compute_ema(dxy_prices, 10) - self._compute_ema(dxy_prices, 50)
            dxy_momentum = dxy_momentum / (dxy_prices + 1e-10) * 100
            
            # Use last value for now (hourly data)
            features[:, col] = dxy_returns[-1] * 100 * self.config.forex_weight
            features[:, col+1] = np.clip(dxy_momentum[-1], -5, 5) * self.config.forex_weight
        col += 2
        
        if eurusd_data is not None and len(eurusd_data) > 0:
            eur_prices = eurusd_data['close'].values if 'close' in eurusd_data.columns else eurusd_data['price'].values
            eur_returns = np.diff(eur_prices, prepend=eur_prices[0]) / (eur_prices + 1e-10)
            eur_momentum = self._compute_ema(eur_prices, 10) - self._compute_ema(eur_prices, 50)
            eur_momentum = eur_momentum / (eur_prices + 1e-10) * 100
            
            features[:, col] = eur_returns[-1] * 100 * self.config.forex_weight
            features[:, col+1] = np.clip(eur_momentum[-1], -5, 5) * self.config.forex_weight
        col += 2
        
        # BTC-FX divergence
        btc_return = returns_1m
        dxy_ret = features[:, col-4] if dxy_data is not None else np.zeros(n)
        divergence = btc_return - dxy_ret
        correlation_sign = np.sign(btc_return) * np.sign(-dxy_ret)  # Should be positive if inverse corr
        
        features[:, col:col+2] = np.column_stack([
            np.clip(divergence, -5, 5),
            correlation_sign,
        ])
        col += 2
        
        # ===== 6. TIME CONTEXT (4 features) =====
        if 'timestamp' in btc_data.columns:
            ts = pd.to_datetime(btc_data['timestamp'])
            hours = ts.dt.hour.values
            dow = ts.dt.dayofweek.values
            
            # Cyclical encoding
            features[:, col] = np.sin(2 * np.pi * hours / 24)
            features[:, col+1] = np.cos(2 * np.pi * hours / 24)
            features[:, col+2] = np.sin(2 * np.pi * dow / 7)
            features[:, col+3] = 0.5  # Time remaining placeholder
        col += 4
        
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features


def create_training_dataset(
    btc_data: pd.DataFrame,
    dxy_data: Optional[pd.DataFrame] = None,
    eurusd_data: Optional[pd.DataFrame] = None,
    candle_minutes: int = 15,
    sequence_length: int = 120,  # 2 hours of 1-minute features
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset with enhanced features.
    
    Returns:
        X: Feature sequences, shape (num_samples, sequence_length, feature_dim)
        y: Binary outcomes (1 = candle closed up, 0 = down)
    """
    logger.info("Creating training dataset with enhanced features...")
    
    builder = EnhancedFeatureBuilder()
    
    # Ensure timestamps
    btc = btc_data.copy()
    if 'timestamp' in btc.columns and not pd.api.types.is_datetime64_any_dtype(btc['timestamp']):
        btc['timestamp'] = pd.to_datetime(btc['timestamp'], utc=True)
    
    logger.info(f"Precomputing features for {len(btc):,} data points...")
    
    # Precompute all features
    all_features = builder.precompute_all_features(btc, dxy_data, eurusd_data)
    
    logger.info("Features precomputed. Creating candles...")
    
    # Group by candles
    candle_seconds = candle_minutes * 60
    btc['candle_idx'] = (btc['timestamp'].astype(np.int64) // 10**9 // candle_seconds).astype(int)
    
    # Get candle info
    candle_groups = btc.groupby('candle_idx')
    
    # Use close if available, else price
    price_col = 'close' if 'close' in btc.columns else 'price'
    
    candle_info = candle_groups.agg({
        price_col: ['first', 'last'],
    }).reset_index()
    candle_info.columns = ['candle_idx', 'open', 'close']
    candle_info['outcome'] = (candle_info['close'] > candle_info['open']).astype(int)
    
    # Get start indices
    candle_starts = candle_groups.apply(lambda x: x.index[0], include_groups=False).values
    
    logger.info(f"Found {len(candle_info)} candles. Extracting sequences...")
    
    # Extract sequences
    X_list = []
    y_list = []
    
    min_idx = sequence_length + 100
    
    for start_idx, row in zip(candle_starts, candle_info.itertuples()):
        if start_idx < min_idx:
            continue
        
        seq_start = start_idx - sequence_length
        seq_end = start_idx
        
        if seq_start < 0 or seq_end > len(all_features):
            continue
        
        features = all_features[seq_start:seq_end]
        
        if len(features) == sequence_length:
            X_list.append(features)
            y_list.append(row.outcome)
    
    logger.info(
        "Created training dataset",
        num_samples=len(X_list),
        feature_dim=builder.feature_dim,
        sequence_length=sequence_length,
    )
    
    if len(X_list) == 0:
        raise ValueError("No valid training samples. Check data length.")
    
    return np.array(X_list), np.array(y_list)
