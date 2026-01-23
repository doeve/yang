"""
Token Price Dataset for Training Edge Detector and SAC.

Since high-frequency Polymarket token price data may be limited,
this module provides:
1. Processing of available historical token data
2. Simulation of realistic token price paths for training
3. Alignment with BTC price data for guidance features

The simulator generates training data that matches real market dynamics:
- 3-phase price evolution (mean-reversion, trending, convergence)
- Realistic spread and noise patterns
- Correlation with BTC movements
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TokenDataConfig:
    """Configuration for token dataset."""

    # Candle parameters
    candle_seconds: int = 15 * 60  # 15 minutes
    steps_per_candle: int = 180  # 5-second steps

    # Simulation parameters
    initial_price_range: Tuple[float, float] = (0.35, 0.65)
    price_noise: float = 0.015
    mean_reversion_strength: float = 0.05
    convergence_rate: float = 0.08

    # BTC correlation
    btc_correlation: float = 0.7  # How much BTC influences token price


class TokenPriceSimulator:
    """
    Simulate realistic token price paths for training.

    Uses historical BTC data and known outcomes to generate
    synthetic token price histories that match real market dynamics.
    """

    def __init__(self, config: Optional[TokenDataConfig] = None):
        self.config = config or TokenDataConfig()

    def simulate_candle(
        self,
        outcome: int,  # 0 = NO wins, 1 = YES wins
        btc_prices: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate YES/NO token prices for one candle.

        Args:
            outcome: Settlement outcome (0 = NO wins, 1 = YES wins)
            btc_prices: Optional BTC prices to correlate with
            seed: Random seed

        Returns:
            Tuple of (yes_prices, no_prices) arrays
        """
        rng = np.random.default_rng(seed)
        n_steps = self.config.steps_per_candle

        # Initial price
        initial_yes = rng.uniform(*self.config.initial_price_range)
        convergence_target = float(outcome)

        yes_prices = [initial_yes]

        for step in range(1, n_steps):
            time_frac = step / n_steps
            current_yes = yes_prices[-1]

            # Phase-dependent dynamics
            if time_frac < 0.3:
                # Phase 1: Mean reversion around 0.5
                drift = (0.5 - current_yes) * self.config.mean_reversion_strength
                noise_scale = self.config.price_noise * 1.2
            elif time_frac < 0.7:
                # Phase 2: Trending toward outcome
                trend_strength = 0.02 + (time_frac - 0.3) * 0.08
                drift = (convergence_target - current_yes) * trend_strength
                noise_scale = self.config.price_noise
            else:
                # Phase 3: Strong convergence
                convergence_strength = self.config.convergence_rate * (1 + (time_frac - 0.7) * 3)
                drift = (convergence_target - current_yes) * convergence_strength
                noise_scale = self.config.price_noise * 0.5

            # Add BTC influence if available
            if btc_prices is not None and step < len(btc_prices):
                btc_return = (btc_prices[step] - btc_prices[step-1]) / btc_prices[step-1]
                btc_influence = btc_return * self.config.btc_correlation * (1 if outcome == 1 else -1)
                drift += btc_influence

            # Random noise
            noise = rng.normal(0, noise_scale)

            # Update price
            new_yes = current_yes + drift + noise
            new_yes = np.clip(new_yes, 0.01, 0.99)
            yes_prices.append(new_yes)

        yes_prices = np.array(yes_prices)
        no_prices = 1.0 - yes_prices

        return yes_prices, no_prices

    def generate_training_data(
        self,
        n_candles: int = 1000,
        btc_data: Optional[pd.DataFrame] = None,
        balance_outcomes: bool = True,
    ) -> pd.DataFrame:
        """
        Generate training dataset with simulated token prices.

        Args:
            n_candles: Number of candles to generate
            btc_data: Optional BTC price data to align with
            balance_outcomes: Whether to balance YES/NO wins

        Returns:
            DataFrame with timestamp, yes_price, no_price, outcome columns
        """
        logger.info(f"Generating {n_candles} simulated candles...")

        all_data = []
        base_timestamp = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)

        for candle_idx in range(n_candles):
            # Balanced outcomes
            if balance_outcomes:
                outcome = candle_idx % 2
            else:
                outcome = np.random.choice([0, 1])

            # Get BTC prices for this candle if available
            btc_prices = None
            if btc_data is not None:
                candle_start = candle_idx * self.config.steps_per_candle
                candle_end = candle_start + self.config.steps_per_candle
                if candle_end <= len(btc_data):
                    btc_prices = btc_data['price'].values[candle_start:candle_end]

            # Simulate token prices
            yes_prices, no_prices = self.simulate_candle(
                outcome=outcome,
                btc_prices=btc_prices,
                seed=candle_idx,
            )

            # Create timestamps
            candle_start_time = base_timestamp + pd.Timedelta(seconds=candle_idx * self.config.candle_seconds)
            timestamps = [
                candle_start_time + pd.Timedelta(seconds=step * 5)  # 5-second steps
                for step in range(len(yes_prices))
            ]

            # Build records
            for i, (ts, yes_p, no_p) in enumerate(zip(timestamps, yes_prices, no_prices)):
                all_data.append({
                    'timestamp': ts,
                    'yes_price': yes_p,
                    'no_price': no_p,
                    'outcome': outcome,
                    'candle_idx': candle_idx,
                    'step_in_candle': i,
                    'time_remaining': 1.0 - i / len(yes_prices),
                })

        df = pd.DataFrame(all_data)

        logger.info(
            "Generated training data",
            n_candles=n_candles,
            n_rows=len(df),
            outcome_balance=f"YES={df.groupby('candle_idx')['outcome'].first().sum()}, NO={n_candles - df.groupby('candle_idx')['outcome'].first().sum()}",
        )

        return df


class TokenDatasetBuilder:
    """
    Build training datasets from historical or simulated data.
    """

    def __init__(self, config: Optional[TokenDataConfig] = None):
        self.config = config or TokenDataConfig()
        self.simulator = TokenPriceSimulator(config)

    def load_historical_data(
        self,
        token_path: str,
        btc_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load historical Polymarket and BTC data."""
        token_df = pd.read_parquet(token_path)
        btc_df = pd.read_parquet(btc_path) if btc_path else None

        logger.info(
            "Loaded historical data",
            token_rows=len(token_df),
            btc_rows=len(btc_df) if btc_df is not None else 0,
        )

        return token_df, btc_df

    def prepare_training_data(
        self,
        token_df: Optional[pd.DataFrame] = None,
        btc_df: Optional[pd.DataFrame] = None,
        n_simulated_candles: int = 2000,
        use_simulation: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare training data, optionally augmenting with simulation.

        Args:
            token_df: Historical token price data
            btc_df: Historical BTC data
            n_simulated_candles: Number of simulated candles to add
            use_simulation: Whether to use simulation (recommended due to limited real data)

        Returns:
            Combined DataFrame ready for training
        """
        all_data = []

        # Add historical data if available
        if token_df is not None and len(token_df) > 0:
            # Process historical data
            # (Would need proper processing based on actual data format)
            logger.info(f"Processing {len(token_df)} historical token records")
            all_data.append(token_df)

        # Generate simulated data
        if use_simulation:
            simulated = self.simulator.generate_training_data(
                n_candles=n_simulated_candles,
                btc_data=btc_df,
            )
            all_data.append(simulated)

        # Combine
        if len(all_data) == 0:
            raise ValueError("No data available for training")

        combined = pd.concat(all_data, ignore_index=True)

        logger.info(
            "Prepared training data",
            total_rows=len(combined),
            n_candles=combined['candle_idx'].nunique() if 'candle_idx' in combined.columns else 'N/A',
        )

        return combined

    def create_edge_detector_dataset(
        self,
        token_df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        samples_per_candle: int = 5,
        test_ratio: float = 0.2,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create dataset for training EdgeDetector model.

        Returns:
            Dictionary with 'train' and 'test' keys, each containing (X, y) tuples
        """
        from src.data.token_features import TokenFeatureBuilder

        feature_builder = TokenFeatureBuilder()

        X_list = []
        y_list = []

        candle_groups = token_df.groupby('candle_idx')

        for candle_idx, candle_df in candle_groups:
            if len(candle_df) < 30:
                continue

            outcome = candle_df['outcome'].iloc[0]

            # Sample multiple points per candle
            sample_indices = np.linspace(
                30, len(candle_df) - 1, samples_per_candle, dtype=int
            )

            for idx in sample_indices:
                time_remaining = candle_df.iloc[idx]['time_remaining']

                # Get price history up to this point
                yes_prices = candle_df['yes_price'].values[:idx+1]
                no_prices = candle_df['no_price'].values[:idx+1]

                # BTC data (if available)
                btc_prices = None
                btc_open = None
                if btc_df is not None and 'price' in btc_df.columns:
                    btc_prices = btc_df['price'].values[:min(idx+1, len(btc_df))]
                    btc_open = btc_prices[0] if len(btc_prices) > 0 else None

                # Compute features
                features = feature_builder.compute_features(
                    yes_prices=yes_prices,
                    no_prices=no_prices,
                    time_remaining=time_remaining,
                    btc_prices=btc_prices,
                    btc_open=btc_open,
                )

                X_list.append(features)
                y_list.append(outcome)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Time-based split (no lookahead)
        n = len(X)
        split_idx = int(n * (1 - test_ratio))

        logger.info(
            "Created edge detector dataset",
            total_samples=n,
            train_samples=split_idx,
            test_samples=n - split_idx,
            feature_dim=X.shape[1],
            class_balance=f"YES={y.sum()}, NO={len(y) - y.sum()}",
        )

        return {
            'train': (X[:split_idx], y[:split_idx]),
            'test': (X[split_idx:], y[split_idx:]),
        }


def create_training_dataset(
    n_candles: int = 3000,
    btc_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to create training dataset.

    Args:
        n_candles: Number of candles to simulate
        btc_data_path: Path to BTC price data (for correlation)
        output_path: Optional path to save dataset

    Returns:
        Training DataFrame
    """
    builder = TokenDatasetBuilder()

    # Load BTC data if available
    btc_df = None
    if btc_data_path:
        btc_df = pd.read_parquet(btc_data_path)
        if 'close' in btc_df.columns:
            btc_df['price'] = btc_df['close']

    # Generate training data
    df = builder.prepare_training_data(
        n_simulated_candles=n_candles,
        btc_df=btc_df,
        use_simulation=True,
    )

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info(f"Saved training data to {output_path}")

    return df
