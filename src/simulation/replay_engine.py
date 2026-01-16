"""
Historical data replay engine.

Replays market data at 1-second granularity for realistic simulation:
- Strict temporal ordering (no lookahead bias)
- Configurable simulation speed
- Checkpoint/resume for long simulations
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd
import structlog

from ..data.preprocessor import FeatureConfig, FeaturePreprocessor

logger = structlog.get_logger(__name__)


@dataclass
class MarketState:
    """
    Current state of the market at a point in time.
    
    This represents all information available to the agent
    at a given moment - no future data is included.
    """
    timestamp: datetime
    price: float
    volume: float
    
    # Derived features (from preprocessor)
    features: np.ndarray  # (num_features,) array
    feature_sequence: np.ndarray  # (seq_len, num_features) for temporal models
    
    # Market metadata
    market_id: str = ""
    time_to_resolution: float = 1.0  # 0-1, 0 = resolved
    is_final: bool = False  # True if this is the last step
    
    # For resolved markets
    outcome: bool | None = None


@dataclass
class ReplayConfig:
    """Configuration for the replay engine."""
    
    # Time settings
    step_size_seconds: int = 1  # Simulation step size
    speed_multiplier: float = 1.0  # 1.0 = real-time, 100.0 = 100x speed
    
    # Feature settings
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Execution settings
    execution_delay_seconds: int = 1  # Delay between decision and execution
    
    # Episode settings
    max_steps: int | None = None  # Maximum steps per episode (None = until end)
    random_start: bool = True  # Start from random position in data


class ReplayEngine:
    """
    Replays historical market data for simulation.
    
    Key features:
    - 1-second granularity replay (interpolated if needed)
    - No lookahead bias: only past/current data visible
    - Configurable simulation speed
    - Checkpoint/resume support
    
    Example:
        engine = ReplayEngine(df, config)
        engine.reset()
        
        while not engine.done:
            state = engine.get_state()
            # Agent makes decision based on state
            engine.step()
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        config: ReplayConfig | None = None,
        market_id: str = "",
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
        outcome: bool | None = None,
    ):
        """
        Initialize replay engine.
        
        Args:
            price_data: DataFrame with 'timestamp', 'price', 'volume' columns
            config: Replay configuration
            market_id: Market identifier
            resolution_at: When the market resolves
            created_at: When the market was created
            outcome: True/False outcome for resolved markets
        """
        self.config = config or ReplayConfig()
        self.market_id = market_id
        self.resolution_at = resolution_at
        self.created_at = created_at
        self.outcome = outcome
        
        # Store raw data
        self._raw_data = price_data.copy()
        self._raw_data = self._raw_data.sort_values("timestamp").reset_index(drop=True)
        
        # Initialize preprocessor and compute features
        self.preprocessor = FeaturePreprocessor(self.config.feature_config)
        self._features_df = self.preprocessor.fit_transform(
            self._raw_data,
            resolution_at=resolution_at,
            created_at=created_at,
        )
        
        # Convert to numpy for fast indexing
        self._timestamps = self._raw_data["timestamp"].values
        self._prices = self._raw_data["price"].values.astype(np.float64)
        self._volumes = self._raw_data["volume"].values.astype(np.float64)
        self._features = self._features_df.drop(columns=["timestamp"]).values.astype(np.float32)
        
        # State
        self._current_step: int = 0
        self._start_step: int = 0
        self._max_step: int = len(self._prices) - 1
        self.done: bool = False
        
        # Checkpoint support
        self._checkpoint: dict | None = None
        
        logger.info(
            "ReplayEngine initialized",
            market_id=market_id,
            data_points=len(self._prices),
            start_time=self._timestamps[0],
            end_time=self._timestamps[-1],
        )
    
    @property
    def current_step(self) -> int:
        """Get current step index."""
        return self._current_step
    
    @property
    def total_steps(self) -> int:
        """Get total number of steps in the data."""
        return len(self._prices)
    
    @property
    def progress(self) -> float:
        """Get progress through the episode (0 to 1)."""
        steps_taken = self._current_step - self._start_step
        total_episode_steps = self._max_step - self._start_step
        if total_episode_steps == 0:
            return 1.0
        return steps_taken / total_episode_steps
    
    def reset(
        self,
        start_step: int | None = None,
        seed: int | None = None,
    ) -> MarketState:
        """
        Reset the replay to a starting position.
        
        Args:
            start_step: Specific step to start from (None = use config)
            seed: Random seed for reproducible starting positions
            
        Returns:
            Initial MarketState
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Determine start position
        seq_len = self.config.feature_config.sequence_length
        
        if start_step is not None:
            self._start_step = max(seq_len, min(start_step, self._max_step - 100))
        elif self.config.random_start:
            # Random start, ensuring enough data for sequences
            max_start = self._max_step - 100  # At least 100 steps to play
            if max_start > seq_len:
                self._start_step = np.random.randint(seq_len, max_start)
            else:
                self._start_step = seq_len
        else:
            self._start_step = seq_len
        
        self._current_step = self._start_step
        
        # Determine end position
        if self.config.max_steps:
            self._max_step = min(
                self._start_step + self.config.max_steps,
                len(self._prices) - 1,
            )
        else:
            self._max_step = len(self._prices) - 1
        
        self.done = False
        self._checkpoint = None
        
        logger.debug(
            "Replay reset",
            start_step=self._start_step,
            max_step=self._max_step,
        )
        
        return self.get_state()
    
    def step(self) -> MarketState:
        """
        Advance simulation by one step.
        
        Returns:
            New MarketState after the step
        """
        if self.done:
            raise RuntimeError("Cannot step after done. Call reset() first.")
        
        self._current_step += self.config.step_size_seconds
        
        # Check if we've reached the end
        if self._current_step >= self._max_step:
            self._current_step = self._max_step
            self.done = True
        
        return self.get_state()
    
    def get_state(self) -> MarketState:
        """
        Get current market state.
        
        Returns:
            MarketState with current price, features, and metadata
        """
        idx = self._current_step
        
        # Get current values
        timestamp = pd.Timestamp(self._timestamps[idx]).to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        price = self._prices[idx]
        volume = self._volumes[idx]
        features = self._features[idx]
        
        # Get feature sequence using PURE NUMPY (no pandas in hot path)
        seq_len = self.config.feature_config.sequence_length
        if idx < seq_len:
            # Pad with first values
            padding = seq_len - idx
            if idx > 0:
                feature_sequence = np.vstack([
                    np.tile(self._features[0], (padding, 1)),
                    self._features[:idx]
                ])
            else:
                feature_sequence = np.tile(self._features[0], (seq_len, 1))
        else:
            feature_sequence = self._features[idx - seq_len:idx]
        
        feature_sequence = feature_sequence.astype(np.float32)
        
        # Compute time to resolution
        if self.resolution_at:
            time_remaining = (self.resolution_at - timestamp).total_seconds()
            if self.created_at:
                total_duration = (self.resolution_at - self.created_at).total_seconds()
                time_to_resolution = max(0, time_remaining) / (total_duration + 1)
            else:
                time_to_resolution = max(0, time_remaining) / (30 * 24 * 3600)
        else:
            time_to_resolution = 0.5
        
        return MarketState(
            timestamp=timestamp,
            price=price,
            volume=volume,
            features=features,
            feature_sequence=feature_sequence,
            market_id=self.market_id,
            time_to_resolution=min(1.0, max(0.0, time_to_resolution)),
            is_final=self.done,
            outcome=self.outcome if self.done else None,
        )
    
    def peek_future(self, steps_ahead: int) -> float | None:
        """
        Peek at future price (FOR EVALUATION ONLY - not available during training).
        
        Args:
            steps_ahead: Number of steps to look ahead
            
        Returns:
            Future price, or None if beyond available data
        """
        future_idx = self._current_step + steps_ahead
        if future_idx >= len(self._prices):
            return None
        return self._prices[future_idx]
    
    def save_checkpoint(self) -> dict:
        """Save current state for later resume."""
        self._checkpoint = {
            "current_step": self._current_step,
            "start_step": self._start_step,
            "max_step": self._max_step,
            "done": self.done,
        }
        return self._checkpoint
    
    def load_checkpoint(self, checkpoint: dict) -> None:
        """Resume from a saved checkpoint."""
        self._current_step = checkpoint["current_step"]
        self._start_step = checkpoint["start_step"]
        self._max_step = checkpoint["max_step"]
        self.done = checkpoint["done"]
    
    def iterate(self) -> Iterator[MarketState]:
        """
        Iterate through all states from current position to end.
        
        Yields:
            MarketState at each step
        """
        while not self.done:
            yield self.get_state()
            if not self.done:
                self.step()
        
        # Yield final state
        yield self.get_state()


class MultiMarketReplayEngine:
    """
    Manages replay across multiple markets for diverse training.
    
    Supports:
    - Random market selection for each episode
    - Weighted sampling (e.g., prefer resolved markets)
    - Market filtering by criteria
    """
    
    def __init__(
        self,
        market_data: dict[str, pd.DataFrame],
        config: ReplayConfig | None = None,
        market_metadata: dict[str, dict] | None = None,
    ):
        """
        Initialize multi-market replay.
        
        Args:
            market_data: Dict mapping market_id to price DataFrames
            config: Replay configuration
            market_metadata: Optional metadata for each market
        """
        self.config = config or ReplayConfig()
        self.market_metadata = market_metadata or {}
        
        # Create engines for each market
        self.engines: dict[str, ReplayEngine] = {}
        
        for market_id, df in market_data.items():
            metadata = self.market_metadata.get(market_id, {})
            
            self.engines[market_id] = ReplayEngine(
                df,
                config=self.config,
                market_id=market_id,
                resolution_at=metadata.get("resolution_at"),
                created_at=metadata.get("created_at"),
                outcome=metadata.get("outcome"),
            )
        
        self.market_ids = list(self.engines.keys())
        self._current_engine: ReplayEngine | None = None
        
        logger.info("MultiMarketReplayEngine initialized", num_markets=len(self.engines))
    
    @property
    def current_market_id(self) -> str | None:
        """Get current market ID."""
        if self._current_engine:
            return self._current_engine.market_id
        return None
    
    def select_market(
        self,
        market_id: str | None = None,
        weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> ReplayEngine:
        """
        Select a market for the next episode.
        
        Args:
            market_id: Specific market to select (random if None)
            weights: Sampling weights by market_id
            seed: Random seed
            
        Returns:
            Selected ReplayEngine
        """
        if seed is not None:
            np.random.seed(seed)
        
        if market_id:
            if market_id not in self.engines:
                raise ValueError(f"Market {market_id} not found")
            self._current_engine = self.engines[market_id]
        else:
            if weights:
                w = np.array([weights.get(mid, 1.0) for mid in self.market_ids])
                w = w / w.sum()
                market_id = np.random.choice(self.market_ids, p=w)
            else:
                market_id = np.random.choice(self.market_ids)
            
            self._current_engine = self.engines[market_id]
        
        return self._current_engine
    
    def reset(self, **kwargs) -> MarketState:
        """Reset current engine or select a new one."""
        if self._current_engine is None:
            self.select_market()
        
        return self._current_engine.reset(**kwargs)  # type: ignore
    
    def step(self) -> MarketState:
        """Step current engine."""
        if self._current_engine is None:
            raise RuntimeError("No market selected. Call select_market() or reset()")
        return self._current_engine.step()
    
    def get_state(self) -> MarketState:
        """Get state from current engine."""
        if self._current_engine is None:
            raise RuntimeError("No market selected. Call select_market() or reset()")
        return self._current_engine.get_state()
    
    @property
    def done(self) -> bool:
        """Check if current episode is done."""
        if self._current_engine is None:
            return True
        return self._current_engine.done
