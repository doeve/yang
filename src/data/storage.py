"""
Data storage layer using SQLite for metadata and Parquet for time-series.

Provides efficient storage and retrieval of market data with:
- SQLite for market metadata and training run logs
- Parquet files for price time-series (memory-mapped for performance)
- Automatic data validation and integrity checks
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .collector import Market, MarketStatus, MarketType, PriceHistory

logger = structlog.get_logger(__name__)

Base = declarative_base()


class MarketModel(Base):
    """SQLAlchemy model for market metadata."""
    
    __tablename__ = "markets"
    
    id = Column(String, primary_key=True)
    condition_id = Column(String, index=True)
    question = Column(Text)
    description = Column(Text)
    market_type = Column(String)
    status = Column(String, index=True)
    created_at = Column(DateTime)
    resolution_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    outcome = Column(Boolean, nullable=True)
    tokens_json = Column(Text)
    tags_json = Column(Text)
    
    # Data availability
    has_price_data = Column(Boolean, default=False)
    price_data_start = Column(DateTime, nullable=True)
    price_data_end = Column(DateTime, nullable=True)
    price_data_points = Column(Integer, default=0)


class TrainingRunModel(Base):
    """SQLAlchemy model for training run logs."""
    
    __tablename__ = "training_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    status = Column(String, default="running")  # running, completed, failed
    
    # Configuration
    config_json = Column(Text)
    
    # Results
    total_episodes = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    final_reward = Column(Float, nullable=True)
    best_reward = Column(Float, nullable=True)
    
    # Metrics (JSON blob)
    metrics_json = Column(Text, nullable=True)
    
    # Model paths
    model_path = Column(String, nullable=True)
    checkpoint_path = Column(String, nullable=True)


class DataStorage:
    """
    Manages data storage for the ML trading system.
    
    Uses a hybrid approach:
    - SQLite for relational data (markets, training runs)
    - Parquet files for time-series data (prices, volumes)
    
    Example:
        storage = DataStorage("/path/to/data")
        storage.save_market(market)
        storage.save_price_history(history)
        df = storage.load_price_data("market_id")
    """
    
    def __init__(self, data_dir: str | Path):
        """
        Initialize data storage.
        
        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize SQLite database
        db_path = self.data_dir / "metadata.db"
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("DataStorage initialized", data_dir=str(self.data_dir))
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # =========================================================================
    # Market operations
    # =========================================================================
    
    def save_market(self, market: Market) -> None:
        """Save market metadata to database."""
        with self.get_session() as session:
            existing = session.query(MarketModel).filter_by(id=market.id).first()
            
            if existing:
                # Update existing
                existing.status = market.status.value
                existing.resolved_at = market.resolved_at
                existing.outcome = market.outcome
            else:
                # Create new
                model = MarketModel(
                    id=market.id,
                    condition_id=market.condition_id,
                    question=market.question,
                    description=market.description,
                    market_type=market.market_type.value,
                    status=market.status.value,
                    created_at=market.created_at,
                    resolution_at=market.resolution_at,
                    resolved_at=market.resolved_at,
                    outcome=market.outcome,
                    tokens_json=json.dumps(market.tokens),
                    tags_json=json.dumps(market.tags),
                )
                session.add(model)
            
            session.commit()
    
    def save_markets(self, markets: list[Market]) -> None:
        """Save multiple markets to database."""
        for market in markets:
            self.save_market(market)
        logger.info("Saved markets", count=len(markets))
    
    def get_market(self, market_id: str) -> Market | None:
        """Get market by ID."""
        with self.get_session() as session:
            model = session.query(MarketModel).filter_by(id=market_id).first()
            if not model:
                return None
            return self._model_to_market(model)
    
    def get_markets(
        self,
        status: MarketStatus | None = None,
        has_price_data: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Get markets with optional filters."""
        with self.get_session() as session:
            query = session.query(MarketModel)
            
            if status:
                query = query.filter(MarketModel.status == status.value)
            
            if has_price_data is not None:
                query = query.filter(MarketModel.has_price_data == has_price_data)
            
            query = query.offset(offset).limit(limit)
            
            return [self._model_to_market(m) for m in query.all()]
    
    def _model_to_market(self, model: MarketModel) -> Market:
        """Convert SQLAlchemy model to Market dataclass."""
        return Market(
            id=model.id,
            condition_id=model.condition_id,
            question=model.question,
            description=model.description or "",
            market_type=MarketType(model.market_type),
            status=MarketStatus(model.status),
            created_at=model.created_at,
            resolution_at=model.resolution_at,
            resolved_at=model.resolved_at,
            outcome=model.outcome,
            tokens=json.loads(model.tokens_json) if model.tokens_json else [],
            tags=json.loads(model.tags_json) if model.tags_json else [],
        )
    
    # =========================================================================
    # Price data operations
    # =========================================================================
    
    def save_price_history(self, history: PriceHistory) -> Path:
        """
        Save price history to Parquet file.
        
        Args:
            history: PriceHistory object to save
            
        Returns:
            Path to the saved Parquet file
        """
        if not history.timestamps:
            raise ValueError("Cannot save empty price history")
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": history.timestamps,
            "price": history.prices,
            "volume": history.volumes,
        })
        
        # Ensure timestamp is datetime with timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # Determine file path based on resolution
        if history.resolution_seconds == 1:
            subdir = self.processed_dir / "1s"
        else:
            subdir = self.raw_dir / f"{history.resolution_seconds}s"
        
        subdir.mkdir(exist_ok=True)
        
        # Use market_id as filename (sanitized)
        safe_id = history.market_id.replace("/", "_").replace(":", "_")
        file_path = subdir / f"{safe_id}.parquet"
        
        # Save to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression="snappy")
        
        # Update market metadata
        with self.get_session() as session:
            market = session.query(MarketModel).filter_by(id=history.market_id).first()
            if market:
                market.has_price_data = True
                market.price_data_start = history.timestamps[0]
                market.price_data_end = history.timestamps[-1]
                market.price_data_points = len(history.timestamps)
                session.commit()
        
        logger.info(
            "Saved price history",
            market_id=history.market_id,
            points=len(history.timestamps),
            file=str(file_path),
        )
        
        return file_path
    
    def load_price_data_numpy(
        self,
        market_id: str,
        resolution_seconds: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Load price data as numpy arrays for efficient processing.
        
        Returns:
            Tuple of (timestamps, prices, volumes) as numpy arrays, or None
        """
        df = self.load_price_data(market_id, resolution_seconds)
        if df is None:
            return None
        
        timestamps = df["timestamp"].values.astype("datetime64[s]").astype(np.float64)
        prices = df["price"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)
        
        return timestamps, prices, volumes
    
    def get_available_markets_with_data(self) -> list[str]:
        """Get list of market IDs that have price data available."""
        market_ids = []
        
        # Check SQLite database for tracked markets
        with self.get_session() as session:
            markets = session.query(MarketModel).filter(
                MarketModel.has_price_data == True  # noqa: E712
            ).all()
            market_ids.extend([m.id for m in markets])
        
        # PRIORITY 1: Check for 100ms tick data files (matches d3v)
        for parquet_file in self.data_dir.glob("*_100ms.parquet"):
            market_id = parquet_file.stem.replace("_100ms", "").upper()
            if market_id not in market_ids:
                market_ids.append(market_id)
        
        # PRIORITY 2: Check for regular price files (1-min candles)
        for parquet_file in self.data_dir.glob("*_prices.parquet"):
            # Extract market_id from filename (e.g., btcusdt_prices.parquet -> BTCUSDT)
            market_id = parquet_file.stem.replace("_prices", "").upper()
            if market_id not in market_ids:
                market_ids.append(market_id)
        
        return market_ids
    
    def load_price_data(
        self,
        market_id: str,
        resolution_seconds: int = 1,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame | None:
        """
        Load price data from Parquet file.
        
        Searches in multiple locations:
        1. Directly in data_dir (from Binance collector)
        2. In processed/1s subdirectory
        3. In raw/{resolution}s subdirectory
        
        Returns:
            DataFrame with timestamp, price, volume columns, or None if not found
        """
        # PRIORITY 1: 100ms tick data (matches d3v live format exactly)
        tick_path = self.data_dir / f"{market_id.lower()}_100ms.parquet"
        if tick_path.exists():
            logger.debug("Loading 100ms tick data", market_id=market_id)
            df = pd.read_parquet(tick_path)
            if start_time:
                df = df[df["timestamp"] >= start_time]
            if end_time:
                df = df[df["timestamp"] <= end_time]
            return df
        
        # PRIORITY 2: Direct parquet file (1-min candles from Binance collector)
        direct_path = self.data_dir / f"{market_id.lower()}_prices.parquet"
        if direct_path.exists():
            df = pd.read_parquet(direct_path)
            if start_time:
                df = df[df["timestamp"] >= start_time]
            if end_time:
                df = df[df["timestamp"] <= end_time]
            return df
        
        # Try subdirectories
        if resolution_seconds == 1:
            subdir = self.processed_dir / "1s"
        else:
            subdir = self.raw_dir / f"{resolution_seconds}s"
        
        safe_id = market_id.replace("/", "_").replace(":", "_")
        file_path = subdir / f"{safe_id}.parquet"
        
        if not file_path.exists():
            logger.warning("Price data not found", market_id=market_id, path=str(file_path))
            return None
        
        df = pd.read_parquet(file_path)
        
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]
        
        return df
    
    # =========================================================================
    # Training run operations
    # =========================================================================
    
    def create_training_run(
        self,
        name: str,
        config: dict[str, Any],
    ) -> int:
        """Create a new training run record."""
        with self.get_session() as session:
            run = TrainingRunModel(
                name=name,
                config_json=json.dumps(config),
            )
            session.add(run)
            session.commit()
            return run.id
    
    def update_training_run(
        self,
        run_id: int,
        status: str | None = None,
        total_episodes: int | None = None,
        total_steps: int | None = None,
        final_reward: float | None = None,
        best_reward: float | None = None,
        metrics: dict[str, Any] | None = None,
        model_path: str | None = None,
    ) -> None:
        """Update training run with progress/results."""
        with self.get_session() as session:
            run = session.query(TrainingRunModel).filter_by(id=run_id).first()
            if not run:
                raise ValueError(f"Training run {run_id} not found")
            
            if status:
                run.status = status
                if status in ["completed", "failed"]:
                    run.completed_at = datetime.now(timezone.utc)
            
            if total_episodes is not None:
                run.total_episodes = total_episodes
            if total_steps is not None:
                run.total_steps = total_steps
            if final_reward is not None:
                run.final_reward = final_reward
            if best_reward is not None:
                run.best_reward = best_reward
            if metrics:
                run.metrics_json = json.dumps(metrics)
            if model_path:
                run.model_path = model_path
            
            session.commit()
    
    def get_training_runs(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get training runs as dictionaries."""
        with self.get_session() as session:
            query = session.query(TrainingRunModel)
            
            if status:
                query = query.filter(TrainingRunModel.status == status)
            
            query = query.order_by(TrainingRunModel.started_at.desc()).limit(limit)
            
            runs = []
            for run in query.all():
                runs.append({
                    "id": run.id,
                    "name": run.name,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "status": run.status,
                    "config": json.loads(run.config_json) if run.config_json else {},
                    "total_episodes": run.total_episodes,
                    "total_steps": run.total_steps,
                    "final_reward": run.final_reward,
                    "best_reward": run.best_reward,
                    "metrics": json.loads(run.metrics_json) if run.metrics_json else {},
                    "model_path": run.model_path,
                })
            
            return runs
    
    # =========================================================================
    # Model storage
    # =========================================================================
    
    def get_model_path(self, name: str, version: str = "latest") -> Path:
        """Get path for model storage."""
        model_dir = self.models_dir / name
        model_dir.mkdir(exist_ok=True)
        return model_dir / f"{version}.pt"
    
    def list_models(self) -> list[dict[str, Any]]:
        """List all saved models."""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                versions = []
                for model_file in model_dir.glob("*.pt"):
                    stat = model_file.stat()
                    versions.append({
                        "version": model_file.stem,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    })
                
                if versions:
                    models.append({
                        "name": model_dir.name,
                        "versions": sorted(versions, key=lambda x: x["modified_at"], reverse=True),
                    })
        
        return models
