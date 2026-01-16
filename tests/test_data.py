"""Tests for data layer components."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.collector import PolymarketCollector, Market
from src.data.storage import DataStorage
from src.data.preprocessor import FeaturePreprocessor, FeatureConfig


class TestPolymarketCollector:
    """Tests for the Polymarket data collector."""
    
    @pytest.fixture
    def mock_price_history(self):
        """Generate mock price history."""
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=100,
            freq="1min",
        )
        return pd.DataFrame({
            "timestamp": timestamps,
            "price": np.random.uniform(0.3, 0.7, 100),
            "volume": np.random.uniform(100, 1000, 100),
        })
    
    @pytest.mark.asyncio
    async def test_interpolate_to_seconds(self, mock_price_history):
        """Test that interpolation produces 1-second data."""
        collector = PolymarketCollector()
        
        # Mock the method
        result = await collector.interpolate_to_seconds(mock_price_history)
        
        # Check that we have more points (roughly 60x more - one per second)
        assert len(result) >= len(mock_price_history)
        
        # Check columns exist
        assert "timestamp" in result.columns
        assert "price" in result.columns


class TestDataStorage:
    """Tests for data storage."""
    
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage."""
        return DataStorage(tmp_path)
    
    def test_create_training_run(self, temp_storage):
        """Test creating a training run."""
        run_id = temp_storage.create_training_run(
            name="test_run",
            config={"timesteps": 1000},
        )
        
        assert run_id is not None
        assert run_id > 0
    
    def test_update_training_run(self, temp_storage):
        """Test updating a training run."""
        run_id = temp_storage.create_training_run(
            name="test_run",
            config={},
        )
        
        temp_storage.update_training_run(
            run_id,
            status="completed",
            final_reward=100.0,
        )
        
        runs = temp_storage.get_training_runs()
        run = next(r for r in runs if r["id"] == run_id)
        
        assert run["status"] == "completed"


class TestFeaturePreprocessor:
    """Tests for feature preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=200,
            freq="1s",
        )
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "price": 0.5 + np.cumsum(np.random.randn(200) * 0.01),
            "volume": np.random.uniform(100, 1000, 200),
        })
    
    def test_transform_produces_features(self, sample_data):
        """Test that transform produces expected feature columns."""
        preprocessor = FeaturePreprocessor()
        
        result = preprocessor.transform(sample_data)
        
        # Check for expected columns
        assert "price_normalized" in result.columns
        assert "returns_1m" in result.columns
        assert "rsi" in result.columns
        assert "macd_histogram" in result.columns
    
    def test_fit_computes_statistics(self, sample_data):
        """Test that fit computes normalization statistics."""
        preprocessor = FeaturePreprocessor()
        
        preprocessor.fit(sample_data)
        
        assert preprocessor._fitted_stats is not None
        assert "mean" in preprocessor._fitted_stats
        assert "std" in preprocessor._fitted_stats
    
    def test_get_sequence_returns_correct_shape(self, sample_data):
        """Test sequence extraction."""
        config = FeatureConfig(sequence_length=30)
        preprocessor = FeaturePreprocessor(config)
        
        features = preprocessor.transform(sample_data)
        sequence = preprocessor.get_sequence(features, len(features) - 1)
        
        assert sequence.shape[0] == 30
        assert sequence.shape[1] > 0
