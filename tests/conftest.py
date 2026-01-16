"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_price_data():
    """Generate sample price data for tests."""
    np.random.seed(42)
    
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=1000,
        freq="1s",
    )
    
    prices = 0.5 + np.cumsum(np.random.randn(1000) * 0.001)
    prices = np.clip(prices, 0.01, 0.99)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "volume": np.abs(np.random.randn(1000)) * 1000,
    })


@pytest.fixture
def small_price_data():
    """Generate small price data for quick tests."""
    np.random.seed(42)
    
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=100,
        freq="1s",
    )
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": np.random.uniform(0.3, 0.7, 100),
        "volume": np.random.uniform(100, 1000, 100),
    })
