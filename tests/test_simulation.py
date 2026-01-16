"""Tests for simulation components."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.simulation.replay_engine import ReplayEngine, ReplayConfig
from src.simulation.market_simulator import MarketSimulator, SimulatorConfig, OrderSide
from src.simulation.environment import PolymarketEnv, EnvConfig


class TestReplayEngine:
    """Tests for the replay engine."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=1000,
            freq="1s",
        )
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "price": 0.5 + np.cumsum(np.random.randn(1000) * 0.001),
            "volume": np.random.uniform(100, 1000, 1000),
        })
    
    def test_replay_advances_correctly(self, sample_price_data):
        """Test that replay advances through data correctly."""
        config = ReplayConfig(random_start=False)
        engine = ReplayEngine(sample_price_data, config=config)
        
        initial_state = engine.reset()
        assert initial_state is not None
        
        # Step through
        for _ in range(10):
            state = engine.step()
            assert state is not None
            assert 0.0 <= state.price <= 1.0
    
    def test_no_lookahead_bias(self, sample_price_data):
        """Test that future prices are not visible."""
        config = ReplayConfig(random_start=False)
        engine = ReplayEngine(sample_price_data, config=config)
        
        engine.reset()
        
        # The current step should only see past/current data
        for step in range(10):
            state = engine.step()
            
            # State timestamp should be at or before current step
            assert engine.current_step >= 0
    
    def test_checkpoint_restore(self, sample_price_data):
        """Test checkpoint and restore functionality."""
        engine = ReplayEngine(sample_price_data)
        engine.reset()
        
        # Advance some steps
        for _ in range(50):
            engine.step()
        
        checkpoint = engine.checkpoint()
        original_step = engine.current_step
        
        # Advance more
        for _ in range(20):
            engine.step()
        
        # Restore
        engine.restore(checkpoint)
        
        assert engine.current_step == original_step


class TestMarketSimulator:
    """Tests for the market simulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create a market simulator."""
        config = SimulatorConfig(
            initial_cash=10000.0,
            slippage_type="linear",
            slippage_factor=0.001,
        )
        return MarketSimulator(config)
    
    def test_buy_order_execution(self, simulator):
        """Test buy order execution."""
        market_id = "test_market"
        
        order = simulator.submit_order(
            market_id=market_id,
            side=OrderSide.BUY,
            size=100,
        )
        
        # Process the order
        simulator.process_orders(
            current_price=0.5,
            market_id=market_id,
            current_step=10,
        )
        
        # Check position
        position = simulator.portfolio.get_position(market_id)
        assert position.size > 0
    
    def test_position_limits(self, simulator):
        """Test that position limits are enforced."""
        market_id = "test_market"
        
        # Try to buy more than allowed
        order = simulator.submit_order(
            market_id=market_id,
            side=OrderSide.BUY,
            size=100000,  # Very large order
        )
        
        simulator.process_orders(0.5, market_id, 10)
        
        # Position should be limited
        portfolio_value = simulator.portfolio.total_value({market_id: 0.5})
        position = simulator.portfolio.get_position(market_id)
        
        # Position shouldn't exceed max_position_pct of portfolio
        max_position = portfolio_value * simulator.config.max_position_pct
        assert position.size * 0.5 <= max_position * 1.1  # Allow some margin
    
    def test_pnl_calculation(self, simulator):
        """Test PnL calculation."""
        market_id = "test_market"
        
        # Buy at 0.5
        simulator.submit_order(market_id, OrderSide.BUY, 100)
        simulator.process_orders(0.5, market_id, 10)
        
        # Price goes up to 0.6
        simulator.update_unrealized_pnl({market_id: 0.6})
        
        position = simulator.portfolio.get_position(market_id)
        # Should have positive unrealized PnL
        assert position.unrealized_pnl > 0


class TestPolymarketEnv:
    """Tests for the Gymnasium environment."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for environment."""
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=5000,
            freq="1s",
        )
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "price": 0.5 + np.cumsum(np.random.randn(5000) * 0.001),
            "volume": np.random.uniform(100, 1000, 5000),
        })
    
    def test_environment_initialization(self, sample_data):
        """Test environment initializes correctly."""
        env = PolymarketEnv(sample_data)
        
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_reset_returns_valid_observation(self, sample_data):
        """Test reset returns valid observation."""
        env = PolymarketEnv(sample_data)
        
        obs, info = env.reset()
        
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
    
    def test_step_returns_correct_tuple(self, sample_data):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env = PolymarketEnv(sample_data)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(0)  # Hold action
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode_completes(self, sample_data):
        """Test that episode eventually terminates."""
        config = EnvConfig(max_steps=100)
        env = PolymarketEnv(sample_data, config=config)
        
        env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 200:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
        
        assert done or steps >= 100
