"""Tests for ML models."""

import pytest
import torch
import numpy as np

from src.models.features import (
    LSTMFeatureExtractor,
    TransformerFeatureExtractor,
    HybridFeatureExtractor,
    create_feature_extractor,
)
from src.models.rewards import RewardCalculator, RewardConfig


class TestFeatureExtractors:
    """Tests for feature extractors."""
    
    @pytest.fixture
    def sample_sequence(self):
        """Create sample input sequence."""
        batch_size = 4
        seq_len = 60
        input_dim = 15
        
        return torch.randn(batch_size, seq_len, input_dim)
    
    def test_lstm_extractor_output_shape(self, sample_sequence):
        """Test LSTM extractor produces correct output shape."""
        extractor = LSTMFeatureExtractor(
            input_dim=15,
            hidden_dim=128,
            output_dim=64,
        )
        
        output = extractor(sample_sequence)
        
        assert output.shape == (4, 64)
    
    def test_transformer_extractor_output_shape(self, sample_sequence):
        """Test Transformer extractor produces correct output shape."""
        extractor = TransformerFeatureExtractor(
            input_dim=15,
            hidden_dim=128,
            output_dim=64,
        )
        
        output = extractor(sample_sequence)
        
        assert output.shape == (4, 64)
    
    def test_hybrid_extractor_output_shape(self, sample_sequence):
        """Test Hybrid extractor produces correct output shape."""
        extractor = HybridFeatureExtractor(
            input_dim=15,
            hidden_dim=128,
            output_dim=64,
        )
        
        output = extractor(sample_sequence)
        
        assert output.shape == (4, 64)
    
    def test_factory_creates_correct_type(self):
        """Test factory function creates correct extractor type."""
        lstm = create_feature_extractor("lstm")
        transformer = create_feature_extractor("transformer")
        hybrid = create_feature_extractor("hybrid")
        
        assert isinstance(lstm, LSTMFeatureExtractor)
        assert isinstance(transformer, TransformerFeatureExtractor)
        assert isinstance(hybrid, HybridFeatureExtractor)
    
    def test_lstm_attention_weights(self, sample_sequence):
        """Test LSTM returns attention weights when requested."""
        extractor = LSTMFeatureExtractor(input_dim=15)
        
        output, attention = extractor(sample_sequence, return_attention=True)
        
        assert attention.shape == (4, 60)  # batch_size, seq_len
        assert torch.allclose(attention.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestRewardCalculator:
    """Tests for reward calculation."""
    
    @pytest.fixture
    def calculator(self):
        """Create reward calculator."""
        config = RewardConfig(reward_scale=1.0)
        return RewardCalculator(config)
    
    def test_positive_pnl_gives_positive_reward(self, calculator):
        """Test that positive PnL contributes positive reward."""
        reward = calculator.compute_reward(
            pnl_delta=100.0,
            position=0.0,
            volatility=0.01,
            transaction_cost=0.0,
            time_to_resolution=0.5,
            current_pnl=100.0,
        )
        
        assert reward > 0
    
    def test_transaction_cost_penalty(self, calculator):
        """Test that transaction costs reduce reward."""
        reward_no_cost = calculator.compute_reward(
            pnl_delta=0.0,
            position=0.0,
            volatility=0.01,
            transaction_cost=0.0,
            time_to_resolution=0.5,
            current_pnl=0.0,
        )
        
        calculator.reset()
        
        reward_with_cost = calculator.compute_reward(
            pnl_delta=0.0,
            position=0.0,
            volatility=0.01,
            transaction_cost=10.0,
            time_to_resolution=0.5,
            current_pnl=0.0,
        )
        
        assert reward_with_cost < reward_no_cost
    
    def test_resolution_bonus(self, calculator):
        """Test resolution bonus for correct positioning."""
        # Long position with True outcome
        reward_correct = calculator.compute_reward(
            pnl_delta=0.0,
            position=100.0,
            volatility=0.01,
            transaction_cost=0.0,
            time_to_resolution=0.05,  # Near resolution
            current_pnl=0.0,
            outcome=True,
            price=0.9,
        )
        
        calculator.reset()
        
        # Long position with False outcome
        reward_wrong = calculator.compute_reward(
            pnl_delta=0.0,
            position=100.0,
            volatility=0.01,
            transaction_cost=0.0,
            time_to_resolution=0.05,
            current_pnl=0.0,
            outcome=False,
            price=0.9,
        )
        
        assert reward_correct > reward_wrong
    
    def test_episode_stats(self, calculator):
        """Test episode statistics calculation."""
        # Generate some rewards
        for i in range(10):
            calculator.compute_reward(
                pnl_delta=np.random.uniform(-10, 20),
                position=0.0,
                volatility=0.01,
                transaction_cost=1.0,
                time_to_resolution=0.5,
                current_pnl=i * 10,
            )
        
        stats = calculator.get_episode_stats()
        
        assert "total_reward" in stats
        assert "mean_reward" in stats
        assert "max_drawdown" in stats
