"""
Custom policy network for Stable-Baselines3.

Integrates temporal feature extractors with the RL policy network
for learning trading strategies from price sequences.
"""

from typing import Dict, List, Tuple, Type, Any

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .features import LSTMFeatureExtractor, TransformerFeatureExtractor, HybridFeatureExtractor


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for trading environment.
    
    Combines:
    - Temporal feature extraction from price sequences
    - Current state features (position, PnL, etc.)
    - Time-based features
    
    This integrates with Stable-Baselines3's policy architecture.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 64,
        sequence_length: int = 300,  # Matches d3v historyLength
        input_features: int = 2,  # Pure ML: price + volume/orderflow
        extractor_type: str = "lstm",
        hidden_dim: int = 128,
    ):
        """
        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            sequence_length: Length of price sequence in observation
            input_features: Number of features per timestep
            extractor_type: Type of temporal extractor ("lstm", "transformer", "hybrid")
            hidden_dim: Hidden dimension for extractor
        """
        # Calculate the flattened observation size
        obs_shape = observation_space.shape
        assert obs_shape is not None
        
        super().__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.input_features = input_features
        
        # Calculate dimensions (PURE ML FORMAT)
        self.seq_dim = sequence_length  # Price history only
        self.volume_dim = sequence_length  # Volume/order flow history
        self.current_dim = 1  # Current price change
        self.portfolio_dim = 4  # position, pnl, cash, exposure
        self.time_dim = 2  # time_to_resolution, progress
        
        # Expected total dimension (pure ML format)
        expected_dim = self.seq_dim + self.volume_dim + self.current_dim + self.portfolio_dim + self.time_dim
        
        # Temporal feature extractor
        if extractor_type == "lstm":
            self.temporal_extractor = LSTMFeatureExtractor(
                input_dim=input_features,
                hidden_dim=hidden_dim,
                output_dim=features_dim // 2,
            )
        elif extractor_type == "transformer":
            self.temporal_extractor = TransformerFeatureExtractor(
                input_dim=input_features,
                hidden_dim=hidden_dim,
                output_dim=features_dim // 2,
            )
        else:
            self.temporal_extractor = HybridFeatureExtractor(
                input_dim=input_features,
                hidden_dim=hidden_dim,
                output_dim=features_dim // 2,
            )
        
        # Non-temporal features encoder (pure ML format)
        non_temporal_dim = self.current_dim + self.portfolio_dim + self.time_dim
        self.non_temporal_encoder = nn.Sequential(
            nn.Linear(non_temporal_dim, features_dim // 2),
            nn.LayerNorm(features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim // 2),
            nn.LayerNorm(features_dim // 2),
            nn.ReLU(),
        )
        
        # Fusion layer to combine temporal and non-temporal features
        self.fusion = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Flattened observation tensor (batch, obs_dim)
            
        Returns:
            Feature tensor (batch, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Split observation into components (PURE ML FORMAT)
        idx = 0
        
        # Extract price history (300 points)
        price_history = observations[:, idx:idx + self.seq_dim]
        idx += self.seq_dim
        
        # Extract volume/order flow history (300 points)
        volume_history = observations[:, idx:idx + self.volume_dim]
        idx += self.volume_dim
        
        # Combine into sequence [300, 2] for temporal model
        sequence = torch.stack([price_history, volume_history], dim=-1)
        
        # Extract current price change
        current = observations[:, idx:idx + self.current_dim]
        idx += self.current_dim
        
        # Extract portfolio state
        portfolio = observations[:, idx:idx + self.portfolio_dim]
        idx += self.portfolio_dim
        
        # Extract time features
        time_features = observations[:, idx:idx + self.time_dim]
        
        # Process temporal features
        temporal_features = self.temporal_extractor(sequence)
        
        # Process non-temporal features
        non_temporal = torch.cat([current, portfolio, time_features], dim=-1)
        non_temporal_features = self.non_temporal_encoder(non_temporal)
        
        # Fuse all features
        combined = torch.cat([temporal_features, non_temporal_features], dim=-1)
        features = self.fusion(combined)
        
        return features


class TradingPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for trading.
    
    Uses the TradingFeaturesExtractor for processing observations
    and standard MLP networks for the policy and value heads.
    
    Example:
        model = PPO(
            TradingPolicy,
            env,
            policy_kwargs=dict(
                features_extractor_class=TradingFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=64,
                    sequence_length=60,
                    input_features=15,
                    extractor_type="lstm",
                ),
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            ),
        )
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: List[int] | Dict[str, List[int]] | None = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = TradingFeaturesExtractor,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        # Default network architecture
        if net_arch is None:
            net_arch = [dict(pi=[128, 64], vf=[128, 64])]
        
        # Default features extractor kwargs (PURE ML FORMAT)
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(
                features_dim=64,
                sequence_length=300,  # Matches d3v
                input_features=2,  # Pure ML: price + volume
                extractor_type="lstm",
            )
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            *args,
            **kwargs,
        )


class SimpleMLP(nn.Module):
    """
    Simple MLP network for comparison/baseline.
    
    This can be used as a baseline to compare against
    the temporal feature extractors.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_policy_kwargs(
    features_dim: int = 64,
    sequence_length: int = 300,
    input_features: int = 2,
    extractor_type: str = "lstm",
    hidden_dim: int = 128,
    net_arch: List[int] | Dict[str, List[int]] | None = None,
) -> Dict[str, Any]:
    """
    Get policy kwargs for creating a PPO model with custom features extractor.
    
    Args:
        features_dim: Output dimension of feature extractor
        sequence_length: Length of price sequence
        input_features: Number of features per timestep
        extractor_type: Type of temporal extractor
        hidden_dim: Hidden dimension for extractor
        net_arch: Network architecture for policy and value heads
        
    Returns:
        Dict of policy kwargs to pass to PPO
    """
    if net_arch is None:
        net_arch = [dict(pi=[128, 64], vf=[128, 64])]
    
    return dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            sequence_length=sequence_length,
            input_features=input_features,
            extractor_type=extractor_type,
            hidden_dim=hidden_dim,
        ),
        net_arch=net_arch,
        activation_fn=nn.ReLU,
    )
