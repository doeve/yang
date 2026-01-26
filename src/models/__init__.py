"""ML models for trading."""

from .features import (
    LSTMFeatureExtractor,
    TransformerFeatureExtractor,
    HybridFeatureExtractor,
    create_feature_extractor,
)
from .policy import TradingPolicy, TradingFeaturesExtractor, get_policy_kwargs
from .agent import TradingAgent, AgentConfig, EnsembleAgent
from .rewards import RewardCalculator, RewardConfig, SharpeRewardCalculator, create_reward_calculator
from .market_predictor import (
    MarketPredictorModel,
    MarketPredictorConfig,
    MarketPredictorTrainer,
    OptimalActionLabeler,
    EnhancedPositionState,
    Action,
    load_market_predictor,
)
from .probability_predictor import (
    ProbabilityPredictorModel,
    ProbabilityPredictorConfig,
    ProbabilityPredictorTrainer,
    calibrate_temperature,
    compute_calibration_metrics,
    load_probability_predictor,
)

__all__ = [
    "LSTMFeatureExtractor",
    "TransformerFeatureExtractor",
    "HybridFeatureExtractor",
    "create_feature_extractor",
    "TradingPolicy",
    "TradingFeaturesExtractor",
    "get_policy_kwargs",
    "TradingAgent",
    "AgentConfig",
    "EnsembleAgent",
    "RewardCalculator",
    "RewardConfig",
    "SharpeRewardCalculator",
    "create_reward_calculator",
    "MarketPredictorModel",
    "MarketPredictorConfig",
    "MarketPredictorTrainer",
    "OptimalActionLabeler",
    "EnhancedPositionState",
    "Action",
    "load_market_predictor",
    "ProbabilityPredictorModel",
    "ProbabilityPredictorConfig",
    "ProbabilityPredictorTrainer",
    "calibrate_temperature",
    "compute_calibration_metrics",
    "load_probability_predictor",
]
