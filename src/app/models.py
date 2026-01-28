"""
Model Abstraction Layer.
"""
import abc
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import internal models
from src.models.market_predictor import load_market_predictor, MarketPredictorModel, Action

logger = logging.getLogger(__name__)

class Predictor(abc.ABC):
    """Abstract base class for all trading models."""
    
    @abc.abstractmethod
    def load(self, path: str):
        """Load model weights."""
        pass
        
    @abc.abstractmethod
    def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction from standardized market data.
        
        Args:
            market_data: standardized dictionary containing:
                - yes_price_history: List[float] (last N prices)
                - no_price_history: List[float]
                - time_remaining: float (0.0 to 1.0)
                - position: Optional[Dict] (current position state)
                
        Returns:
            Dict containing:
                - action: str (WAIT, BUY_YES, BUY_NO, EXIT, HOLD)
                - confidence: float (0.0 to 1.0)
                - expected_return: float
                - raw_output: Any (debug info)
        """
        pass

class UnifiedPredictor(Predictor):
    """Adapter for MarketPredictorModel (v1/v2)."""
    
    def __init__(self, device: str = "cpu"):
        self.model: Optional[MarketPredictorModel] = None
        self.device = device
        # We need feature builders
        # NOTE: In a real app we might inject these or load config
        # NOTE: In a real app we might inject these or load config
        from src.data.enhanced_features import create_enhanced_feature_builder
        
        self.enhanced_builder = create_enhanced_feature_builder()
        
    def load(self, path: str):
        logger.info(f"Loading UnifiedPredictor from {path}")
        self.model = load_market_predictor(path, self.device)
        self.model.eval()
        
    def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        # Extract data
        # Extract data
        # Use 1-minute sampled history for model (as fixed in data service)
        # Fallback to yes_price_history if model history is missing (should not happen after warmup)
        yes_hist = np.array(market_data.get("model_yes_history") or market_data.get("yes_price_history", []))
        no_hist = np.array(market_data.get("model_no_history") or market_data.get("no_price_history", []))
        
        btc_hist = np.array(market_data.get("btc_price_history", []))
        btc_open = market_data.get("btc_open_price", 0.0)
        time_rem = market_data.get("time_remaining", 0.0)
        
        if len(yes_hist) < 20: 
            return {"action": "WAIT", "confidence": 0.0, "expected_return": 0.0}

        # 1. Build Features
        # Note: Unified model V1 (market_predictor_v1) might assume specific builder
        # We'll use EnhancedFeatureBuilder as standard if compatible
        # For this skeleton, we assume the model input matches enhanced builder
        
        features = self.enhanced_builder.compute_features(
            yes_prices=yes_hist,
            no_prices=no_hist,
            time_remaining=time_rem,
            btc_prices=btc_hist,
            btc_open=btc_open
        )
        
        # 2. Build Position State
        pos = market_data.get("position", {})
        from src.models.market_predictor import EnhancedPositionState
        
        has_pos = pos.get("side") is not None
        pos_state = EnhancedPositionState.compute(
            has_position=has_pos,
            position_side=pos.get("side"),
            entry_price=pos.get("entry_price", 0.0),
            current_price=yes_hist[-1] if pos.get("side") == "yes" else no_hist[-1],
            time_remaining=time_rem,
            ticks_held=pos.get("ticks_held", 0),
            max_pnl_seen=pos.get("max_pnl", 0.0)
        )
        
        # 3. Inference
        with torch.no_grad():
            feat_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            pos_t = torch.FloatTensor(pos_state).unsqueeze(0).to(self.device)
            
            # Using get_action with deterministic=True
            result = self.model.get_action(feat_t, pos_t, deterministic=True)
            
        action_idx = int(result["action"].item())
        action_name = Action.names()[action_idx]
        
        return {
            "action": action_name,
            "confidence": float(result["confidence"].item()),
            "expected_return": float(result["expected_return"].item()),
            "raw_output": result
        }

class ModelFactory:
    @staticmethod
    def load_model(path: str, type_hint: str = "unified") -> Predictor:
        # Simple factory
        if type_hint == "unified":
            p = UnifiedPredictor()
            p.load(path)
            return p
        raise ValueError(f"Unknown model type: {type_hint}")
