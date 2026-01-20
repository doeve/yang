"""
Two-layer inference pipeline for Polymarket trading.

Combines:
- Layer 1: ProbabilisticLSTM for P(outcome) prediction
- Layer 2: SAC policy for execution decisions

This is the production inference module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np
import pandas as pd
import torch
import structlog

from src.models.probabilistic_lstm import (
    ProbabilisticLSTM,
    ProbabilisticLSTMConfig,
)
from src.models.calibration import CalibrationWrapper
from src.data.multi_resolution_features import MultiResolutionFeatureBuilder

logger = structlog.get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for two-layer inference."""
    
    # Probability model
    probability_model_path: str = ""
    model_type: str = "lstm"
    
    # Execution model
    execution_model_path: str = ""
    
    # Decision thresholds
    min_edge_to_trade: float = 0.03      # Only trade if edge > 3%
    min_confidence_to_trade: float = 0.55  # Model prob must be > 55% or < 45%
    hold_threshold: float = 0.5           # SAC hold probability threshold
    
    # Position sizing
    max_position_size: float = 0.25      # Max position as fraction of balance
    
    # Feature building
    sequence_length: int = 180


class TwoLayerPolymarketBot:
    """
    Production inference combining LSTM probability model with SAC execution policy.
    
    Workflow:
    1. Receive market data (BTC prices, FX, Polymarket odds)
    2. Build multi-resolution features
    3. Run LSTM to get P(outcome)
    4. Construct SAC state (model prob + market conditions)
    5. Run SAC to get action (direction, size, hold)
    6. Return trading decision
    
    The bot learns to:
    - Trade only when there's significant edge
    - Size positions proportionally to confidence
    - Abstain when uncertain
    """
    
    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or InferenceConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.probability_model: Optional[torch.nn.Module] = None
        self.execution_model: Optional[Any] = None
        self.feature_builder = MultiResolutionFeatureBuilder()
        
        # State tracking
        self._last_prediction: Optional[Dict[str, Any]] = None
        self._trade_history: list[Dict[str, Any]] = []
        
        logger.info(
            "TwoLayerPolymarketBot initialized",
            device=self.device,
        )
    
    def load_models(
        self,
        probability_model_path: str,
        execution_model_path: Optional[str] = None,
    ) -> None:
        """
        Load trained models from disk.
        
        Args:
            probability_model_path: Path to trained probability model (.pt)
            execution_model_path: Optional path to SAC model (.zip)
        """
        logger.info("Loading probability model", path=probability_model_path)
        
        prob_path = Path(probability_model_path)
        
        # Load config
        config_path = prob_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config_dict = json.load(f)["model_config"]
                model_config = ProbabilisticLSTMConfig(**model_config_dict)
        else:
            model_config = ProbabilisticLSTMConfig()
        
        # Load model
        self.probability_model = ProbabilisticLSTM(model_config)
        
        # Check for calibrated model
        calibrated_path = prob_path.parent / "calibrated_model.pt"
        if calibrated_path.exists():
            wrapper = CalibrationWrapper(self.probability_model)
            wrapper.load_state_dict(torch.load(calibrated_path, map_location=self.device))
            self.probability_model = wrapper
            logger.info("Loaded calibrated model")
        else:
            self.probability_model.load_state_dict(
                torch.load(probability_model_path, map_location=self.device)
            )
        
        self.probability_model = self.probability_model.to(self.device)
        self.probability_model.eval()
        
        # Load execution model (SAC)
        if execution_model_path:
            logger.info("Loading execution model", path=execution_model_path)
            try:
                from stable_baselines3 import SAC
                self.execution_model = SAC.load(execution_model_path)
                logger.info("SAC execution model loaded")
            except Exception as e:
                logger.warning(f"Could not load SAC model: {e}")
                self.execution_model = None
    
    def predict_probability(
        self,
        btc_data: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
        candle_end_timestamp: Optional[pd.Timestamp] = None,
    ) -> float:
        """
        Get probability prediction from Layer 1 model.
        
        Args:
            btc_data: Recent BTC price/volume data
            dxy_data: Optional DXY data
            eurusd_data: Optional EUR/USD data
            candle_end_timestamp: End of current candle
            
        Returns:
            Probability P(candle closes up) in [0, 1]
        """
        if self.probability_model is None:
            raise RuntimeError("Probability model not loaded")
        
        # Build features
        features = self.feature_builder.build_sequence(
            btc_data,
            dxy_data=dxy_data,
            eurusd_data=eurusd_data,
            sequence_length=self.config.sequence_length,
            candle_end_timestamp=candle_end_timestamp,
        )
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            if hasattr(self.probability_model, "predict_proba"):
                prob = self.probability_model.predict_proba(x)
            else:
                logits = self.probability_model(x)
                prob = torch.sigmoid(logits)
        
        return float(prob.item())
    
    def get_execution_decision(
        self,
        model_prob: float,
        market_yes_price: float,
        spread: float = 0.02,
        time_remaining: float = 1.0,
        current_position: float = 0.0,
        dxy_momentum: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get execution decision from Layer 2 SAC policy.
        
        If no SAC model is loaded, uses simple rule-based logic.
        
        Args:
            model_prob: Probability from Layer 1
            market_yes_price: Current YES price on Polymarket
            spread: Bid-ask spread
            time_remaining: Normalized time to resolution
            current_position: Current position (-1 to 1)
            dxy_momentum: DXY momentum indicator
            
        Returns:
            Dict with 'action', 'size', 'edge', 'confidence'
        """
        # Calculate edge
        edge = model_prob - market_yes_price
        
        # Model confidence (distance from 0.5)
        model_confidence = abs(model_prob - 0.5) * 2
        
        if self.execution_model is not None:
            # Use SAC policy
            obs = np.array([
                model_prob,
                market_yes_price,
                1.0 - market_yes_price,
                spread,
                time_remaining,
                1.0 - time_remaining,
                current_position,
                0.0,  # unrealized_pnl
                model_confidence,
                dxy_momentum,
                0,  # trades_this_candle
                0.5,  # recent_win_rate
                1.0,  # balance_normalized
                edge,
            ], dtype=np.float32)
            
            action, _ = self.execution_model.predict(obs, deterministic=True)
            
            direction = float(action[0])
            size = float(np.clip(action[1], 0.0, 1.0))
            hold_prob = float(action[2])
            
            if hold_prob > self.config.hold_threshold:
                return {
                    "action": "hold",
                    "size": 0.0,
                    "edge": edge,
                    "confidence": model_confidence,
                    "reason": "SAC chose to hold",
                }
            
            trade_action = "buy_yes" if direction > 0 else "buy_no"
            
            return {
                "action": trade_action,
                "size": size * self.config.max_position_size,
                "edge": edge,
                "confidence": model_confidence,
                "reason": "SAC decision",
            }
        
        else:
            # Rule-based fallback
            return self._rule_based_decision(
                model_prob, market_yes_price, edge, model_confidence
            )
    
    def _rule_based_decision(
        self,
        model_prob: float,
        market_yes_price: float,
        edge: float,
        model_confidence: float,
    ) -> Dict[str, Any]:
        """Simple rule-based trading logic as SAC fallback."""
        
        # Check minimum edge
        if abs(edge) < self.config.min_edge_to_trade:
            return {
                "action": "hold",
                "size": 0.0,
                "edge": edge,
                "confidence": model_confidence,
                "reason": f"Edge {abs(edge):.3f} below threshold {self.config.min_edge_to_trade}",
            }
        
        # Check minimum confidence
        if model_confidence < (self.config.min_confidence_to_trade - 0.5) * 2:
            return {
                "action": "hold",
                "size": 0.0,
                "edge": edge,
                "confidence": model_confidence,
                "reason": f"Model confidence {model_confidence:.3f} too low",
            }
        
        # Determine direction
        if edge > 0:
            action = "buy_yes"
        else:
            action = "buy_no"
        
        # Size based on edge (Kelly-like)
        # Optimal: f = edge / odds
        # Simplified: larger edge = larger position
        size = min(abs(edge) * 5, 1.0) * self.config.max_position_size
        
        return {
            "action": action,
            "size": size,
            "edge": edge,
            "confidence": model_confidence,
            "reason": "Rule-based: sufficient edge and confidence",
        }
    
    def step(
        self,
        btc_data: pd.DataFrame,
        market_yes_price: float,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
        candle_end_timestamp: Optional[pd.Timestamp] = None,
        spread: float = 0.02,
        time_remaining: float = 1.0,
        current_position: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Full inference step: predict probability and get trading decision.
        
        Args:
            btc_data: Recent BTC data
            market_yes_price: Current Polymarket YES price
            dxy_data: Optional DXY data
            eurusd_data: Optional EUR/USD data
            candle_end_timestamp: End of current candle
            spread: Market spread
            time_remaining: Normalized time to resolution
            current_position: Current position
            
        Returns:
            Complete decision dict with probability, action, size, edge, etc.
        """
        # Get probability prediction
        model_prob = self.predict_probability(
            btc_data, dxy_data, eurusd_data, candle_end_timestamp
        )
        
        # Get DXY momentum if available
        dxy_momentum = 0.0
        if dxy_data is not None and len(dxy_data) > 1:
            dxy_returns = dxy_data["price"].pct_change().fillna(0)
            dxy_momentum = dxy_returns.iloc[-100:].mean() * 100 if len(dxy_returns) > 100 else 0
        
        # Get execution decision
        decision = self.get_execution_decision(
            model_prob=model_prob,
            market_yes_price=market_yes_price,
            spread=spread,
            time_remaining=time_remaining,
            current_position=current_position,
            dxy_momentum=dxy_momentum,
        )
        
        # Add model probability to decision
        decision["model_probability"] = model_prob
        decision["market_price"] = market_yes_price
        
        # Store for history
        self._last_prediction = decision
        
        return decision
    
    def record_trade_outcome(
        self,
        decision: Dict[str, Any],
        outcome: bool,
        pnl: float,
    ) -> None:
        """Record trade outcome for performance tracking."""
        record = {
            **decision,
            "outcome": outcome,
            "pnl": pnl,
        }
        self._trade_history.append(record)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary from trade history."""
        if not self._trade_history:
            return {"num_trades": 0}
        
        trades = [t for t in self._trade_history if t["action"] != "hold"]
        holds = [t for t in self._trade_history if t["action"] == "hold"]
        
        if not trades:
            return {
                "num_trades": 0,
                "num_holds": len(holds),
                "hold_rate": 1.0,
            }
        
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        
        return {
            "num_trades": len(trades),
            "num_holds": len(holds),
            "hold_rate": len(holds) / len(self._trade_history),
            "win_rate": wins / len(trades),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls),
            "sharpe": np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(len(trades)),
            "max_win": max(pnls),
            "max_loss": min(pnls),
        }


def create_bot(
    probability_model_path: str,
    execution_model_path: Optional[str] = None,
) -> TwoLayerPolymarketBot:
    """
    Factory function to create a configured bot.
    
    Args:
        probability_model_path: Path to trained probability model
        execution_model_path: Optional path to SAC execution model
        
    Returns:
        Configured TwoLayerPolymarketBot
    """
    config = InferenceConfig(
        probability_model_path=probability_model_path,
        execution_model_path=execution_model_path or "",
    )
    
    bot = TwoLayerPolymarketBot(config)
    bot.load_models(probability_model_path, execution_model_path)
    
    return bot
