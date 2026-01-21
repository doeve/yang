"""
Two-layer inference pipeline combining DeepLOB + SAC.

Layer 1: DeepLOB model for 3-class prediction (Down/Hold/Up)
Layer 2: SAC policy for execution decision (when + how much to trade)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DeepLOBInferenceConfig:
    """Configuration for DeepLOB two-layer inference."""
    
    # Thresholds
    min_confidence_to_trade: float = 0.45
    min_edge_to_trade: float = 0.02
    
    # Position sizing
    max_position_size: float = 0.25
    kelly_fraction: float = 0.25  # Use 25% of Kelly for safety
    
    # Rule-based fallback (when no SAC model)
    use_rule_based: bool = True
    
    # Bias correction
    contrarian_mode: str = "auto"  # "auto", "always", "never"
    contrarian_threshold: float = 0.4  # Invert signals if accuracy below this
    bias_window: int = 50  # Rolling window for bias calculation


class BiasCorrector:
    """Track model predictions and detect/correct systematic bias."""
    
    def __init__(self, window: int = 50, contrarian_threshold: float = 0.4):
        self.window = window
        self.contrarian_threshold = contrarian_threshold
        self.predictions: list[tuple[int, int]] = []  # (predicted_class, actual_outcome)
    
    def update(self, predicted_class: int, actual_outcome: int) -> None:
        """Record a prediction and outcome."""
        self.predictions.append((predicted_class, actual_outcome))
        if len(self.predictions) > self.window:
            self.predictions.pop(0)
    
    def get_bias(self) -> float:
        """
        Calculate model's directional bias.
        
        Returns:
            +1 = always predicts Up
            -1 = always predicts Down
            0 = balanced
        """
        if not self.predictions:
            return 0.0
        
        up_preds = sum(1 for p, _ in self.predictions if p == 2)
        down_preds = sum(1 for p, _ in self.predictions if p == 0)
        total = up_preds + down_preds
        
        if total == 0:
            return 0.0
        
        return (up_preds - down_preds) / total
    
    def get_accuracy_by_direction(self) -> Dict[str, float]:
        """Get accuracy for Up and Down predictions separately."""
        up_preds = [(p, o) for p, o in self.predictions if p == 2]
        down_preds = [(p, o) for p, o in self.predictions if p == 0]
        
        up_correct = sum(1 for p, o in up_preds if o == 2) if up_preds else 0
        down_correct = sum(1 for p, o in down_preds if o == 0) if down_preds else 0
        
        return {
            "up_accuracy": up_correct / max(len(up_preds), 1),
            "down_accuracy": down_correct / max(len(down_preds), 1),
            "up_count": len(up_preds),
            "down_count": len(down_preds),
        }
    
    def get_overall_accuracy(self) -> float:
        """Overall prediction accuracy."""
        if not self.predictions:
            return 0.5
        
        correct = sum(1 for p, o in self.predictions if p == o)
        return correct / len(self.predictions)
    
    def should_invert(self) -> bool:
        """Return True if model is consistently wrong and signals should be inverted."""
        if len(self.predictions) < 10:
            return False
        
        accuracy = self.get_overall_accuracy()
        return accuracy < self.contrarian_threshold
    
    def invert_probabilities(self, class_probs: Dict[str, float]) -> Dict[str, float]:
        """Swap Up and Down probabilities for contrarian betting."""
        return {
            "up": class_probs["down"],
            "down": class_probs["up"],
            "hold": class_probs["hold"],
            "predicted_class": 2 if class_probs["down"] > class_probs["up"] else 0,
            "confidence": max(class_probs["down"], class_probs["up"], class_probs["hold"]),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current bias correction stats."""
        return {
            "predictions_tracked": len(self.predictions),
            "bias": self.get_bias(),
            "overall_accuracy": self.get_overall_accuracy(),
            "should_invert": self.should_invert(),
            **self.get_accuracy_by_direction(),
        }


class DeepLOBTwoLayerBot:
    """
    Production inference bot combining DeepLOB + SAC.
    
    Usage:
        bot = DeepLOBTwoLayerBot()
        bot.load_models("./logs/deep_lob_v2", "./logs/sac_execution")
        
        decision = bot.step(
            btc_data=btc_df,
            market_yes_price=0.52,
        )
    """
    
    def __init__(
        self,
        config: Optional[DeepLOBInferenceConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DeepLOBInferenceConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.deep_lob_model: Optional[nn.Module] = None
        self.sac_model: Optional[Any] = None
        self.feature_builder = None
        
        # Normalization stats
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        
        # Trade history
        self._trade_history: list[Dict[str, Any]] = []
        
        # Bias correction
        self.bias_corrector = BiasCorrector(
            window=self.config.bias_window,
            contrarian_threshold=self.config.contrarian_threshold,
        )
        
        logger.info("DeepLOBTwoLayerBot initialized", device=self.device)

    
    def load_models(
        self,
        deep_lob_path: str,
        sac_path: Optional[str] = None,
    ) -> None:
        """
        Load trained models.
        
        Args:
            deep_lob_path: Path to DeepLOB model directory
            sac_path: Optional path to SAC execution model
        """
        from src.models.deep_lob_model import DeepLOBModel, DeepLOBConfig
        from src.data.deep_lob_features import DeepLOBFeatureBuilder
        
        deep_lob_dir = Path(deep_lob_path)
        
        # Load config
        config_path = deep_lob_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
            
            # Load normalization stats
            if "feature_mean" in saved_config:
                self.feature_mean = np.array(saved_config["feature_mean"])
                self.feature_std = np.array(saved_config["feature_std"])
            
            model_config = DeepLOBConfig(**saved_config.get("model_config", {}))
        else:
            model_config = DeepLOBConfig()
        
        # Load DeepLOB model
        self.deep_lob_model = DeepLOBModel(model_config).to(self.device)
        
        model_weights = deep_lob_dir / "final_model.pt"
        if model_weights.exists():
            self.deep_lob_model.load_state_dict(
                torch.load(model_weights, map_location=self.device, weights_only=True)
            )
            logger.info("Loaded DeepLOB model", path=str(model_weights))
        
        self.deep_lob_model.eval()
        
        # Feature builder
        self.feature_builder = DeepLOBFeatureBuilder()
        
        # Load SAC model if provided
        if sac_path:
            from stable_baselines3 import SAC
            
            sac_dir = Path(sac_path)
            sac_model_path = sac_dir / "execution_model.zip"
            
            if sac_model_path.exists():
                self.sac_model = SAC.load(str(sac_model_path))
                logger.info("Loaded SAC execution model", path=str(sac_model_path))
            else:
                logger.warning("SAC model not found, using rule-based execution")
        else:
            logger.info("No SAC model provided, using rule-based execution")
    
    def predict_class_probabilities(
        self,
        btc_data: pd.DataFrame,
        sequence_length: int = 120,
    ) -> Dict[str, float]:
        """
        Get 3-class probabilities from DeepLOB.
        
        Returns:
            {"down": prob, "hold": prob, "up": prob, "predicted_class": int}
        """
        if self.deep_lob_model is None:
            raise RuntimeError("DeepLOB model not loaded")
        
        # Build features
        features = self.feature_builder.precompute_all_features(btc_data)
        
        # Take last sequence_length samples
        if len(features) < sequence_length:
            # Pad with zeros
            pad = np.zeros((sequence_length - len(features), features.shape[1]))
            features = np.vstack([pad, features])
        else:
            features = features[-sequence_length:]
        
        # Normalize
        if self.feature_mean is not None:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
            features = np.clip(features, -5, 5)
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            logits = self.deep_lob_model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        return {
            "down": float(probs[0]),
            "hold": float(probs[1]),
            "up": float(probs[2]),
            "predicted_class": int(np.argmax(probs)),
            "confidence": float(probs.max()),
        }
    
    def get_execution_decision(
        self,
        class_probs: Dict[str, float],
        market_yes_price: float,
        market_spread: float = 0.02,
        time_remaining: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Get execution decision from SAC or rule-based fallback.
        
        Returns:
            {
                "action": "buy_yes" | "buy_no" | "hold",
                "size": float (0-1),
                "confidence": float,
                "edge": float,
                "reason": str,
            }
        """
        prob_down = class_probs["down"]
        prob_hold = class_probs["hold"]
        prob_up = class_probs["up"]
        predicted_class = class_probs["predicted_class"]
        confidence = class_probs["confidence"]
        
        # Calculate model-implied price and edge
        model_implied_price = 0.5 + (prob_up - prob_down) * 0.5
        edge = model_implied_price - market_yes_price
        
        if self.sac_model is not None:
            # Use SAC policy
            obs = self._build_sac_observation(class_probs, market_yes_price, market_spread, time_remaining)
            action, _ = self.sac_model.predict(obs, deterministic=True)
            
            direction = float(action[0])
            size = float(np.clip(action[1], 0, 1))
            hold_prob = float(action[2])
            
            if hold_prob > 0.5:
                return {
                    "action": "hold",
                    "size": 0.0,
                    "confidence": confidence,
                    "edge": edge,
                    "reason": "SAC chose to hold",
                }
            
            if direction > 0:
                return {
                    "action": "buy_yes",
                    "size": size * self.config.max_position_size,
                    "confidence": confidence,
                    "edge": edge,
                    "reason": "SAC chose to buy YES",
                }
            else:
                return {
                    "action": "buy_no",
                    "size": size * self.config.max_position_size,
                    "confidence": confidence,
                    "edge": edge,
                    "reason": "SAC chose to buy NO",
                }
        
        else:
            # Rule-based fallback
            return self._rule_based_decision(class_probs, market_yes_price, edge)
    
    def _rule_based_decision(
        self,
        class_probs: Dict[str, float],
        market_yes_price: float,
        edge: float,
        aggressive: bool = False,
    ) -> Dict[str, Any]:
        """Rule-based execution when no SAC model is available.
        
        Args:
            aggressive: If True, trade based on Up vs Down probability difference
                       instead of waiting for predicted class != Hold
        """
        predicted_class = class_probs["predicted_class"]
        confidence = class_probs["confidence"]
        prob_up = class_probs["up"]
        prob_down = class_probs["down"]
        
        # Aggressive mode: trade if Up or Down has higher prob than Hold
        if aggressive:
            up_down_diff = prob_up - prob_down
            
            # Trade if clear direction (Up or Down significantly higher)
            if abs(up_down_diff) > 0.03:  # 3% difference threshold (lowered)
                position_size = min(abs(up_down_diff) * 2, self.config.max_position_size)
                
                if up_down_diff > 0:
                    return {
                        "action": "buy_yes",
                        "size": position_size,
                        "confidence": prob_up,
                        "edge": edge,
                        "reason": f"Aggressive: Up ({prob_up:.2f}) > Down ({prob_down:.2f})",
                    }
                else:
                    return {
                        "action": "buy_no",
                        "size": position_size,
                        "confidence": prob_down,
                        "edge": edge,
                        "reason": f"Aggressive: Down ({prob_down:.2f}) > Up ({prob_up:.2f})",
                    }
            
            # Low directional confidence
            return {
                "action": "hold",
                "size": 0.0,
                "confidence": confidence,
                "edge": edge,
                "reason": f"Aggressive: No clear direction (diff={abs(up_down_diff):.3f})",
            }
        
        # Standard mode: Don't trade if Hold is predicted
        if predicted_class == 1:
            return {
                "action": "hold",
                "size": 0.0,
                "confidence": confidence,
                "edge": edge,
                "reason": "Model predicts Hold",
            }
        
        # Don't trade if low confidence
        if confidence < self.config.min_confidence_to_trade:
            return {
                "action": "hold",
                "size": 0.0,
                "confidence": confidence,
                "edge": edge,
                "reason": f"Low confidence ({confidence:.2f} < {self.config.min_confidence_to_trade})",
            }
        
        # Don't trade if edge is too small
        if abs(edge) < self.config.min_edge_to_trade:
            return {
                "action": "hold",
                "size": 0.0,
                "confidence": confidence,
                "edge": edge,
                "reason": f"Small edge ({abs(edge):.3f} < {self.config.min_edge_to_trade})",
            }
        
        # Calculate position size using Kelly criterion
        win_prob = confidence
        loss_prob = 1 - win_prob
        reward_ratio = abs(edge) * 10
        
        kelly_size = max(0, (win_prob * reward_ratio - loss_prob) / reward_ratio)
        position_size = min(kelly_size * self.config.kelly_fraction, self.config.max_position_size)
        
        if position_size < 0.01:
            return {
                "action": "hold",
                "size": 0.0,
                "confidence": confidence,
                "edge": edge,
                "reason": "Kelly size too small",
            }
        
        # Determine direction
        if predicted_class == 2:  # Up predicted
            return {
                "action": "buy_yes",
                "size": position_size,
                "confidence": confidence,
                "edge": edge,
                "reason": f"Up predicted (conf={confidence:.2f}, edge={edge:.3f})",
            }
        else:  # Down predicted
            return {
                "action": "buy_no",
                "size": position_size,
                "confidence": confidence,
                "edge": edge,
                "reason": f"Down predicted (conf={confidence:.2f}, edge={edge:.3f})",
            }

    
    def _build_sac_observation(
        self,
        class_probs: Dict[str, float],
        market_yes_price: float,
        market_spread: float,
        time_remaining: float,
    ) -> np.ndarray:
        """Build observation for SAC model."""
        model_implied_price = 0.5 + (class_probs["up"] - class_probs["down"]) * 0.5
        edge = model_implied_price - market_yes_price
        
        return np.array([
            class_probs["down"],
            class_probs["hold"],
            class_probs["up"],
            class_probs["predicted_class"] / 2.0,
            class_probs["confidence"],
            market_yes_price,
            1.0 - market_yes_price,
            market_spread,
            time_remaining,
            0.0,  # position
            0.0,  # unrealized_pnl
            0.0,  # trades_this_candle
            0.5,  # win_rate
            1.0,  # balance_norm
            edge,
        ], dtype=np.float32)
    
    def step(
        self,
        btc_data: pd.DataFrame,
        market_yes_price: float,
        market_spread: float = 0.02,
        time_remaining: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Full inference step: predict + decide.
        
        Args:
            btc_data: Recent BTC data with price, volume, buy_pressure, timestamp
            market_yes_price: Current Polymarket YES price
            market_spread: Current bid-ask spread
            time_remaining: Time remaining in candle (0-1)
            
        Returns:
            Complete decision with action, size, reasoning
        """
        # Layer 1: Get class probabilities
        class_probs = self.predict_class_probabilities(btc_data)
        
        # Apply contrarian mode if configured
        apply_inversion = False
        if self.config.contrarian_mode == "always":
            apply_inversion = True
        elif self.config.contrarian_mode == "auto":
            apply_inversion = self.bias_corrector.should_invert()
        
        original_class_probs = class_probs.copy()
        if apply_inversion:
            class_probs = self.bias_corrector.invert_probabilities(class_probs)
            logger.info("Applying contrarian mode - inverting signals")
        
        # Layer 2: Get execution decision
        decision = self.get_execution_decision(
            class_probs,
            market_yes_price,
            market_spread,
            time_remaining,
        )
        
        # Add class probabilities to decision
        decision.update({
            "prob_down": original_class_probs["down"],
            "prob_hold": original_class_probs["hold"],
            "prob_up": original_class_probs["up"],
            "predicted_class": ["Down", "Hold", "Up"][original_class_probs["predicted_class"]],
            "contrarian_applied": apply_inversion,
            "bias_stats": self.bias_corrector.get_stats(),
        })
        
        return decision

    
    def record_outcome(
        self,
        decision: Dict[str, Any],
        outcome: int,  # 0=Down, 1=Hold, 2=Up
        pnl: float,
    ) -> None:
        """Record trade outcome for analysis."""
        self._trade_history.append({
            **decision,
            "outcome": outcome,
            "pnl": pnl,
        })
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._trade_history:
            return {}
        
        trades = [t for t in self._trade_history if t["action"] != "hold"]
        holds = [t for t in self._trade_history if t["action"] == "hold"]
        
        if trades:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            total_pnl = sum(t["pnl"] for t in trades)
            
            return {
                "total_decisions": len(self._trade_history),
                "trades": len(trades),
                "holds": len(holds),
                "hold_rate": len(holds) / len(self._trade_history),
                "win_rate": wins / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": total_pnl / len(trades) if trades else 0,
            }
        
        return {"total_decisions": len(self._trade_history), "trades": 0}


def create_deep_lob_bot(
    deep_lob_path: str,
    sac_path: Optional[str] = None,
) -> DeepLOBTwoLayerBot:
    """Factory function to create and load bot."""
    bot = DeepLOBTwoLayerBot()
    bot.load_models(deep_lob_path, sac_path)
    return bot
