"""
Model export utilities for production deployment.

Supports multiple export formats:
- PyTorch checkpoint (for continued training)
- TorchScript (for optimized inference)
- ONNX (for cross-platform deployment)
"""

from pathlib import Path
from typing import Any, Literal
import json

import torch
import numpy as np
import structlog

from ..models.agent import TradingAgent, AgentConfig
from ..models.features import create_feature_extractor

logger = structlog.get_logger(__name__)


class ModelExporter:
    """
    Exports trained models for production deployment.
    
    Supports:
    - Full checkpoint (model + config)
    - TorchScript (optimized for inference)
    - ONNX (cross-platform)
    
    Example:
        exporter = ModelExporter()
        exporter.export_checkpoint(agent, "model.pt")
        exporter.export_onnx(agent, "model.onnx", input_shape=(1, 906))
    """
    
    def export_checkpoint(
        self,
        agent: TradingAgent,
        output_path: str | Path,
        include_optimizer: bool = False,
    ) -> Path:
        """
        Export full checkpoint for training resume.
        
        Args:
            agent: Trained agent
            output_path: Output file path
            include_optimizer: Whether to include optimizer state
            
        Returns:
            Path to saved checkpoint
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if agent.model is None:
            raise ValueError("Agent has no model to export")
        
        # Save using SB3's built-in save
        agent.save(output_path)
        
        # Also save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        
        metadata = {
            "format": "checkpoint",
            "agent_config": {
                "features_dim": agent.config.features_dim,
                "hidden_dim": agent.config.hidden_dim,
                "sequence_length": agent.config.sequence_length,
                "input_features": agent.config.input_features,
                "extractor_type": agent.config.extractor_type,
            },
            "observation_dim": self._get_observation_dim(agent.config),
            "action_type": "discrete",
            "num_actions": 6,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Checkpoint exported", path=str(output_path))
        
        return output_path
    
    def export_torchscript(
        self,
        agent: TradingAgent,
        output_path: str | Path,
        optimize: bool = True,
    ) -> Path:
        """
        Export model as TorchScript for optimized inference.
        
        Args:
            agent: Trained agent
            output_path: Output file path
            optimize: Whether to apply optimization passes
            
        Returns:
            Path to saved TorchScript model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if agent.model is None:
            raise ValueError("Agent has no model to export")
        
        # Get the policy network
        policy = agent.model.policy
        
        # Create wrapper for inference
        class InferenceWrapper(torch.nn.Module):
            def __init__(self, features_extractor, mlp_extractor, action_net, value_net):
                super().__init__()
                self.features_extractor = features_extractor
                self.mlp_extractor = mlp_extractor
                self.action_net = action_net
                self.value_net = value_net
            
            def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                features = self.features_extractor(obs)
                latent_pi, latent_vf = self.mlp_extractor(features)
                action_logits = self.action_net(latent_pi)
                value = self.value_net(latent_vf)
                return action_logits, value
            
            def predict(self, obs: torch.Tensor) -> torch.Tensor:
                action_logits, _ = self.forward(obs)
                return torch.argmax(action_logits, dim=-1)
        
        wrapper = InferenceWrapper(
            policy.features_extractor,
            policy.mlp_extractor,
            policy.action_net,
            policy.value_net,
        )
        
        wrapper.eval()
        
        # Create example input
        obs_dim = self._get_observation_dim(agent.config)
        example_input = torch.randn(1, obs_dim)
        
        # Trace the model
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example_input)
        
        # Optimize if requested
        if optimize:
            traced = torch.jit.optimize_for_inference(traced)
        
        # Save
        traced.save(str(output_path))
        
        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata = {
            "format": "torchscript",
            "observation_dim": obs_dim,
            "action_type": "discrete",
            "num_actions": 6,
            "optimized": optimize,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("TorchScript exported", path=str(output_path))
        
        return output_path
    
    def export_onnx(
        self,
        agent: TradingAgent,
        output_path: str | Path,
        opset_version: int = 14,
    ) -> Path:
        """
        Export model as ONNX for cross-platform deployment.
        
        Args:
            agent: Trained agent
            output_path: Output file path
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if agent.model is None:
            raise ValueError("Agent has no model to export")
        
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX export requires onnx package. Install with: pip install onnx")
        
        # Get the policy network
        policy = agent.model.policy
        
        # Create inference wrapper
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, features_extractor, mlp_extractor, action_net):
                super().__init__()
                self.features_extractor = features_extractor
                self.mlp_extractor = mlp_extractor
                self.action_net = action_net
            
            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                features = self.features_extractor(obs)
                latent_pi, _ = self.mlp_extractor(features)
                action_logits = self.action_net(latent_pi)
                return action_logits
        
        wrapper = ONNXWrapper(
            policy.features_extractor,
            policy.mlp_extractor,
            policy.action_net,
        )
        
        wrapper.eval()
        
        # Create example input
        obs_dim = self._get_observation_dim(agent.config)
        example_input = torch.randn(1, obs_dim)
        
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            example_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
        )
        
        # Verify the model
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        
        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata = {
            "format": "onnx",
            "opset_version": opset_version,
            "observation_dim": obs_dim,
            "action_type": "discrete",
            "num_actions": 6,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("ONNX exported", path=str(output_path))
        
        return output_path
    
    def _get_observation_dim(self, config: AgentConfig) -> int:
        """Calculate observation dimension from config."""
        seq_dim = config.sequence_length * config.input_features
        current_dim = config.input_features
        portfolio_dim = 4
        time_dim = 2
        return seq_dim + current_dim + portfolio_dim + time_dim
    
    def validate_export(
        self,
        model_path: str | Path,
        format: Literal["torchscript", "onnx"],
        test_input: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate an exported model.
        
        Args:
            model_path: Path to exported model
            format: Model format
            test_input: Optional test input
            
        Returns:
            Validation results
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            return {"valid": False, "error": "Model file not found"}
        
        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Create test input if not provided
        if test_input is None:
            obs_dim = metadata.get("observation_dim", 906)
            test_input = np.random.randn(1, obs_dim).astype(np.float32)
        
        try:
            if format == "torchscript":
                model = torch.jit.load(str(model_path))
                model.eval()
                
                with torch.no_grad():
                    output = model.predict(torch.from_numpy(test_input))
                
                return {
                    "valid": True,
                    "format": "torchscript",
                    "output_shape": list(output.shape),
                    "metadata": metadata,
                }
            
            elif format == "onnx":
                import onnxruntime as ort
                
                session = ort.InferenceSession(str(model_path))
                
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: test_input})
                
                return {
                    "valid": True,
                    "format": "onnx",
                    "output_shape": list(output[0].shape),
                    "metadata": metadata,
                }
            
            else:
                return {"valid": False, "error": f"Unknown format: {format}"}
                
        except Exception as e:
            return {"valid": False, "error": str(e)}


class InferenceEngine:
    """
    Lightweight inference engine for production deployment.
    
    Supports loading and running models in different formats.
    """
    
    def __init__(self, model_path: str | Path, format: str = "auto"):
        """
        Args:
            model_path: Path to model file
            format: Model format ("auto", "torchscript", "onnx")
        """
        self.model_path = Path(model_path)
        
        if format == "auto":
            if self.model_path.suffix == ".onnx":
                format = "onnx"
            else:
                format = "torchscript"
        
        self.format = format
        self._model: Any = None
        self._session: Any = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model."""
        if self.format == "torchscript":
            self._model = torch.jit.load(str(self.model_path))
            self._model.eval()
            
        elif self.format == "onnx":
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self.model_path))
        
        logger.info("Model loaded", path=str(self.model_path), format=self.format)
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict action from observation.
        
        Args:
            observation: Observation array
            
        Returns:
            Action index
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        if self.format == "torchscript":
            with torch.no_grad():
                tensor = torch.from_numpy(observation.astype(np.float32))
                action = self._model.predict(tensor)
                return int(action.item())
        
        elif self.format == "onnx":
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: observation.astype(np.float32)})
            return int(np.argmax(output[0]))
        
        raise ValueError(f"Unknown format: {self.format}")
    
    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict actions for a batch of observations.
        
        Args:
            observations: Batch of observations (N, obs_dim)
            
        Returns:
            Array of action indices (N,)
        """
        if self.format == "torchscript":
            with torch.no_grad():
                tensor = torch.from_numpy(observations.astype(np.float32))
                actions = self._model.predict(tensor)
                return actions.numpy()
        
        elif self.format == "onnx":
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: observations.astype(np.float32)})
            return np.argmax(output[0], axis=-1)
        
        raise ValueError(f"Unknown format: {self.format}")
