"""
Trading agent wrapper for RL training.

Provides a high-level interface for training and using RL agents:
- PPO-based training with custom temporal features
- Model checkpointing and loading
- Evaluation and backtesting
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import structlog
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from .policy import TradingPolicy, get_policy_kwargs
from ..simulation.environment import PolymarketEnv, EnvConfig

logger = structlog.get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the trading agent.
    
    Hyperparameters tuned based on RL trading research:
    - Lower learning rate (1e-4) for stability
    - Higher entropy coefficient for exploration
    - Appropriate batch size for trading
    """
    
    # Model architecture (PURE ML - matches d3v)
    features_dim: int = 64
    hidden_dim: int = 128
    sequence_length: int = 300  # Matches d3v historyLength
    input_features: int = 2  # Pure ML: price + volume/orderflow
    extractor_type: Literal["lstm", "transformer", "hybrid"] = "lstm"
    
    # PPO hyperparameters (tuned for trading on high-RAM server)
    learning_rate: float = 1e-4  # Lower for stability
    n_steps: int = 2048  # Rollout steps per env (larger = better gradients)
    batch_size: int = 256  # Larger batch for 62GB RAM server
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05  # Higher entropy for more exploration (was 0.01)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture (wider networks)
    policy_net: list[int] = None  # type: ignore
    value_net: list[int] = None  # type: ignore
    
    # Training settings
    device: str = "auto"
    seed: int | None = None
    verbose: int = 1
    
    def __post_init__(self):
        if self.policy_net is None:
            self.policy_net = [256, 128]  # Wider networks (was [128, 64])
        if self.value_net is None:
            self.value_net = [256, 128]


class TradingAgent:
    """
    High-level trading agent for Polymarket.
    
    Wraps a PPO model with custom temporal feature extractors
    and provides methods for training, evaluation, and inference.
    
    Example:
        agent = TradingAgent(config)
        agent.train(env, total_timesteps=100000)
        
        # Inference
        action = agent.predict(observation)
        
        # Save/load
        agent.save("model_path")
        agent = TradingAgent.load("model_path")
    """
    
    def __init__(
        self,
        config: AgentConfig | None = None,
        env: PolymarketEnv | VecEnv | None = None,
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            env: Optional environment for initialization
        """
        self.config = config or AgentConfig()
        self.model: PPO | None = None
        self._env = env
        
        if env is not None:
            self._initialize_model(env)
    
    def _initialize_model(self, env: PolymarketEnv | VecEnv) -> None:
        """Initialize the PPO model with custom policy."""
        # Get policy kwargs
        policy_kwargs = get_policy_kwargs(
            features_dim=self.config.features_dim,
            sequence_length=self.config.sequence_length,
            input_features=self.config.input_features,
            extractor_type=self.config.extractor_type,
            hidden_dim=self.config.hidden_dim,
            net_arch=[dict(
                pi=self.config.policy_net,
                vf=self.config.value_net,
            )],
        )
        
        self.model = PPO(
            policy=TradingPolicy,
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=self.config.device,
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        
        logger.info(
            "Agent initialized",
            extractor_type=self.config.extractor_type,
            features_dim=self.config.features_dim,
        )
    
    def train(
        self,
        env: PolymarketEnv | VecEnv | None = None,
        total_timesteps: int = 100_000,
        callback: BaseCallback | list[BaseCallback] | None = None,
        log_interval: int = 10,
        progress_bar: bool = True,
    ) -> "TradingAgent":
        """
        Train the agent.
        
        Args:
            env: Training environment (uses stored env if None)
            total_timesteps: Total timesteps to train for
            callback: Training callbacks
            log_interval: Logging interval in episodes
            progress_bar: Whether to show progress bar
            
        Returns:
            self for chaining
        """
        if env is not None:
            if self.model is None:
                self._initialize_model(env)
            else:
                self.model.set_env(env)
        
        if self.model is None:
            raise ValueError("No environment provided for training")
        
        logger.info("Starting training", total_timesteps=total_timesteps)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )
        
        logger.info("Training completed")
        
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        state: Any | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray | int, Any]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            state: Optional recurrent state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, next_state)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        action, state = self.model.predict(
            observation,
            state=state,
            deterministic=deterministic,
        )
        
        return action, state
    
    def evaluate(
        self,
        env: PolymarketEnv | VecEnv,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate the agent on an environment.
        
        Args:
            env: Evaluation environment
            n_eval_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        from stable_baselines3.common.evaluation import evaluate_policy
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=False,
        )
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_episodes": n_eval_episodes,
        }
    
    def save(self, path: str | Path) -> None:
        """
        Save the agent to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(path))
        
        # Save config
        import json
        config_path = path.parent / f"{path.stem}_config.json"
        
        config_dict = {
            "features_dim": self.config.features_dim,
            "hidden_dim": self.config.hidden_dim,
            "sequence_length": self.config.sequence_length,
            "input_features": self.config.input_features,
            "extractor_type": self.config.extractor_type,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "policy_net": self.config.policy_net,
            "value_net": self.config.value_net,
        }
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("Agent saved", path=str(path))
    
    @classmethod
    def load(
        cls,
        path: str | Path,
        env: PolymarketEnv | VecEnv | None = None,
        device: str = "auto",
    ) -> "TradingAgent":
        """
        Load an agent from disk.
        
        Args:
            path: Path to the saved model
            env: Optional environment for the loaded model
            device: Device to load model on
            
        Returns:
            Loaded agent
        """
        path = Path(path)
        
        # Load config if exists
        config_path = path.parent / f"{path.stem}_config.json"
        
        config = AgentConfig()
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = AgentConfig(
                features_dim=config_dict.get("features_dim", 64),
                hidden_dim=config_dict.get("hidden_dim", 128),
                sequence_length=config_dict.get("sequence_length", 60),
                input_features=config_dict.get("input_features", 15),
                extractor_type=config_dict.get("extractor_type", "lstm"),
                policy_net=config_dict.get("policy_net", [128, 64]),
                value_net=config_dict.get("value_net", [128, 64]),
            )
        
        # Create agent
        agent = cls(config)
        
        # Load model with custom objects
        custom_objects = {
            "policy_class": TradingPolicy,
        }
        
        agent.model = PPO.load(
            str(path),
            env=env,
            device=device,
            custom_objects=custom_objects,
        )
        
        logger.info("Agent loaded", path=str(path))
        
        return agent
    
    def get_policy_state_dict(self) -> dict[str, torch.Tensor]:
        """Get the policy network state dict."""
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.policy.state_dict()
    
    def set_policy_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Set the policy network state dict."""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.policy.load_state_dict(state_dict)


class EnsembleAgent:
    """
    Ensemble of multiple trading agents.
    
    Combines predictions from multiple agents for more robust trading.
    """
    
    def __init__(self, agents: list[TradingAgent]):
        """
        Args:
            agents: List of trading agents
        """
        self.agents = agents
    
    def predict(
        self,
        observation: np.ndarray,
        voting: Literal["majority", "average"] = "majority",
        deterministic: bool = True,
    ) -> tuple[np.ndarray | int, None]:
        """
        Predict action using ensemble voting.
        
        Args:
            observation: Current observation
            voting: Voting method ("majority" or "average")
            deterministic: Whether to use deterministic policy
            
        Returns:
            Ensemble action
        """
        actions = []
        for agent in self.agents:
            action, _ = agent.predict(observation, deterministic=deterministic)
            actions.append(action)
        
        if voting == "majority":
            # Majority voting for discrete actions
            from collections import Counter
            counter = Counter(actions)
            return counter.most_common(1)[0][0], None
        else:
            # Average for continuous actions
            return np.mean(actions, axis=0), None
    
    def save(self, path: str | Path) -> None:
        """Save all agents in ensemble."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent.save(path / f"agent_{i}")
    
    @classmethod
    def load(cls, path: str | Path, env: Any = None) -> "EnsembleAgent":
        """Load ensemble from directory."""
        path = Path(path)
        
        agents = []
        for agent_path in sorted(path.glob("agent_*.zip")):
            agent = TradingAgent.load(agent_path, env=env)
            agents.append(agent)
        
        return cls(agents)
