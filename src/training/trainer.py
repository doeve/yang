"""
Training orchestrator for the ML trading system.

Manages the full training lifecycle:
- Environment setup
- Training loop with callbacks
- Evaluation and validation
- Hyperparameter optimization
- Model checkpointing
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import structlog
from stable_baselines3.common.callbacks import CallbackList

from .callbacks import MetricsCallback, CheckpointCallback, EarlyStoppingCallback
from .evaluation import Evaluator, BacktestResult
from ..models.agent import TradingAgent, AgentConfig
from ..simulation.environment import PolymarketEnv, EnvConfig, make_vec_env
from ..data.storage import DataStorage

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training duration
    total_timesteps: int = 1_000_000
    eval_freq: int = 50_000  # Evaluate every 50k steps (was 10k - too frequent)
    n_eval_episodes: int = 2  # Fewer eval episodes for speed (was 5)
    
    # Checkpointing
    checkpoint_freq: int = 50_000
    save_best_only: bool = True
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_improvement: float = 0.01
    
    # Environment
    n_envs: int = 4
    env_config: EnvConfig | None = None
    
    # Agent
    agent_config: AgentConfig | None = None
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: list[dict[str, Any]] = field(default_factory=list)
    
    # Logging
    log_dir: str = "./logs"
    experiment_name: str = "training"
    
    # Reproducibility
    seed: int | None = 42


class TrainingOrchestrator:
    """
    Orchestrates the full training lifecycle.
    
    Handles:
    - Data loading and splitting
    - Environment creation
    - Training with callbacks
    - Evaluation on held-out data
    - Model saving and checkpointing
    - Hyperparameter optimization (optional)
    
    Example:
        orchestrator = TrainingOrchestrator(storage, config)
        result = orchestrator.train()
        
        # Access trained model
        agent = orchestrator.agent
    """
    
    def __init__(
        self,
        storage: DataStorage,
        config: TrainingConfig | None = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            storage: Data storage instance
            config: Training configuration
        """
        self.storage = storage
        self.config = config or TrainingConfig()
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Initialize components
        self.agent: TradingAgent | None = None
        self.evaluator = Evaluator()
        
        # Training state
        self._run_id: int | None = None
        self._start_time: datetime | None = None
        self._best_reward: float = float("-inf")
        self._training_history: list[dict[str, Any]] = []
        
        # Create directories
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("TrainingOrchestrator initialized", config=self.config)
    
    def load_data(
        self,
        market_ids: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        Load and split data for training.
        
        Args:
            market_ids: Optional list of specific markets to use
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if market_ids is None:
            market_ids = self.storage.get_available_markets_with_data()
        
        if not market_ids:
            raise ValueError("No market data available")
        
        logger.info("Loading data", num_markets=len(market_ids))
        
        # Load all market data
        all_data: dict[str, tuple[pd.DataFrame, dict]] = {}
        
        for market_id in market_ids:
            df = self.storage.load_price_data(market_id)  # Load any resolution
            if df is None or len(df) < 100:  # Skip very small datasets
                continue
            
            market = self.storage.get_market(market_id)
            metadata = {
                "resolution_at": market.resolution_at if market else None,
                "created_at": market.created_at if market else None,
                "outcome": market.outcome if market else None,
            }
            
            all_data[market_id] = (df, metadata)
        
        if not all_data:
            raise ValueError("No valid market data found")
        
        # Split by time within each market
        train_data: dict[str, Any] = {}
        val_data: dict[str, Any] = {}
        test_data: dict[str, Any] = {}
        
        for market_id, (df, metadata) in all_data.items():
            n = len(df)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            
            train_data[market_id] = (df.iloc[:train_end].copy(), metadata)
            val_data[market_id] = (df.iloc[train_end:val_end].copy(), metadata)
            test_data[market_id] = (df.iloc[val_end:].copy(), metadata)
        
        logger.info(
            "Data loaded and split",
            train_markets=len(train_data),
            val_markets=len(val_data),
            test_markets=len(test_data),
        )
        
        return train_data, val_data, test_data
    
    def create_env(
        self,
        market_data: dict[str, tuple[pd.DataFrame, dict]],
        n_envs: int = 1,
        normalize: bool = True,
    ) -> PolymarketEnv | Any:
        """
        Create environment(s) from market data.
        
        Args:
            market_data: Dict of market_id -> (df, metadata)
            n_envs: Number of parallel environments
            normalize: Whether to wrap with VecNormalize (recommended)
            
        Returns:
            Environment or vectorized environment
        """
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        
        env_config = self.config.env_config or EnvConfig()
        
        if n_envs == 1:
            # Single environment - use first market
            market_id = list(market_data.keys())[0]
            df, metadata = market_data[market_id]
            
            env = PolymarketEnv(
                df,
                config=env_config,
                market_id=market_id,
                resolution_at=metadata.get("resolution_at"),
                created_at=metadata.get("created_at"),
                outcome=metadata.get("outcome"),
            )
            
            if normalize:
                # Wrap in DummyVecEnv then VecNormalize
                env = DummyVecEnv([lambda: env])
                env = VecNormalize(
                    env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0,
                    gamma=0.99,
                )
            
            return env
        else:
            # Vectorized environment
            vec_env = make_vec_env(market_data, num_envs=n_envs, config=env_config)
            
            if normalize:
                vec_env = VecNormalize(
                    vec_env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0,
                    gamma=0.99,
                )
            
            return vec_env
    
    def train(
        self,
        market_ids: list[str] | None = None,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full training pipeline.
        
        Args:
            market_ids: Optional specific markets to use
            resume_from: Optional path to resume training from
            
        Returns:
            Dict with training results
        """
        self._start_time = datetime.now(timezone.utc)
        
        # Create training run in storage
        self._run_id = self.storage.create_training_run(
            name=f"{self.config.experiment_name}_{self._start_time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "total_timesteps": self.config.total_timesteps,
                "n_envs": self.config.n_envs,
                "agent_config": vars(self.config.agent_config) if self.config.agent_config else {},
            },
        )
        
        try:
            # Load and split data
            train_data, val_data, test_data = self.load_data(market_ids)
            
            # Create environments
            train_env = self.create_env(train_data, n_envs=self.config.n_envs)
            val_env = self.create_env(val_data, n_envs=1)
            
            # Initialize or load agent
            if resume_from:
                self.agent = TradingAgent.load(resume_from, env=train_env)
                logger.info("Resumed training from checkpoint", path=resume_from)
            else:
                agent_config = self.config.agent_config or AgentConfig()
                self.agent = TradingAgent(agent_config, env=train_env)
            
            # Setup callbacks
            callbacks = self._create_callbacks(val_env)
            
            # Train
            logger.info("Starting training", total_timesteps=self.config.total_timesteps)
            
            self.agent.train(
                env=train_env,
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
            )
            
            # Final evaluation on test set
            test_env = self.create_env(test_data, n_envs=1)
            test_results = self.evaluator.evaluate(
                self.agent,
                test_env,
                n_episodes=10,
            )
            
            # Backtest
            backtest_results = self._run_backtest(test_data)
            
            # Save final model
            final_model_path = self.log_dir / self.config.experiment_name / "final_model"
            self.agent.save(final_model_path)
            
            # Update training run
            self.storage.update_training_run(
                self._run_id,
                status="completed",
                final_reward=test_results["mean_reward"],
                best_reward=self._best_reward,
                model_path=str(final_model_path),
                metrics={
                    "test_results": test_results,
                    "backtest": backtest_results,
                    "training_history": self._training_history[-10:],  # Last 10 evals
                },
            )
            
            results = {
                "run_id": self._run_id,
                "test_results": test_results,
                "backtest_results": backtest_results,
                "best_reward": self._best_reward,
                "model_path": str(final_model_path),
                "training_duration": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
            }
            
            logger.info("Training completed", results=results)
            
            return results
            
        except Exception as e:
            logger.error("Training failed", error=str(e))
            
            if self._run_id:
                self.storage.update_training_run(
                    self._run_id,
                    status="failed",
                    metrics={"error": str(e)},
                )
            
            raise
    
    def _create_callbacks(self, val_env: Any) -> CallbackList:
        """Create training callbacks."""
        callbacks = []
        
        # Metrics callback
        metrics_callback = MetricsCallback(
            eval_env=val_env,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            log_path=self.log_dir / self.config.experiment_name / "metrics.json",
            on_eval=self._on_evaluation,
        )
        callbacks.append(metrics_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=str(self.log_dir / self.config.experiment_name / "checkpoints"),
            name_prefix="model",
            save_best_only=self.config.save_best_only,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.early_stopping:
            early_stopping = EarlyStoppingCallback(
                patience=self.config.patience,
                min_improvement=self.config.min_improvement,
            )
            callbacks.append(early_stopping)
        
        return CallbackList(callbacks)
    
    def _on_evaluation(
        self,
        mean_reward: float,
        std_reward: float,
        timesteps: int,
    ) -> None:
        """Callback for evaluation results."""
        self._training_history.append({
            "timesteps": timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            logger.info("New best reward", reward=mean_reward, timesteps=timesteps)
    
    def _run_backtest(
        self,
        test_data: dict[str, tuple[pd.DataFrame, dict]],
    ) -> dict[str, Any]:
        """Run backtest on test data."""
        if self.agent is None:
            raise ValueError("No agent to backtest")
        
        all_results = {}
        
        for market_id, (df, metadata) in test_data.items():
            if len(df) < 100:
                continue
            
            result = self.evaluator.backtest(
                self.agent,
                df,
                market_id=market_id,
                resolution_at=metadata.get("resolution_at"),
                created_at=metadata.get("created_at"),
                outcome=metadata.get("outcome"),
            )
            
            all_results[market_id] = result.to_dict()
        
        # Aggregate results
        if all_results:
            aggregate = {
                "total_pnl": sum(r["total_pnl"] for r in all_results.values()),
                "mean_sharpe": np.mean([r["sharpe_ratio"] for r in all_results.values()]),
                "max_drawdown": max(r["max_drawdown"] for r in all_results.values()),
                "win_rate": np.mean([r["win_rate"] for r in all_results.values()]),
                "num_markets": len(all_results),
            }
        else:
            aggregate = {}
        
        return {
            "per_market": all_results,
            "aggregate": aggregate,
        }
    
    def hyperparameter_search(
        self,
        param_space: dict[str, list[Any]],
        n_trials: int = 10,
        market_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run hyperparameter search.
        
        Uses random search over the parameter space.
        
        Args:
            param_space: Dict of parameter name -> list of values to try
            n_trials: Number of trials
            market_ids: Optional specific markets
            
        Returns:
            List of trial results sorted by reward
        """
        logger.info("Starting hyperparameter search", n_trials=n_trials)
        
        results = []
        
        for trial in range(n_trials):
            # Sample parameters
            params = {
                key: np.random.choice(values)
                for key, values in param_space.items()
            }
            
            logger.info(f"Trial {trial + 1}/{n_trials}", params=params)
            
            # Update config with sampled params
            config = TrainingConfig(
                total_timesteps=self.config.total_timesteps // 5,  # Shorter for search
                eval_freq=self.config.eval_freq,
                experiment_name=f"{self.config.experiment_name}_trial_{trial}",
                seed=self.config.seed + trial if self.config.seed else trial,
            )
            
            # Update agent config
            agent_config = AgentConfig(
                learning_rate=params.get("learning_rate", 3e-4),
                hidden_dim=params.get("hidden_dim", 128),
                features_dim=params.get("features_dim", 64),
                extractor_type=params.get("extractor_type", "lstm"),
            )
            config.agent_config = agent_config
            
            # Run training
            try:
                orchestrator = TrainingOrchestrator(self.storage, config)
                result = orchestrator.train(market_ids)
                
                results.append({
                    "params": params,
                    "reward": result["test_results"]["mean_reward"],
                    "sharpe": result["backtest_results"]["aggregate"].get("mean_sharpe", 0),
                    "trial": trial,
                })
            except Exception as e:
                logger.warning(f"Trial {trial} failed", error=str(e))
                results.append({
                    "params": params,
                    "reward": float("-inf"),
                    "sharpe": 0,
                    "trial": trial,
                    "error": str(e),
                })
        
        # Sort by reward
        results.sort(key=lambda x: x["reward"], reverse=True)
        
        logger.info("Hyperparameter search completed", best_reward=results[0]["reward"])
        
        return results
