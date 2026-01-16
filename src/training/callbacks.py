"""
Training callbacks for monitoring and control.

Provides callbacks for:
- Metrics logging and evaluation
- Model checkpointing
- Early stopping
- Learning rate scheduling
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import structlog
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

logger = structlog.get_logger(__name__)


class MetricsCallback(BaseCallback):
    """
    Callback for logging training metrics and running evaluation.
    
    Features:
    - Periodic evaluation on validation environment
    - Metrics logging to JSON file
    - Custom callback on evaluation
    """
    
    def __init__(
        self,
        eval_env: VecEnv | Any,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: str | Path | None = None,
        on_eval: Callable[[float, float, int], None] | None = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        """
        Args:
            eval_env: Evaluation environment
            eval_freq: Frequency of evaluation (in timesteps)
            n_eval_episodes: Number of evaluation episodes
            log_path: Path to save metrics log
            on_eval: Callback function (mean_reward, std_reward, timesteps)
            deterministic: Use deterministic policy for eval
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = Path(log_path) if log_path else None
        self.on_eval = on_eval
        self.deterministic = deterministic
        
        self.evaluations: list[dict[str, Any]] = []
        self.best_mean_reward = float("-inf")
        self._last_eval_step = 0
    
    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.n_calls
            self._evaluate()
        
        return True
    
    def _evaluate(self) -> None:
        """Run evaluation."""
        from stable_baselines3.common.evaluation import evaluate_policy
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=False,
        )
        
        # Create evaluation record
        eval_record = {
            "timesteps": self.num_timesteps,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        self.evaluations.append(eval_record)
        
        # Update best reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
        
        # Log
        if self.verbose >= 1:
            logger.info(
                "Evaluation",
                timesteps=self.num_timesteps,
                mean_reward=mean_reward,
                std_reward=std_reward,
                best_reward=self.best_mean_reward,
            )
        
        # Save to file
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(self.evaluations, f, indent=2)
        
        # Custom callback
        if self.on_eval:
            self.on_eval(mean_reward, std_reward, self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        # Final evaluation
        self._evaluate()


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints.
    
    Features:
    - Periodic checkpointing
    - Save best model only option
    - Naming with timesteps
    """
    
    def __init__(
        self,
        save_freq: int = 50000,
        save_path: str | Path = "./checkpoints",
        name_prefix: str = "model",
        save_best_only: bool = False,
        verbose: int = 1,
    ):
        """
        Args:
            save_freq: Frequency of saving (in timesteps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint names
            save_best_only: Only save when reward improves
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_best_only = save_best_only
        
        self._last_save_step = 0
        self._best_reward = float("-inf")
    
    def _init_callback(self) -> None:
        """Initialize callback."""
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls - self._last_save_step >= self.save_freq:
            self._last_save_step = self.n_calls
            
            if not self.save_best_only:
                self._save_checkpoint()
        
        return True
    
    def _save_checkpoint(self, suffix: str = "") -> None:
        """Save a checkpoint."""
        if suffix:
            name = f"{self.name_prefix}_{suffix}"
        else:
            name = f"{self.name_prefix}_{self.num_timesteps}"
        
        path = self.save_path / name
        self.model.save(str(path))
        
        if self.verbose >= 1:
            logger.info("Checkpoint saved", path=str(path))
    
    def update_best_reward(self, reward: float) -> None:
        """Update best reward and save if improved."""
        if reward > self._best_reward:
            self._best_reward = reward
            if self.save_best_only:
                self._save_checkpoint("best")


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on evaluation reward.
    
    Stops training if reward doesn't improve for `patience` evaluations.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_improvement: float = 0.01,
        verbose: int = 1,
    ):
        """
        Args:
            patience: Number of evaluations without improvement before stopping
            min_improvement: Minimum improvement to count as progress
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.patience = patience
        self.min_improvement = min_improvement
        
        self._best_reward = float("-inf")
        self._no_improvement_count = 0
    
    def update_reward(self, reward: float) -> None:
        """Update with new evaluation reward."""
        if reward > self._best_reward + self.min_improvement:
            self._best_reward = reward
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
            
            if self.verbose >= 1:
                logger.info(
                    "No improvement",
                    count=self._no_improvement_count,
                    patience=self.patience,
                )
    
    def _on_step(self) -> bool:
        """Check if should stop."""
        if self._no_improvement_count >= self.patience:
            if self.verbose >= 1:
                logger.info("Early stopping triggered")
            return False
        return True


class LearningRateScheduleCallback(BaseCallback):
    """
    Callback for learning rate scheduling.
    
    Supports:
    - Linear decay
    - Step decay
    - Cosine annealing
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        initial_lr: float = 3e-4,
        final_lr: float = 1e-5,
        decay_steps: int | None = None,
        step_size: int = 100000,
        step_gamma: float = 0.5,
        verbose: int = 1,
    ):
        """
        Args:
            schedule_type: One of "linear", "step", "cosine"
            initial_lr: Initial learning rate
            final_lr: Final learning rate (for linear/cosine)
            decay_steps: Total steps for decay (None = use total_timesteps)
            step_size: Step size for step decay
            step_gamma: Multiplicative factor for step decay
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps
        self.step_size = step_size
        self.step_gamma = step_gamma
        
        self._total_timesteps: int = 0
    
    def _init_callback(self) -> None:
        """Initialize with total timesteps."""
        if self.decay_steps is None:
            # Try to get from model
            self._total_timesteps = getattr(self.model, "_total_timesteps", 1000000)
        else:
            self._total_timesteps = self.decay_steps
    
    def _on_step(self) -> bool:
        """Update learning rate."""
        progress = self.num_timesteps / self._total_timesteps
        
        if self.schedule_type == "linear":
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        
        elif self.schedule_type == "step":
            num_steps = self.num_timesteps // self.step_size
            lr = self.initial_lr * (self.step_gamma ** num_steps)
        
        elif self.schedule_type == "cosine":
            lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (
                1 + np.cos(np.pi * progress)
            )
        
        else:
            lr = self.initial_lr
        
        # Update optimizer learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = lr
        
        return True


class TensorBoardCallback(BaseCallback):
    """
    Callback for logging to TensorBoard.
    
    Logs training metrics and custom scalars.
    """
    
    def __init__(
        self,
        log_dir: str | Path = "./tensorboard",
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        Args:
            log_dir: TensorBoard log directory
            log_freq: Logging frequency
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self._writer = None
    
    def _init_callback(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(self.log_dir))
        except ImportError:
            logger.warning("TensorBoard not available")
    
    def _on_step(self) -> bool:
        """Log metrics."""
        if self._writer is None:
            return True
        
        if self.n_calls % self.log_freq == 0:
            # Log rollout info if available
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                
                self._writer.add_scalar(
                    "rollout/ep_rew_mean",
                    np.mean(ep_rewards),
                    self.num_timesteps,
                )
                self._writer.add_scalar(
                    "rollout/ep_len_mean",
                    np.mean(ep_lengths),
                    self.num_timesteps,
                )
        
        return True
    
    def log_scalar(self, tag: str, value: float) -> None:
        """Log a custom scalar."""
        if self._writer:
            self._writer.add_scalar(tag, value, self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """Close writer."""
        if self._writer:
            self._writer.close()
