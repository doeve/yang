"""
Trainer for advanced candle prediction with position sizing.

Uses SAC (Soft Actor-Critic) which is better suited for
continuous action spaces than PPO.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from ..simulation.advanced_candle_env import (
    AdvancedCandleEnv,
    AdvancedCandleConfig,
    make_advanced_vec_env,
)

logger = structlog.get_logger(__name__)


@dataclass
class AdvancedTrainingConfig:
    """Configuration for advanced training."""
    
    total_timesteps: int = 500_000
    n_envs: int = 8  # SAC works better with fewer envs
    
    # SAC settings
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    tau: float = 0.005  # Soft update coefficient
    gamma: float = 0.99
    
    # Evaluation
    eval_freq: int = 25_000
    
    # Output
    log_dir: str = "./logs"
    experiment_name: str = "advanced_candle"
    
    # Environment
    candle_minutes: int = 15
    initial_balance: float = 1000.0


class BalanceCallback(BaseCallback):
    """Tracks balance and trading metrics during training."""
    
    def __init__(self, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.balances: list[float] = []
        self.position_sizes: list[float] = []
        self.correct_trades: int = 0
        self.total_trades: int = 0
        
    def _on_step(self) -> bool:
        if "infos" in self.locals:
            dones = self.locals.get("dones", [])
            for i, info in enumerate(self.locals["infos"]):
                if i < len(dones) and dones[i]:
                    if info.get("position_taken"):
                        self.total_trades += 1
                        self.balances.append(info.get("balance", 1000))
                        self.position_sizes.append(info.get("position_size", 0))
                        
                        # Check if trade was correct
                        direction = info.get("position_direction")
                        candle_ret = info.get("candle_return", 0)
                        if direction == "UP" and candle_ret > 0:
                            self.correct_trades += 1
                        elif direction == "DOWN" and candle_ret < 0:
                            self.correct_trades += 1
        
        if self.n_calls % self.eval_freq == 0 and self.total_trades > 10:
            accuracy = self.correct_trades / self.total_trades
            avg_balance = np.mean(self.balances[-100:]) if self.balances else 1000
            avg_size = np.mean(self.position_sizes[-100:]) if self.position_sizes else 0
            
            logger.info(
                "Training progress",
                accuracy=f"{accuracy:.1%}",
                avg_balance=f"${avg_balance:.2f}",
                avg_position_size=f"{avg_size:.2%}",
                total_trades=self.total_trades,
                timesteps=self.num_timesteps,
            )
        
        return True


class AdvancedTrainer:
    """Trainer for position-sized candle prediction."""
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        config: AdvancedTrainingConfig | None = None,
    ):
        self.price_data = price_data
        self.config = config or AdvancedTrainingConfig()
        
        self.log_dir = Path(self.config.log_dir) / self.config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: SAC | None = None
        
        logger.info(
            "AdvancedTrainer initialized",
            data_points=len(price_data),
            config=self.config,
        )
    
    def train(self) -> dict[str, Any]:
        """Run training."""
        start_time = datetime.now(timezone.utc)
        
        env_config = AdvancedCandleConfig(
            candle_minutes=self.config.candle_minutes,
            initial_balance=self.config.initial_balance,
        )
        
        logger.info("Creating training environments", n_envs=self.config.n_envs)
        
        # SAC doesn't work well with SubprocVecEnv, use DummyVecEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        def make_env():
            return AdvancedCandleEnv(self.price_data, config=env_config)
        
        train_env = DummyVecEnv([make_env for _ in range(self.config.n_envs)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        logger.info("Creating SAC model")
        
        self.model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            verbose=1,
        )
        
        balance_callback = BalanceCallback(eval_freq=self.config.eval_freq)
        
        logger.info("Starting training", total_timesteps=self.config.total_timesteps)
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=CallbackList([balance_callback]),
            progress_bar=True,
        )
        
        # Save
        model_path = self.log_dir / "advanced_model"
        self.model.save(str(model_path))
        train_env.save(str(self.log_dir / "vec_normalize.pkl"))
        
        # Metrics
        final_accuracy = balance_callback.correct_trades / max(balance_callback.total_trades, 1)
        avg_balance = np.mean(balance_callback.balances[-100:]) if balance_callback.balances else 1000
        avg_size = np.mean(balance_callback.position_sizes[-100:]) if balance_callback.position_sizes else 0
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        results = {
            "model_path": str(model_path),
            "final_accuracy": final_accuracy,
            "avg_final_balance": avg_balance,
            "avg_position_size": avg_size,
            "total_trades": balance_callback.total_trades,
            "training_duration_seconds": duration,
        }
        
        logger.info("Training complete", **results)
        return results
