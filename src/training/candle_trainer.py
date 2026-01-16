"""
Specialized trainer for binary candle prediction.

Tracks accuracy metrics in addition to reward, and uses
the CandlePredictionEnv for training.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from ..simulation.candle_env import CandlePredictionEnv, CandleEnvConfig, make_candle_vec_env

logger = structlog.get_logger(__name__)


@dataclass
class CandleTrainingConfig:
    """Configuration for candle prediction training."""
    
    # Training
    total_timesteps: int = 500_000
    n_envs: int = 24  # Match server cores
    
    # PPO settings optimized for discrete actions
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01  # Lower entropy for simpler action space
    
    # Evaluation
    eval_freq: int = 50_000
    
    # Output
    log_dir: str = "./logs"
    experiment_name: str = "candle_prediction"
    
    # Environment
    candle_minutes: int = 15
    price_history_length: int = 300


class AccuracyCallback(BaseCallback):
    """Tracks prediction accuracy during training."""
    
    def __init__(self, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.predictions: list[bool] = []
        self.bet_timings: list[float] = []
        
    def _on_step(self) -> bool:
        # Check for episode ends
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if info.get("bet_placed") is not None:
                    # Episode ended with a bet
                    predicted = info.get("bet_placed")
                    actual = info.get("candle_direction")
                    
                    if predicted and actual:
                        correct = (
                            (predicted == "BET_UP" and actual == "UP") or
                            (predicted == "BET_DOWN" and actual == "DOWN")
                        )
                        self.predictions.append(correct)
                        self.bet_timings.append(info.get("time_progress", 0.5))
        
        # Log accuracy periodically
        if self.n_calls % self.eval_freq == 0 and len(self.predictions) > 10:
            accuracy = np.mean(self.predictions[-100:])  # Last 100 predictions
            avg_timing = np.mean(self.bet_timings[-100:])
            
            logger.info(
                "Prediction accuracy",
                accuracy=f"{accuracy:.1%}",
                avg_bet_timing=f"{avg_timing:.1%}",
                total_predictions=len(self.predictions),
                timesteps=self.num_timesteps,
            )
        
        return True


class CandleTrainer:
    """
    Trainer specialized for binary candle prediction.
    
    Uses PPO with discrete action space and tracks
    accuracy metrics in addition to reward.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        config: CandleTrainingConfig | None = None,
    ):
        self.price_data = price_data
        self.config = config or CandleTrainingConfig()
        
        self.log_dir = Path(self.config.log_dir) / self.config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: PPO | None = None
        
        logger.info(
            "CandleTrainer initialized",
            data_points=len(price_data),
            config=self.config,
        )
    
    def train(self) -> dict[str, Any]:
        """Run training and return results."""
        start_time = datetime.now(timezone.utc)
        
        # Create environments
        env_config = CandleEnvConfig(
            candle_minutes=self.config.candle_minutes,
            price_history_length=self.config.price_history_length,
        )
        
        logger.info("Creating training environments", n_envs=self.config.n_envs)
        
        train_env = make_candle_vec_env(
            self.price_data,
            num_envs=self.config.n_envs,
            config=env_config,
        )
        
        # Wrap with normalization
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        
        # Create PPO model
        logger.info("Creating PPO model")
        
        self.model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            verbose=1,
        )
        
        # Setup callbacks
        accuracy_callback = AccuracyCallback(eval_freq=self.config.eval_freq)
        
        # Train
        logger.info("Starting training", total_timesteps=self.config.total_timesteps)
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=CallbackList([accuracy_callback]),
            progress_bar=True,
        )
        
        # Save model
        model_path = self.log_dir / "candle_model"
        self.model.save(str(model_path))
        train_env.save(str(self.log_dir / "vec_normalize.pkl"))
        
        # Compute final metrics
        final_accuracy = np.mean(accuracy_callback.predictions[-100:]) if accuracy_callback.predictions else 0.0
        avg_bet_timing = np.mean(accuracy_callback.bet_timings[-100:]) if accuracy_callback.bet_timings else 0.5
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        results = {
            "model_path": str(model_path),
            "final_accuracy": final_accuracy,
            "avg_bet_timing": avg_bet_timing,
            "total_predictions": len(accuracy_callback.predictions),
            "training_duration_seconds": duration,
        }
        
        logger.info("Training complete", **results)
        
        return results
    
    def evaluate(self, n_candles: int = 100) -> dict[str, Any]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("No model to evaluate. Train first.")
        
        env_config = CandleEnvConfig(
            candle_minutes=self.config.candle_minutes,
            price_history_length=self.config.price_history_length,
            random_start=True,
        )
        
        env = CandlePredictionEnv(self.price_data, config=env_config)
        
        correct = 0
        total = 0
        bet_timings = []
        
        for _ in range(n_candles):
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            
            if info.get("bet_placed"):
                total += 1
                predicted = info["bet_placed"]
                actual = info["candle_direction"]
                
                if (predicted == "BET_UP" and actual == "UP") or \
                   (predicted == "BET_DOWN" and actual == "DOWN"):
                    correct += 1
                
                bet_timings.append(info.get("time_progress", 0.5))
        
        accuracy = correct / max(total, 1)
        
        return {
            "accuracy": accuracy,
            "total_candles": n_candles,
            "bets_placed": total,
            "correct_predictions": correct,
            "avg_bet_timing": np.mean(bet_timings) if bet_timings else 0.5,
        }
