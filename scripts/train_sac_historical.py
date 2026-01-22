#!/usr/bin/env python3
"""
Train SAC on Historical Polymarket Data.

Uses real BTC 15min candle outcomes with:
- Normalized rewards [-1, 1]
- Single-step episodes (one decision per candle)
- Proper risk-adjusted cost model
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.simulation.historical_trading_env import (
    HistoricalTradingConfig,
    HistoricalTradingEnv,
)

console = Console()


class WinRateCallback(BaseCallback):
    """Log win rate during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._wins = 0
        self._trades = 0
    
    def _on_step(self):
        # Check for episode end
        for info in self.locals.get("infos", []):
            action = info.get("action_taken", "")
            if action == "win":
                self._wins += 1
                self._trades += 1
            elif action == "loss":
                self._trades += 1
        
        # Log every 10k steps
        if self.num_timesteps % 10000 == 0 and self._trades > 0:
            win_rate = self._wins / self._trades
            console.print(
                f"[dim]Step {self.num_timesteps:,} | "
                f"Trades: {self._trades} | Win Rate: {win_rate:.1%}[/dim]"
            )
        
        return True


def make_env(config: HistoricalTradingConfig):
    """Create environment factory."""
    def _init():
        return HistoricalTradingEnv(config)
    return _init


def train_historical_sac(
    output_dir: str = "./logs/sac_historical",
    total_timesteps: int = 500000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
):
    """Train SAC on historical Polymarket data."""
    
    console.print("[bold blue]SAC Historical Trading Training[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Timesteps: {total_timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config for historical data
    config = HistoricalTradingConfig(
        data_path="./data/polymarket/btc_15min_candles.parquet",
        max_position_size=0.25,
        spread_cost=0.02,
        slippage=0.01,
        win_reward=1.0,
        loss_penalty=-1.0,
        hold_reward=0.0,
        wrong_direction_penalty=-0.5,
        confidence_bonus_scale=0.2,
    )
    
    # Create vectorized environments
    console.print("[dim]Creating environments...[/dim]")
    envs = DummyVecEnv([make_env(config) for _ in range(n_envs)])
    
    # Normalize observations but NOT rewards (already normalized)
    envs = VecNormalize(
        envs,
        norm_obs=True,
        norm_reward=False,  # Keep rewards in [-1, 1]
        clip_obs=10.0,
        clip_reward=5.0,
    )
    
    # Eval environment
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path=str(output_path / "checkpoints"),
        name_prefix="sac_historical",
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=10000 // n_envs,
        n_eval_episodes=50,  # More episodes for stable metrics
        deterministic=True,
    )
    
    win_rate_callback = WinRateCallback()
    
    # Create SAC model
    console.print("[dim]Creating SAC model...[/dim]")
    model = SAC(
        "MlpPolicy",
        envs,
        learning_rate=learning_rate,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,  # High discount for single-step
        train_freq=1,
        gradient_steps=2,  # More updates per step
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        policy_kwargs={
            "net_arch": [128, 128],  # Smaller network for simpler task
        },
        verbose=1,
        tensorboard_log=str(output_path / "tb_logs"),
    )
    
    console.print("[bold green]Starting training...[/bold green]")
    console.print(f"  Action Space: 3D [direction, size, hold_prob]")
    console.print(f"  Observation Space: 15D")
    console.print(f"  Reward Range: [-2, 2] (normalized)")
    console.print(f"  Training Candles: ~670")
    console.print()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, win_rate_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_path = output_path / "final_model"
    model.save(str(final_path))
    envs.save(str(output_path / "vecnormalize.pkl"))
    
    console.print()
    console.print(f"[bold green]✓ Training complete[/bold green]")
    console.print(f"  Model saved to: {final_path}")
    console.print(f"  VecNormalize saved to: {output_path / 'vecnormalize.pkl'}")
    
    # Final evaluation
    console.print()
    console.print("[bold]Final Evaluation:[/bold]")
    
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=100, deterministic=True
    )
    console.print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    
    # Estimate win rate from reward
    # win_reward ~= 1.0, loss_penalty ~= -1.0
    estimated_win_rate = (mean_reward + 1) / 2
    console.print(f"  Estimated Win Rate: {estimated_win_rate:.1%}")
    
    return str(final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on historical data")
    parser.add_argument("--output", default="./logs/sac_historical", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train_historical_sac(
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
    )
