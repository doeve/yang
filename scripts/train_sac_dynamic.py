#!/usr/bin/env python3
"""
Train SAC on Dynamic Trading Environment.

Features:
- 4D action space (direction, size, hold, exit)
- Multi-step episodes within candles  
- Risk-aware rewards (Sharpe + drawdown)
- Extended observation space (20 dims)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.simulation.deep_lob_dynamic_env import (
    DynamicTradingConfig,
    DeepLOBDynamicEnv,
)

console = Console()


def make_env(config: DynamicTradingConfig):
    """Create environment factory."""
    def _init():
        return DeepLOBDynamicEnv(config)
    return _init


def train_dynamic_sac(
    output_dir: str = "./logs/sac_dynamic",
    total_timesteps: int = 500000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
):
    """Train SAC with dynamic entry/exit actions."""
    
    console.print("[bold blue]SAC Dynamic Trading Training[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Timesteps: {total_timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config with dynamic trading
    config = DynamicTradingConfig(
        max_trades_per_candle=5,  # Allow multiple entries/exits
        steps_per_candle=60,  # Multi-step episodes
        sharpe_weight=0.3,
        drawdown_penalty_weight=0.2,
        use_dsr_reward=True,
    )
    
    # Create vectorized environments
    console.print("[dim]Creating environments...[/dim]")
    envs = DummyVecEnv([make_env(config) for _ in range(n_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Eval environment
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=str(output_path / "checkpoints"),
        name_prefix="sac_dynamic",
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=20000 // n_envs,
        n_eval_episodes=20,
        deterministic=True,
    )
    
    # Create SAC model
    console.print("[dim]Creating SAC model...[/dim]")
    model = SAC(
        "MlpPolicy",
        envs,
        learning_rate=learning_rate,
        buffer_size=200000,  # Larger buffer for more experience
        learning_starts=2000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        policy_kwargs={
            "net_arch": [256, 256],  # Larger network
        },
        verbose=1,
        tensorboard_log=str(output_path / "tb_logs"),
    )
    
    console.print("[bold green]Starting training...[/bold green]")
    console.print(f"  Action Space: 4D [direction, size, hold, exit]")
    console.print(f"  Observation Space: 20D")
    console.print(f"  Steps per candle: {config.steps_per_candle}")
    console.print()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
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
    
    return str(final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC with dynamic entry/exit")
    parser.add_argument("--output", default="./logs/sac_dynamic", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train_dynamic_sac(
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
    )
