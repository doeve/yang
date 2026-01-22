#!/usr/bin/env python3
"""
Train SAC agent with DSR (Dynamic Semantic Reward) from Trade-R1 paper.

Uses DeepLOBExecutionEnv with:
- DSR reward formula (amplify logical wins, reduce noise-loss penalty)
- Almgren-Chriss cost model (spread + slippage)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.simulation.deep_lob_execution_env import (
    DeepLOBExecutionConfig,
    DeepLOBExecutionEnv,
)

console = Console()


def make_env(config: DeepLOBExecutionConfig):
    """Create environment factory."""
    def _init():
        return DeepLOBExecutionEnv(config)
    return _init


def train_sac_dsr(
    output_dir: str = "./logs/sac_dsr",
    total_timesteps: int = 200000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    use_dsr: bool = True,
):
    """Train SAC with DSR reward formula."""
    
    console.print("[bold blue]SAC Training with DSR Rewards[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Timesteps: {total_timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print(f"  DSR Enabled: {use_dsr}")
    console.print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config with DSR rewards
    config = DeepLOBExecutionConfig(
        use_dsr_reward=use_dsr,
        dsr_profit_multiplier_base=0.5,
        dsr_loss_multiplier_base=2.0,
        spread_cost=0.002,
        slippage_linear=0.001,
        slippage_quadratic=0.0005,
        pnl_scale=10.0,
        max_trades_per_candle=3,  # Allow multiple trades
    )
    
    # Create vectorized environments
    console.print("[dim]Creating environments...[/dim]")
    envs = DummyVecEnv([make_env(config) for _ in range(n_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path=str(output_path / "checkpoints"),
        name_prefix="sac_dsr",
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=10000 // n_envs,
        n_eval_episodes=20,
        deterministic=True,
    )
    
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
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        verbose=1,
        tensorboard_log=str(output_path / "tb_logs"),
    )
    
    console.print("[bold green]Starting training...[/bold green]")
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
    parser = argparse.ArgumentParser(description="Train SAC with DSR rewards")
    parser.add_argument("--output", default="./logs/sac_dsr", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-dsr", action="store_true", help="Disable DSR rewards")
    
    args = parser.parse_args()
    
    train_sac_dsr(
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
        use_dsr=not args.no_dsr,
    )
