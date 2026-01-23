#!/usr/bin/env python3
"""
Train SAC on Dynamic Trading Environment.

Features:
- 4D action space (direction, size, hold, exit)
- Multi-step episodes within candles
- Risk-aware rewards (Sharpe + drawdown)
- Extended observation space (26 dims)

Fixes Applied (v6):
- Unbiased outcome generation (removed 0.8 multiplier)
- Realistic intra-candle price dynamics (mean-reversion phases)
- Recovery incentive for losing positions
- Symmetric DSR (no loss amplification)
- Enhanced wait rewards (entry timing optimization)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.simulation.deep_lob_dynamic_env import (
    DynamicTradingConfig,
    DeepLOBDynamicEnv,
)

console = Console()


class OutcomeDistributionCallback(BaseCallback):
    """Monitor outcome distribution to detect bias."""

    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.outcomes = []

    def _on_step(self) -> bool:
        # Collect outcomes from info dicts
        for info in self.locals.get('infos', []):
            if 'outcome' in info and info['outcome'] is not None:
                self.outcomes.append(info['outcome'])

        # Log distribution periodically
        if self.n_calls % self.log_freq == 0 and len(self.outcomes) > 100:
            up_count = self.outcomes.count(2)
            down_count = self.outcomes.count(0)
            hold_count = self.outcomes.count(1)
            total = len(self.outcomes)

            console.print(f"[dim]Outcome dist (n={total}): "
                         f"Up={up_count/total:.1%} Down={down_count/total:.1%} "
                         f"Hold={hold_count/total:.1%}[/dim]")

            # Reset for next window
            self.outcomes = []

        return True


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
    """Train SAC with dynamic entry/exit actions.

    This version (v6) includes fixes for:
    - Directional bias (unbiased outcome generation)
    - Poor loss recovery (symmetric DSR, recovery incentives)
    - Entry timing (enhanced wait rewards, realistic price dynamics)
    """

    console.print("[bold blue]SAC Dynamic Trading Training (v6 - Fixed)[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Timesteps: {total_timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print()

    console.print("[green]Fixes applied:[/green]")
    console.print("  ✓ Unbiased outcome generation")
    console.print("  ✓ Realistic intra-candle price dynamics")
    console.print("  ✓ Recovery incentive for losing positions")
    console.print("  ✓ Symmetric DSR (no loss amplification)")
    console.print("  ✓ Enhanced wait rewards")
    console.print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Config with dynamic trading - IMPROVED DEFAULTS
    config = DynamicTradingConfig(
        max_trades_per_candle=5,  # Allow multiple entries/exits
        steps_per_candle=60,  # Multi-step episodes
        sharpe_weight=0.2,  # Moderate Sharpe weight
        drawdown_penalty_weight=0.15,  # Moderate drawdown penalty
        use_dsr_reward=True,  # Enable symmetric DSR
        hold_reward=0.15,  # Increased to make waiting viable
        time_urgency_multiplier=2.0,  # Reward holding near settlement
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

    # Monitor outcome distribution for bias detection
    outcome_callback = OutcomeDistributionCallback(log_freq=10000)
    
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
    console.print(f"  Observation Space: 26D (with momentum, convergence, arbitrage)")
    console.print(f"  Steps per candle: {config.steps_per_candle}")
    console.print()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, outcome_callback],
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
