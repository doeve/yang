#!/usr/bin/env python3
"""
Train SAC with K-Fold Temporal Cross-Validation.

Uses real DeepLOB predictions (if available) and proper temporal splits
to ensure unbiased training without look-ahead bias.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from src.simulation.historical_trading_env import (
    HistoricalTradingConfig,
    HistoricalTradingEnv,
    make_historical_vec_env,
)
from src.simulation.temporal_cv import TemporalKFold, print_fold_info

console = Console()


def make_env_for_fold(
    config: HistoricalTradingConfig,
    indices: np.ndarray,
    num_envs: int = 4,
):
    """Create vectorized environment for a specific fold."""
    return make_historical_vec_env(
        num_envs=num_envs,
        config=config,
        candle_indices=indices,
    )


def train_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: HistoricalTradingConfig,
    output_dir: Path,
    timesteps: int = 100000,
    n_envs: int = 4,
) -> Dict[str, Any]:
    """Train SAC on a single fold and evaluate."""
    
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold blue]Fold {fold_idx + 1}[/bold blue]")
    console.print(f"  Train: {len(train_indices)} candles [0:{train_indices[-1]+1}]")
    console.print(f"  Val: {len(val_indices)} candles [{val_indices[0]}:{val_indices[-1]+1}]")
    
    # Create environments
    train_env = make_env_for_fold(config, train_indices, n_envs)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    
    val_env = make_env_for_fold(config, val_indices, 1)
    val_env = VecNormalize(
        val_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    
    # Eval callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(fold_dir / "best"),
        log_path=str(fold_dir / "eval_logs"),
        eval_freq=10000 // n_envs,
        n_eval_episodes=min(50, len(val_indices)),
        deterministic=True,
        verbose=0,
    )
    
    # Create and train SAC
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=500,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto",
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        tensorboard_log=str(fold_dir / "tb_logs"),
    )
    
    console.print(f"  Training for {timesteps:,} steps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )
    
    # Evaluate on validation set
    mean_reward, std_reward = evaluate_policy(
        model, val_env, n_eval_episodes=min(100, len(val_indices)), deterministic=True
    )
    
    # Save model
    model.save(str(fold_dir / "final_model"))
    train_env.save(str(fold_dir / "vecnormalize.pkl"))
    
    # Calculate metrics
    estimated_win_rate = (mean_reward + 1) / 2  # Assuming normalized rewards
    
    result = {
        "fold": fold_idx,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "val_reward": mean_reward,
        "val_std": std_reward,
        "estimated_win_rate": estimated_win_rate,
    }
    
    console.print(f"  [green]Val Reward: {mean_reward:.3f} ± {std_reward:.3f}[/green]")
    console.print(f"  Estimated Win Rate: {estimated_win_rate:.1%}")
    
    return result


def train_with_cv(
    data_path: str = "./data/polymarket/btc_15min_candles.parquet",
    predictions_path: str = "",
    output_dir: str = "./logs/sac_cv",
    n_folds: int = 5,
    timesteps_per_fold: int = 100000,
    n_envs: int = 4,
):
    """Train SAC with k-fold temporal cross-validation."""
    
    console.print("[bold blue]SAC K-Fold Temporal Cross-Validation[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Folds: {n_folds}")
    console.print(f"  Timesteps per fold: {timesteps_per_fold:,}")
    console.print()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data to get size
    config = HistoricalTradingConfig(
        data_path=data_path,
        predictions_path=predictions_path,
    )
    
    if predictions_path and Path(predictions_path).exists():
        data = pd.read_parquet(predictions_path)
        console.print(f"[green]✓ Using real predictions from {predictions_path}[/green]")
    else:
        data = pd.read_parquet(data_path)
        data = data[data["closed"] == True]
        console.print("[yellow]⚠ Using synthetic predictions[/yellow]")
    
    console.print(f"  Total candles: {len(data)}")
    
    # Create temporal k-fold splitter
    cv = TemporalKFold(n_splits=n_folds, val_ratio=0.25, min_train_size=50)
    
    # Print fold structure
    console.print("\n[bold]Fold Structure:[/bold]")
    print_fold_info(data, cv, timestamp_col="timestamp")
    
    # Train each fold
    results: List[Dict[str, Any]] = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data)):
        result = train_fold(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            config=config,
            output_dir=output_path,
            timesteps=timesteps_per_fold,
            n_envs=n_envs,
        )
        results.append(result)
    
    # Summary table
    console.print("\n[bold]Cross-Validation Results:[/bold]")
    table = Table(show_header=True)
    table.add_column("Fold", style="cyan")
    table.add_column("Train", style="dim")
    table.add_column("Val", style="dim")
    table.add_column("Reward", style="green")
    table.add_column("Win Rate", style="yellow")
    
    for r in results:
        table.add_row(
            str(r["fold"] + 1),
            str(r["train_size"]),
            str(r["val_size"]),
            f"{r['val_reward']:.3f} ± {r['val_std']:.2f}",
            f"{r['estimated_win_rate']:.1%}",
        )
    
    console.print(table)
    
    # Overall metrics
    rewards = [r["val_reward"] for r in results]
    win_rates = [r["estimated_win_rate"] for r in results]
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Mean Val Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    console.print(f"  Mean Win Rate: {np.mean(win_rates):.1%}")
    console.print(f"  Fold Variance: {np.std(rewards):.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "cv_results.csv", index=False)
    console.print(f"\n[green]✓ Results saved to {output_path / 'cv_results.csv'}[/green]")
    
    # Identify best fold
    best_fold = max(results, key=lambda x: x["val_reward"])
    console.print(f"\n[bold green]Best Fold: {best_fold['fold'] + 1}[/bold green]")
    best_fold_idx = best_fold["fold"]
    console.print(f"  Model: {output_path / f'fold_{best_fold_idx}' / 'best' / 'best_model.zip'}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC with temporal cross-validation")
    parser.add_argument("--data", default="./data/polymarket/btc_15min_candles.parquet")
    parser.add_argument("--predictions", default="", help="Pre-generated predictions dataset")
    parser.add_argument("--output", default="./logs/sac_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--envs", type=int, default=4)
    
    args = parser.parse_args()
    
    train_with_cv(
        data_path=args.data,
        predictions_path=args.predictions,
        output_dir=args.output,
        n_folds=args.folds,
        timesteps_per_fold=args.timesteps,
        n_envs=args.envs,
    )
