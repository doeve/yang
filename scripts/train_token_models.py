#!/usr/bin/env python3
"""
Train Token-Centric Models for Polymarket Trading.

This script trains:
1. EdgeDetector: Predicts P(YES wins) to identify mispricing
2. SAC Policy: Learns optimal entry/exit timing on token prices

Usage:
    # Train edge detector only
    python scripts/train_token_models.py --edge-only --epochs 100

    # Train SAC only (assumes edge detector exists)
    python scripts/train_token_models.py --sac-only --timesteps 500000

    # Train both
    python scripts/train_token_models.py --epochs 100 --timesteps 500000
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def train_edge_detector(
    output_dir: str = "./logs/edge_detector_v1",
    n_candles: int = 3000,
    epochs: int = 100,
    batch_size: int = 64,
    btc_data_path: str = None,
):
    """Train the edge detector model."""
    from src.data.token_dataset import TokenDatasetBuilder
    from src.models.edge_detector import EdgeDetectorConfig, EdgeDetectorTrainer

    console.print("[bold blue]Training Edge Detector Model[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Simulated candles: {n_candles}")
    console.print(f"  Epochs: {epochs}")
    console.print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load BTC data if available
    btc_df = None
    if btc_data_path and Path(btc_data_path).exists():
        console.print(f"Loading BTC data from {btc_data_path}...")
        btc_df = pd.read_parquet(btc_data_path)
        if 'close' in btc_df.columns:
            btc_df['price'] = btc_df['close']
        console.print(f"  Loaded {len(btc_df)} BTC price points")

    # Generate training data
    console.print("Generating training data...")
    builder = TokenDatasetBuilder()
    token_df = builder.prepare_training_data(
        n_simulated_candles=n_candles,
        btc_df=btc_df,
        use_simulation=True,
    )
    console.print(f"  Generated {len(token_df)} data points")

    # Create edge detector dataset
    console.print("Creating feature dataset...")
    dataset = builder.create_edge_detector_dataset(
        token_df,
        btc_df=btc_df,
        samples_per_candle=5,
        test_ratio=0.2,
    )

    X_train, y_train = dataset['train']
    X_test, y_test = dataset['test']

    # Split train into train/val
    val_split = int(len(X_train) * 0.85)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    console.print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train model
    config = EdgeDetectorConfig(
        input_dim=X_train.shape[1],
        hidden_dims=(128, 64, 32),
        dropout=0.3,
    )

    trainer = EdgeDetectorTrainer(config)
    model, history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    # Evaluate on test set
    console.print("\n[bold]Test Set Evaluation:[/bold]")
    results = trainer.evaluate(model, X_test, y_test)
    console.print(f"  Accuracy: {results['accuracy']:.3f}")
    console.print(f"  Brier Score: {results['brier_score']:.4f}")
    console.print(f"  Calibration Error: {results['calibration_error']:.4f}")

    console.print(f"\n[bold green]Edge detector saved to {output_dir}[/bold green]")

    return model, results


def train_sac_policy(
    output_dir: str = "./logs/token_sac_v1",
    total_timesteps: int = 500000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
):
    """Train SAC policy on token trading environment."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.simulation.token_trading_env import TokenTradingEnv, TokenTradingConfig

    console.print("[bold blue]Training SAC Policy on Token Trading[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Timesteps: {total_timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    config = TokenTradingConfig(
        steps_per_candle=180,
        max_trades_per_candle=5,
    )

    def make_env():
        return TokenTradingEnv(config=config)

    envs = DummyVecEnv([make_env for _ in range(n_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        envs,
        learning_rate=learning_rate,
        buffer_size=200000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(output_path / "tensorboard"),
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=str(output_path / "checkpoints"),
        name_prefix="token_sac",
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

    console.print("[bold green]Starting SAC training...[/bold green]")
    console.print(f"  Action Space: 4D [direction, size, hold, exit]")
    console.print(f"  Observation Space: 55D (51 token features + 4 position state)")
    console.print(f"  Features: EMA-smoothed momentum, non-linear BTC correlation")
    console.print()

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(str(output_path / "final_model"))
    envs.save(str(output_path / "vecnormalize.pkl"))

    console.print(f"\n[bold green]SAC policy saved to {output_dir}[/bold green]")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train token-centric trading models"
    )

    # Model selection
    parser.add_argument("--edge-only", action="store_true", help="Train only edge detector")
    parser.add_argument("--sac-only", action="store_true", help="Train only SAC policy")

    # Edge detector args
    parser.add_argument("--edge-output", type=str, default="./logs/edge_detector_v1")
    parser.add_argument("--n-candles", type=int, default=3000, help="Simulated candles for training")
    parser.add_argument("--epochs", type=int, default=100, help="Edge detector training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--btc-data", type=str, default="./data/btcusdt_1s_30days.parquet")

    # SAC args
    parser.add_argument("--sac-output", type=str, default="./logs/token_sac_v1")
    parser.add_argument("--timesteps", type=int, default=500000, help="SAC training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    console.print("[bold]Token-Centric Model Training[/bold]")
    console.print("=" * 50)
    console.print()

    # Train edge detector
    if not args.sac_only:
        train_edge_detector(
            output_dir=args.edge_output,
            n_candles=args.n_candles,
            epochs=args.epochs,
            batch_size=args.batch_size,
            btc_data_path=args.btc_data,
        )
        console.print()

    # Train SAC
    if not args.edge_only:
        train_sac_policy(
            output_dir=args.sac_output,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
        )

    console.print()
    console.print("[bold green]Training complete![/bold green]")
    console.print()
    console.print("[bold]Next Steps:[/bold]")
    console.print(f"  1. Run paper trading: python src/paper_trade_token.py")
    console.print(f"  2. Monitor with tensorboard: tensorboard --logdir {args.sac_output}/tensorboard")


if __name__ == "__main__":
    main()
