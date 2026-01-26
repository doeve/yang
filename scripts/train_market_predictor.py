#!/usr/bin/env python3
"""
Train Unified Market Predictor Model.

This replaces the separate EdgeDetector + SAC architecture with a single
end-to-end model trained on REAL historical data with optimal action labels.

Key improvements over previous approach:
1. Trains on REAL data, not simulated 50/50 balanced data
2. Predicts OPTIMAL ACTIONS (with expected return), not just P(YES)
3. Single unified model instead of two separate models
4. No hardcoded thresholds - model learns optimal values
5. Time-aware: naturally learns when to avoid late entries
6. Trend-aware: learns from actual BTC/token price correlations

Usage:
    # Collect data and train
    python scripts/train_market_predictor.py --collect-data --days 30

    # Train only (assumes data already collected)
    python scripts/train_market_predictor.py --train-only

    # Full pipeline with custom settings
    python scripts/train_market_predictor.py --days 30 --epochs 150 --batch-size 256
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.model_selection import train_test_split

console = Console()


async def collect_historical_data(
    output_dir: str = "./data/historical",
    days_back: int = 30,
    btc_interval: str = "1m",  # Use 1m for faster collection, still good resolution
):
    """Collect historical data from Binance and Polymarket."""
    from src.data.historical_data_collector import HistoricalDataCollector

    console.print("[bold blue]Step 1: Collecting Historical Data[/bold blue]")
    console.print(f"  Days: {days_back}")
    console.print(f"  BTC interval: {btc_interval}")
    console.print(f"  Output: {output_dir}")
    console.print()

    collector = HistoricalDataCollector(output_dir=output_dir)
    data = await collector.collect_all_data(
        days_back=days_back,
        btc_interval=btc_interval,
    )

    return data


def build_training_data(
    data_dir: str = "./data/historical",
    days_back: int = 30,
    btc_interval: str = "1m",
    samples_per_candle: int = 15,
):
    """Build training examples from collected data."""
    from src.data.historical_data_collector import TrainingDataBuilder

    console.print("[bold blue]Step 2: Building Training Data[/bold blue]")
    console.print(f"  Data dir: {data_dir}")
    console.print(f"  Samples per candle: {samples_per_candle}")
    console.print()

    builder = TrainingDataBuilder(data_dir=data_dir)
    data = builder.load_data(days_back=days_back, btc_interval=btc_interval)

    if not data:
        console.print("[red]No data found! Run with --collect-data first.[/red]")
        return None, None, None, None

    features, position_states, actions, returns = builder.build_training_examples(
        data=data,
        samples_per_candle=samples_per_candle,
    )

    if len(features) == 0:
        console.print("[red]No training examples generated![/red]")
        return None, None, None, None

    console.print(f"[green]Built {len(features)} training examples[/green]")
    console.print(f"  Feature dim: {features.shape[1]}")
    console.print(f"  Position state dim: {position_states.shape[1]}")

    # Analyze action distribution
    from src.models.market_predictor import Action
    action_counts = np.bincount(actions, minlength=Action.num_actions())
    console.print("\n[bold]Action Distribution:[/bold]")
    for i, name in enumerate(Action.names()):
        pct = action_counts[i] / len(actions) * 100
        console.print(f"  {name}: {action_counts[i]} ({pct:.1f}%)")

    # Analyze return distribution
    console.print(f"\n[bold]Return Statistics:[/bold]")
    console.print(f"  Mean: {returns.mean():.4f}")
    console.print(f"  Std: {returns.std():.4f}")
    console.print(f"  Min: {returns.min():.4f}")
    console.print(f"  Max: {returns.max():.4f}")

    return features, position_states, actions, returns


def train_model(
    features: np.ndarray,
    position_states: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    output_dir: str = "./logs/market_predictor_v1",
    epochs: int = 100,
    batch_size: int = 128,
    patience: int = 15,
    val_split: float = 0.15,
    test_split: float = 0.10,
):
    """Train the unified market predictor model."""
    from src.models.market_predictor import (
        MarketPredictorConfig,
        MarketPredictorTrainer,
        Action,
    )

    console.print("[bold blue]Step 3: Training Model[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Patience: {patience}")
    console.print()

    # Split data: train/val/test
    # First split: separate test set
    (
        features_trainval, features_test,
        pos_trainval, pos_test,
        actions_trainval, actions_test,
        returns_trainval, returns_test,
    ) = train_test_split(
        features, position_states, actions, returns,
        test_size=test_split,
        random_state=42,
        stratify=actions,  # Stratify by action to maintain distribution
    )

    # Second split: separate validation from training
    val_ratio = val_split / (1 - test_split)  # Adjust ratio after test split
    (
        features_train, features_val,
        pos_train, pos_val,
        actions_train, actions_val,
        returns_train, returns_val,
    ) = train_test_split(
        features_trainval, pos_trainval, actions_trainval, returns_trainval,
        test_size=val_ratio,
        random_state=42,
        stratify=actions_trainval,
    )

    console.print(f"[green]Data split:[/green]")
    console.print(f"  Train: {len(features_train)}")
    console.print(f"  Val: {len(features_val)}")
    console.print(f"  Test: {len(features_test)}")

    # Create config with correct dimensions
    config = MarketPredictorConfig(
        base_feature_dim=features.shape[1],
        position_state_dim=position_states.shape[1],
        hidden_dims=(256, 128, 64),
        attention_heads=4,
        dropout=0.25,
        learning_rate=3e-4,
        weight_decay=1e-5,
    )

    # Train
    trainer = MarketPredictorTrainer(config)
    model, history = trainer.train(
        train_features=features_train,
        train_position_states=pos_train,
        train_actions=actions_train,
        train_returns=returns_train,
        val_features=features_val,
        val_position_states=pos_val,
        val_actions=actions_val,
        val_returns=returns_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        output_dir=output_dir,
    )

    # Evaluate on test set
    console.print("\n[bold]Test Set Evaluation:[/bold]")

    import torch
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        test_features_t = torch.FloatTensor(features_test).to(device)
        test_pos_t = torch.FloatTensor(pos_test).to(device)
        test_actions_t = torch.LongTensor(actions_test).to(device)
        test_returns_t = torch.FloatTensor(returns_test).to(device)

        x = torch.cat([test_features_t, test_pos_t], dim=-1)
        outputs = model(x)

        # Action accuracy
        pred_actions = outputs['action_logits'].argmax(dim=-1)
        accuracy = (pred_actions == test_actions_t).float().mean().item()

        # Return MSE
        pred_returns = outputs['expected_return'].squeeze(-1)
        return_mse = ((pred_returns - test_returns_t) ** 2).mean().item()

        # Per-action accuracy
        console.print(f"  Overall Accuracy: {accuracy:.3f}")
        console.print(f"  Return MSE: {return_mse:.4f}")

        console.print("\n  Per-Action Accuracy:")
        for action_id, action_name in enumerate(Action.names()):
            mask = test_actions_t == action_id
            if mask.sum() > 0:
                action_acc = (pred_actions[mask] == action_id).float().mean().item()
                console.print(f"    {action_name}: {action_acc:.3f} ({mask.sum().item()} samples)")

    # Save test results
    test_results = {
        "accuracy": accuracy,
        "return_mse": return_mse,
        "test_size": len(features_test),
        "train_size": len(features_train),
        "val_size": len(features_val),
    }

    output_path = Path(output_dir)
    with open(output_path / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    console.print(f"\n[bold green]Model saved to {output_dir}[/bold green]")

    return model, history, test_results


async def main():
    parser = argparse.ArgumentParser(description="Train unified market predictor")

    # Data collection
    parser.add_argument("--collect-data", action="store_true", help="Collect historical data first")
    parser.add_argument("--train-only", action="store_true", help="Skip data collection, train only")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--btc-interval", type=str, default="1m", help="BTC data interval")
    parser.add_argument("--data-dir", type=str, default="./data/historical")

    # Training
    parser.add_argument("--output", type=str, default="./logs/market_predictor_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--samples-per-candle", type=int, default=15)

    args = parser.parse_args()

    console.print("[bold]Unified Market Predictor Training[/bold]")
    console.print("=" * 50)
    console.print()

    # Step 1: Data collection (optional)
    if args.collect_data and not args.train_only:
        await collect_historical_data(
            output_dir=args.data_dir,
            days_back=args.days,
            btc_interval=args.btc_interval,
        )
        console.print()

    # Step 2: Build training data
    features, position_states, actions, returns = build_training_data(
        data_dir=args.data_dir,
        days_back=args.days,
        btc_interval=args.btc_interval,
        samples_per_candle=args.samples_per_candle,
    )

    if features is None:
        console.print("[red]Training aborted: No data available[/red]")
        return

    console.print()

    # Step 3: Train model
    model, history, results = train_model(
        features=features,
        position_states=position_states,
        actions=actions,
        returns=returns,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    console.print()
    console.print("[bold green]Training complete![/bold green]")
    console.print()
    console.print("[bold]Next Steps:[/bold]")
    console.print(f"  1. Run paper trading: python src/paper_trade_unified.py --model {args.output}")
    console.print(f"  2. Analyze results in {args.output}/")


if __name__ == "__main__":
    asyncio.run(main())
