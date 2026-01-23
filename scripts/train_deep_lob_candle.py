#!/usr/bin/env python3
"""
Train DeepLOB model with CANDLE-LEVEL labels.

This fixes the semantic mismatch where the old model predicted 10-second returns
but paper trading settled based on 15-minute candle outcomes.

The new model predicts: "Will this candle close above or below its open?"

Usage:
    python scripts/train_deep_lob_candle.py --data data/btc_1s.parquet --output logs/deep_lob_candle_v1
"""

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from src.training.deep_lob_trainer import (
    DeepLOBTrainer,
    DeepLOBTrainingConfig,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLOB with candle-level labels"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to BTC 1s data (parquet or csv)",
    )
    parser.add_argument(
        "--trades",
        type=str,
        default=None,
        help="Optional path to aggregated trades data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./logs/deep_lob_candle_v1",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--candle-minutes",
        type=int,
        default=15,
        help="Candle duration in minutes",
    )
    parser.add_argument(
        "--legacy-labels",
        action="store_true",
        help="Use legacy 10-second labels instead of candle-level (NOT RECOMMENDED)",
    )

    args = parser.parse_args()

    console.print("[bold blue]DeepLOB Training with Candle-Level Labels[/bold blue]")
    console.print()

    # Load data
    console.print(f"Loading data from {args.data}...")
    data_path = Path(args.data)

    if data_path.suffix == ".parquet":
        btc_data = pd.read_parquet(data_path)
    else:
        btc_data = pd.read_csv(data_path)

    console.print(f"  Loaded {len(btc_data):,} rows")

    # Check required columns
    required_cols = ["timestamp", "price", "volume"]
    missing = [c for c in required_cols if c not in btc_data.columns]
    if missing:
        console.print(f"[red]Missing columns: {missing}[/red]")
        return

    # Add buy_pressure if not present
    if "buy_pressure" not in btc_data.columns:
        console.print("[yellow]buy_pressure column not found, using 0.5 default[/yellow]")
        btc_data["buy_pressure"] = 0.5

    # Load trades data if provided
    trades_data = None
    if args.trades:
        console.print(f"Loading trades from {args.trades}...")
        trades_path = Path(args.trades)
        if trades_path.suffix == ".parquet":
            trades_data = pd.read_parquet(trades_path)
        else:
            trades_data = pd.read_csv(trades_path)
        console.print(f"  Loaded {len(trades_data):,} trades")

    # Configure training
    use_candle_labels = not args.legacy_labels

    if use_candle_labels:
        console.print()
        console.print("[bold green]Using CANDLE-LEVEL labels[/bold green]")
        console.print("  Prediction target: Will candle close above/below open?")
        console.print("  This is the CORRECT semantic for Polymarket 15-min candles")
    else:
        console.print()
        console.print("[bold yellow]WARNING: Using LEGACY short-term labels[/bold yellow]")
        console.print("  Prediction target: Will price move 0.1% in 10 seconds?")
        console.print("  This does NOT match Polymarket candle semantics!")

    config = DeepLOBTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_candle_labels=use_candle_labels,
        # Candle-level labels use tighter threshold since we're predicting 15-min moves
        candle_alpha_threshold=0.0003,  # 0.03%
        # Legacy labels use wider threshold for 10-second moves
        alpha_threshold=0.001,  # 0.1%
        # Other settings
        balanced_sampling=True,
        focal_gamma=3.0,
        patience=30,
    )

    console.print()
    console.print(f"[dim]Configuration:[/dim]")
    console.print(f"  Epochs: {config.epochs}")
    console.print(f"  Batch size: {config.batch_size}")
    console.print(f"  Learning rate: {config.learning_rate}")
    console.print(f"  Use candle labels: {config.use_candle_labels}")
    console.print(f"  Output: {args.output}")
    console.print()

    # Create trainer
    trainer = DeepLOBTrainer(config)

    # Prepare data
    train_loader, val_loader, test_loader, class_weights = trainer.prepare_data(
        btc_data=btc_data,
        trades_data=trades_data,
    )

    # Train
    model, history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        output_dir=args.output,
    )

    # Evaluate
    results = trainer.evaluate_and_report(
        model=model,
        test_loader=test_loader,
        output_dir=args.output,
    )

    console.print()
    console.print(f"[bold green]Training complete![/bold green]")
    console.print(f"  Model saved to: {args.output}")
    console.print(f"  Test F1 (macro): {results['f1_macro']:.3f}")
    console.print(f"  Test accuracy: {results['accuracy']:.3f}")

    # Print recommendations
    console.print()
    console.print("[bold]Next Steps:[/bold]")
    console.print(f"  1. Update paper_trade_dynamic.py to use: --deep-lob {args.output}")
    console.print("  2. Retrain SAC with: python scripts/train_sac_dynamic.py --output logs/sac_dynamic_v6")
    console.print("  3. Run paper trading to validate improvements")


if __name__ == "__main__":
    main()
