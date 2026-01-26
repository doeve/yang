#!/usr/bin/env python3
"""
Train Attention-Based Market Predictor with KAMA Spectrum.

This script trains the time-aware attention model that dynamically weights
KAMA features based on market regime and time remaining.

Key features:
- KAMA Spectrum: 12 adaptive moving averages at different periods
- Time-Aware Context: time_remaining, distance_to_strike, volatility
- Feature Attention: Model learns which KAMA to trust based on context
- Attention Logging: Tracks which KAMAs are attended to by time phase

Usage:
    # Train attention model
    python scripts/train_attention_market_predictor.py --epochs 100

    # With data collection
    python scripts/train_attention_market_predictor.py --collect-data --days 30

    # Custom output
    python scripts/train_attention_market_predictor.py --output ./logs/attention_v1
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

console = Console()


async def collect_historical_data(
    output_dir: str = "./data/historical",
    days_back: int = 30,
    btc_interval: str = "1m",
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


def build_attention_training_data(
    data_dir: str = "./data/historical",
    days_back: int = 30,
    btc_interval: str = "1m",
    samples_per_candle: int = 15,
):
    """Build training examples with KAMA spectrum features."""
    from src.data.historical_data_collector import TrainingDataBuilder

    console.print("[bold blue]Step 2: Building KAMA Training Data[/bold blue]")
    console.print(f"  Data dir: {data_dir}")
    console.print(f"  Samples per candle: {samples_per_candle}")
    console.print()

    builder = TrainingDataBuilder(data_dir=data_dir)
    data = builder.load_data(days_back=days_back, btc_interval=btc_interval)

    if not data:
        console.print("[red]No data found! Run with --collect-data first.[/red]")
        return None

    # Build attention training examples
    (
        kama_spectrum, context_features, base_features,
        position_states, actions, returns
    ) = builder.build_attention_training_examples(
        data=data,
        samples_per_candle=samples_per_candle,
    )

    if len(kama_spectrum) == 0:
        console.print("[red]No training examples generated![/red]")
        return None

    console.print(f"[green]Built {len(kama_spectrum)} training examples[/green]")
    console.print(f"  KAMA spectrum dim: {kama_spectrum.shape[1]}")
    console.print(f"  Context dim: {context_features.shape[1]}")
    console.print(f"  Base features dim: {base_features.shape[1]}")
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

    return {
        'kama_spectrum': kama_spectrum,
        'context_features': context_features,
        'base_features': base_features,
        'position_states': position_states,
        'actions': actions,
        'returns': returns,
    }


def train_attention_model(
    data: dict,
    output_dir: str = "./logs/attention_market_predictor_v1",
    epochs: int = 100,
    batch_size: int = 128,
    patience: int = 15,
    val_split: float = 0.15,
    test_split: float = 0.10,
    attention_log_interval: int = 10,
):
    """Train the attention-based market predictor."""
    from src.models.attention_market_predictor import (
        AttentionMarketPredictorConfig,
        AttentionMarketPredictorTrainer,
        AttentionMarketPredictorModel,
    )
    from src.models.market_predictor import Action

    console.print("[bold blue]Step 3: Training Attention Model[/bold blue]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Patience: {patience}")
    console.print(f"  Attention log interval: {attention_log_interval} epochs")
    console.print()

    # Extract data
    kama_spectrum = data['kama_spectrum']
    context_features = data['context_features']
    base_features = data['base_features']
    position_states = data['position_states']
    actions = data['actions']
    returns = data['returns']

    # Split data: train/val/test
    # First split: separate test set
    (
        kama_trainval, kama_test,
        context_trainval, context_test,
        base_trainval, base_test,
        pos_trainval, pos_test,
        actions_trainval, actions_test,
        returns_trainval, returns_test,
    ) = train_test_split(
        kama_spectrum, context_features, base_features,
        position_states, actions, returns,
        test_size=test_split,
        random_state=42,
        stratify=actions,
    )

    # Second split: separate validation from training
    val_ratio = val_split / (1 - test_split)
    (
        kama_train, kama_val,
        context_train, context_val,
        base_train, base_val,
        pos_train, pos_val,
        actions_train, actions_val,
        returns_train, returns_val,
    ) = train_test_split(
        kama_trainval, context_trainval, base_trainval,
        pos_trainval, actions_trainval, returns_trainval,
        test_size=val_ratio,
        random_state=42,
        stratify=actions_trainval,
    )

    console.print(f"[green]Data split:[/green]")
    console.print(f"  Train: {len(kama_train)}")
    console.print(f"  Val: {len(kama_val)}")
    console.print(f"  Test: {len(kama_test)}")

    # Create config
    num_kamas = kama_spectrum.shape[1] // 2  # deviation + slope per KAMA
    config = AttentionMarketPredictorConfig(
        num_kamas=num_kamas,
        kama_features_per_period=2,
        context_dim=context_features.shape[1],
        base_feature_dim=base_features.shape[1],
        position_state_dim=position_states.shape[1],
        hidden_dims=(256, 128, 64),
        attention_hidden_dim=64,
        dropout=0.25,
        learning_rate=3e-4,
        weight_decay=1e-5,
    )

    console.print(f"\n[bold]Model Configuration:[/bold]")
    console.print(f"  Num KAMAs: {config.num_kamas}")
    console.print(f"  KAMA spectrum dim: {config.kama_spectrum_dim}")
    console.print(f"  Context dim: {config.context_dim}")
    console.print(f"  Base feature dim: {config.base_feature_dim}")
    console.print(f"  Hidden dims: {config.hidden_dims}")

    # Train
    trainer = AttentionMarketPredictorTrainer(config)
    model, history = trainer.train(
        train_kama=kama_train,
        train_context=context_train,
        train_base=base_train,
        train_position_states=pos_train,
        train_actions=actions_train,
        train_returns=returns_train,
        val_kama=kama_val,
        val_context=context_val,
        val_base=base_val,
        val_position_states=pos_val,
        val_actions=actions_val,
        val_returns=returns_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        output_dir=output_dir,
        attention_log_interval=attention_log_interval,
    )

    # Evaluate on test set
    console.print("\n[bold]Test Set Evaluation:[/bold]")

    import torch
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        kama_test_t = torch.FloatTensor(kama_test).to(device)
        context_test_t = torch.FloatTensor(context_test).to(device)
        base_test_t = torch.FloatTensor(base_test).to(device)
        pos_test_t = torch.FloatTensor(pos_test).to(device)
        actions_test_t = torch.LongTensor(actions_test).to(device)
        returns_test_t = torch.FloatTensor(returns_test).to(device)

        outputs = model(kama_test_t, context_test_t, base_test_t, pos_test_t)

        # Apply action mask for evaluation
        has_position = pos_test_t[:, 0] > 0.5
        masked_logits = model._apply_action_mask(outputs['action_logits'], has_position)
        
        # Action accuracy
        pred_actions = masked_logits.argmax(dim=-1)
        accuracy = (pred_actions == actions_test_t).float().mean().item()

        # Return MSE
        pred_returns = outputs['expected_return'].squeeze(-1)
        return_mse = ((pred_returns - returns_test_t) ** 2).mean().item()

        # Attention weight analysis
        attn_weights = outputs['attention_weights'].cpu().numpy()
        time_remaining = context_test[:, 0]  # First context feature is time_remaining_pct

        console.print(f"  Overall Accuracy: {accuracy:.3f}")
        console.print(f"  Return MSE: {return_mse:.4f}")

        console.print("\n  Per-Action Accuracy:")
        for action_id, action_name in enumerate(Action.names()):
            mask = actions_test_t == action_id
            if mask.sum() > 0:
                action_acc = (pred_actions[mask] == action_id).float().mean().item()
                console.print(f"    {action_name}: {action_acc:.3f} ({mask.sum().item()} samples)")

        # Final attention analysis
        console.print("\n[bold]Final Attention Pattern Analysis:[/bold]")
        _print_attention_analysis(attn_weights, time_remaining)

    # Save test results
    test_results = {
        "accuracy": accuracy,
        "return_mse": return_mse,
        "test_size": len(kama_test),
        "train_size": len(kama_train),
        "val_size": len(kama_val),
    }

    output_path = Path(output_dir)
    with open(output_path / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    console.print(f"\n[bold green]Model saved to {output_dir}[/bold green]")

    return model, history, test_results


def _print_attention_analysis(attn_weights: np.ndarray, time_remaining: np.ndarray):
    """Print attention weight analysis by time phase."""
    kama_periods = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80]
    
    phases = [
        ('EARLY (>60%)', 0.6, 1.0),
        ('MID (30-60%)', 0.3, 0.6),
        ('LATE (<30%)', 0.0, 0.3),
    ]

    for phase_name, low, high in phases:
        mask = (time_remaining >= low) & (time_remaining < high)
        if mask.sum() == 0:
            continue

        phase_weights = attn_weights[mask].mean(axis=0)

        table = Table(title=f"Time Phase: {phase_name}")
        table.add_column("KAMA Period", style="cyan")
        table.add_column("Weight", style="green")
        table.add_column("Bar", style="yellow")

        for i, period in enumerate(kama_periods[:len(phase_weights)]):
            weight = phase_weights[i]
            bar = "█" * int(weight * 30)
            table.add_row(f"KAMA_{period}", f"{weight:.3f}", bar)

        console.print(table)
        console.print()


async def main():
    parser = argparse.ArgumentParser(description="Train attention-based market predictor")

    # Data collection
    parser.add_argument("--collect-data", action="store_true", help="Collect historical data first")
    parser.add_argument("--train-only", action="store_true", help="Skip data collection, train only")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--btc-interval", type=str, default="1m", help="BTC data interval")
    parser.add_argument("--data-dir", type=str, default="./data/historical")

    # Training
    parser.add_argument("--output", type=str, default="./logs/attention_market_predictor_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--samples-per-candle", type=int, default=15)
    parser.add_argument("--attention-log-interval", type=int, default=10,
                        help="Log attention weights every N epochs")

    args = parser.parse_args()

    console.print("[bold]Attention Market Predictor Training[/bold]")
    console.print("[blue]Time-Aware KAMA Spectrum with Feature Attention[/blue]")
    console.print("=" * 60)
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
    data = build_attention_training_data(
        data_dir=args.data_dir,
        days_back=args.days,
        btc_interval=args.btc_interval,
        samples_per_candle=args.samples_per_candle,
    )

    if data is None:
        console.print("[red]Training aborted: No data available[/red]")
        return

    console.print()

    # Step 3: Train model
    model, history, results = train_attention_model(
        data=data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        attention_log_interval=args.attention_log_interval,
    )

    console.print()
    console.print("[bold green]Training complete![/bold green]")
    console.print()
    console.print("[bold]Next Steps:[/bold]")
    console.print(f"  1. Review attention weights in {args.output}/attention_weights_*.csv")
    console.print(f"  2. Run paper trading: python src/paper_trade_unified.py --model {args.output} --attention")
    console.print(f"  3. Analyze results in {args.output}/")


if __name__ == "__main__":
    asyncio.run(main())
