#!/usr/bin/env python3
"""
Train Probability Predictor.

This script trains a calibrated probability model that predicts P(YES | features).

Key differences from the old training:
1. Target is binary outcome (0 or 1), NOT action labels
2. No position state in features
3. Loss is Brier score for calibration
4. Post-hoc temperature scaling for final calibration
5. Explicit calibration metrics before deployment

Usage:
    python scripts/train_probability_predictor.py --output ./logs/probability_v1

    # With focal loss for class imbalance:
    python scripts/train_probability_predictor.py --output ./logs/probability_v1 --focal-loss
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

from src.data.enhanced_features import EnhancedFeatureBuilder
from src.models.probability_predictor import (
    ProbabilityPredictorConfig,
    ProbabilityPredictorTrainer,
    calibrate_temperature,
    compute_calibration_metrics,
)

logger = structlog.get_logger(__name__)
console = Console()


class ProbabilityTrainingDataBuilder:
    """
    Build training data for probability predictor.

    This differs from the old TrainingDataBuilder:
    1. No position state - only market features
    2. Target is just the outcome (0 or 1)
    3. No synthetic position scenarios
    4. Symmetry augmentation creates balanced YES/NO examples
    """

    def __init__(self, data_dir: str = "./data/historical"):
        self.data_dir = Path(data_dir)
        self.feature_builder = EnhancedFeatureBuilder()

    def load_data(
        self,
        days_back: int = 30,
        btc_interval: str = "1s",
    ) -> Dict[str, pd.DataFrame]:
        """Load previously collected data."""
        data = {}

        btc_path = self.data_dir / f"btc_{btc_interval}_{days_back}d.parquet"
        if btc_path.exists():
            data['btc'] = pd.read_parquet(btc_path)
            console.print(f"[blue]Loaded {len(data['btc'])} BTC records[/blue]")

        candles_path = self.data_dir / f"polymarket_candles_{days_back}d.parquet"
        if candles_path.exists():
            data['candles'] = pd.read_parquet(candles_path)
            console.print(f"[blue]Loaded {len(data['candles'])} candles[/blue]")

        prices_path = self.data_dir / f"polymarket_prices_{days_back}d.parquet"
        if prices_path.exists():
            data['prices'] = pd.read_parquet(prices_path)
            console.print(f"[blue]Loaded {len(data['prices'])} price points[/blue]")

        return data

    def build_training_examples(
        self,
        data: Dict[str, pd.DataFrame],
        samples_per_candle: int = 20,
        augment_symmetry: bool = True,
        min_time_remaining: float = 0.35,  # Match execution filter
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training examples for probability prediction.

        For each sample point in each candle:
        - Features: market state at that time
        - Target: binary outcome (did YES win?)

        Args:
            data: Dictionary with 'prices', 'candles', 'btc' DataFrames
            samples_per_candle: Number of sample points per candle
            augment_symmetry: If True, create flipped examples for balance
            min_time_remaining: Don't sample late in candle (match execution)

        Returns:
            (features, outcomes) arrays
        """
        features_list = []
        outcomes_list = []

        prices_df = data.get('prices', pd.DataFrame())
        candles_df = data.get('candles', pd.DataFrame())
        btc_df = data.get('btc', pd.DataFrame())

        if prices_df.empty or candles_df.empty:
            console.print("[red]No price or candle data available![/red]")
            return np.array([]), np.array([])

        # Group prices by candle
        candle_groups = prices_df.groupby('candle_timestamp')

        console.print(f"Building training examples from {len(candle_groups)} candles...")
        console.print(f"  Samples per candle: {samples_per_candle}")
        console.print(f"  Symmetry augmentation: {augment_symmetry}")
        console.print(f"  Min time remaining: {min_time_remaining:.0%}")

        outcome_counts = {0: 0, 1: 0}

        for candle_ts, candle_prices in candle_groups:
            # Need at least 20 prices for meaningful samples
            if len(candle_prices) < 20:
                continue

            outcome = int(candle_prices['outcome'].iloc[0])
            yes_prices = candle_prices['price'].values
            no_prices = 1.0 - yes_prices

            # Get BTC data for this candle
            candle_start = pd.Timestamp(candle_ts, unit='s', tz='UTC')
            candle_end = candle_start + pd.Timedelta(minutes=15)

            btc_prices = None
            btc_open = None
            if not btc_df.empty and 'timestamp' in btc_df.columns:
                btc_mask = (btc_df['timestamp'] >= candle_start) & (btc_df['timestamp'] < candle_end)
                btc_candle = btc_df.loc[btc_mask]
                if not btc_candle.empty:
                    btc_prices = btc_candle['close'].values
                    btc_open = btc_prices[0]

            # Sample points within candle (respecting min_time_remaining)
            max_idx = len(yes_prices) - 1
            min_sample_idx = 5  # Skip first few for price history
            max_sample_idx = int(max_idx * (1 - min_time_remaining))

            if max_sample_idx <= min_sample_idx:
                continue

            sample_indices = np.linspace(
                min_sample_idx, max_sample_idx, samples_per_candle, dtype=int
            )
            sample_indices = np.unique(sample_indices)

            for idx in sample_indices:
                if idx >= len(yes_prices):
                    continue

                time_remaining = max(0.0, 1.0 - idx / len(yes_prices))

                # Skip if too late (shouldn't happen due to max_sample_idx)
                if time_remaining < min_time_remaining:
                    continue

                # Compute features (NO position state)
                btc_slice = None
                if btc_prices is not None and len(btc_prices) > 0:
                    btc_idx = min(idx + 1, len(btc_prices))
                    btc_slice = btc_prices[:btc_idx]

                features = self.feature_builder.compute_features(
                    yes_prices=yes_prices[:idx+1],
                    no_prices=no_prices[:idx+1],
                    time_remaining=time_remaining,
                    btc_prices=btc_slice,
                    btc_open=btc_open,
                )

                features_list.append(features)
                outcomes_list.append(float(outcome))
                outcome_counts[outcome] += 1

                # Symmetry augmentation: flip YES<->NO
                # This creates a balanced dataset and removes YES/NO bias
                if augment_symmetry:
                    aug_features = self.feature_builder.compute_features(
                        yes_prices=no_prices[:idx+1],  # Swapped
                        no_prices=yes_prices[:idx+1],  # Swapped
                        time_remaining=time_remaining,
                        btc_prices=btc_slice,
                        btc_open=btc_open,
                    )
                    features_list.append(aug_features)
                    outcomes_list.append(float(1 - outcome))  # Flipped outcome
                    outcome_counts[1 - outcome] += 1

        features_arr = np.array(features_list, dtype=np.float32)
        outcomes_arr = np.array(outcomes_list, dtype=np.float32)

        console.print(f"[green]Built {len(features_arr)} training examples[/green]")
        console.print(f"  Outcome distribution: YES={outcome_counts[1]}, NO={outcome_counts[0]}")

        if augment_symmetry:
            console.print(f"  (Should be balanced due to symmetry augmentation)")

        return features_arr, outcomes_arr


def print_calibration_report(metrics: Dict, title: str = "Calibration Report"):
    """Print calibration metrics in a nice table."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Value", width=15)
    table.add_column("Threshold", width=15)
    table.add_column("Status", width=10)

    # Brier score
    brier = metrics["brier_score"]
    brier_ok = brier < 0.24
    table.add_row(
        "Brier Score",
        f"{brier:.4f}",
        "< 0.24",
        "[green]PASS[/]" if brier_ok else "[red]FAIL[/]"
    )

    # ECE
    ece = metrics["ece"]
    ece_ok = ece < 0.05
    table.add_row(
        "ECE",
        f"{ece:.4f}",
        "< 0.05",
        "[green]PASS[/]" if ece_ok else "[red]FAIL[/]"
    )

    # Correlation (diagnostic only)
    corr = metrics["correlation"]
    table.add_row(
        "Correlation",
        f"{corr:.4f}",
        "> 0.10 (diag)",
        f"[yellow]{corr:.4f}[/]"
    )

    # Accuracy
    acc = metrics["accuracy"]
    table.add_row(
        "Accuracy",
        f"{acc:.3f}",
        "N/A",
        ""
    )

    console.print(table)

    # Reliability diagram
    console.print("\n[bold]Reliability Diagram:[/bold]")
    rd = metrics["reliability_diagram"]

    rd_table = Table(show_header=True)
    rd_table.add_column("Bin", width=10)
    rd_table.add_column("Confidence", width=12)
    rd_table.add_column("Accuracy", width=12)
    rd_table.add_column("Count", width=10)
    rd_table.add_column("Gap", width=10)

    for i, (conf, acc, count) in enumerate(zip(
        rd["bin_confidences"],
        rd["bin_accuracies"],
        rd["bin_counts"]
    )):
        if conf is not None and acc is not None:
            gap = abs(acc - conf)
            gap_color = "green" if gap < 0.05 else "yellow" if gap < 0.10 else "red"
            rd_table.add_row(
                f"{i/10:.1f}-{(i+1)/10:.1f}",
                f"{conf:.3f}",
                f"{acc:.3f}",
                str(count),
                f"[{gap_color}]{gap:.3f}[/]"
            )
        else:
            rd_table.add_row(
                f"{i/10:.1f}-{(i+1)/10:.1f}",
                "N/A",
                "N/A",
                str(count),
                ""
            )

    console.print(rd_table)

    # Overall assessment
    all_pass = brier_ok and ece_ok
    console.print()
    if all_pass:
        console.print("[bold green]CALIBRATION GATES: PASSED[/bold green]")
    else:
        console.print("[bold red]CALIBRATION GATES: FAILED[/bold red]")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Train probability predictor")
    parser.add_argument("--output", type=str, default="./logs/probability_v1")
    parser.add_argument("--data-dir", type=str, default="./data/historical")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--samples-per-candle", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss instead of Brier")
    parser.add_argument("--no-augment", action="store_true", help="Disable symmetry augmentation")
    parser.add_argument("--min-time", type=float, default=0.35, help="Min time remaining for samples")

    args = parser.parse_args()

    console.print("[bold blue]Training Probability Predictor[/bold blue]")
    console.print(f"Output: {args.output}")
    console.print(f"Data: {args.data_dir}")

    # Build training data
    builder = ProbabilityTrainingDataBuilder(args.data_dir)
    data = builder.load_data(days_back=args.days)

    if not data:
        console.print("[red]No data found! Run data collection first.[/red]")
        return

    features, outcomes = builder.build_training_examples(
        data,
        samples_per_candle=args.samples_per_candle,
        augment_symmetry=not args.no_augment,
        min_time_remaining=args.min_time,
    )

    if len(features) == 0:
        console.print("[red]No training examples generated![/red]")
        return

    # Split data
    train_features, temp_features, train_outcomes, temp_outcomes = train_test_split(
        features, outcomes, test_size=0.3, random_state=42
    )
    val_features, test_features, val_outcomes, test_outcomes = train_test_split(
        temp_features, temp_outcomes, test_size=0.5, random_state=42
    )

    console.print(f"\n[bold]Data Split:[/bold]")
    console.print(f"  Train: {len(train_features)}")
    console.print(f"  Val: {len(val_features)}")
    console.print(f"  Test: {len(test_features)}")

    # Train model
    config = ProbabilityPredictorConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = ProbabilityPredictorTrainer(config)

    console.print("\n[bold]Starting training...[/bold]")
    model, history = trainer.train(
        train_features=train_features,
        train_outcomes=train_outcomes,
        val_features=val_features,
        val_outcomes=val_outcomes,
        output_dir=args.output,
        use_focal_loss=args.focal_loss,
    )

    # Calibrate temperature
    console.print("\n[bold]Calibrating temperature...[/bold]")
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    optimal_temp = calibrate_temperature(model, val_features, val_outcomes, device)
    model.set_temperature(optimal_temp)

    # Update saved config with calibrated temperature
    config_path = Path(args.output) / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config_dict["temperature"] = optimal_temp
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Re-save model with calibrated temperature
    __import__('torch').save(model.state_dict(), Path(args.output) / "final_model.pt")

    # Evaluate on test set
    console.print("\n[bold]Evaluating on test set...[/bold]")
    model.eval()
    with __import__('torch').no_grad():
        test_features_t = __import__('torch').FloatTensor(test_features).to(device)
        test_preds = model(test_features_t).squeeze().cpu().numpy()

    metrics = compute_calibration_metrics(test_preds, test_outcomes)

    # Print report
    gates_passed = print_calibration_report(metrics, "Test Set Calibration")

    # Save metrics
    metrics_path = Path(args.output) / "calibration_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "brier_score": float(metrics["brier_score"]),
            "ece": float(metrics["ece"]),
            "correlation": float(metrics["correlation"]),
            "accuracy": float(metrics["accuracy"]),
            "temperature": optimal_temp,
            "gates_passed": bool(gates_passed),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    console.print(f"\n[green]Model saved to: {args.output}[/green]")
    console.print(f"[green]Calibration metrics saved to: {metrics_path}[/green]")

    # Final recommendation
    console.print("\n" + "=" * 60)
    if gates_passed:
        console.print("[bold green]MODEL READY FOR PAPER TRADING[/bold green]")
        console.print(f"\nRun with:")
        console.print(f"  python -m src.paper_trade_safeguarded --model {args.output} --model-type probability")
    else:
        console.print("[bold red]MODEL NOT READY - CALIBRATION FAILED[/bold red]")
        console.print("\nConsider:")
        console.print("  - More training data")
        console.print("  - Different learning rate")
        console.print("  - Feature engineering")


if __name__ == "__main__":
    main()
