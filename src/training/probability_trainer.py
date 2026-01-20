"""
Training pipeline for probabilistic LSTM model.

Trains with proper calibration using:
- Strict time-based splits (no shuffling)
- Brier score as primary metric
- Temperature scaling calibration on held-out set
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.models.probabilistic_lstm import (
    ProbabilisticLSTM,
    ProbabilisticGRU,
    ProbabilisticLSTMConfig,
    create_probability_model,
)
from src.models.calibration import (
    BrierScoreLoss,
    TemperatureScaling,
    CalibrationWrapper,
    compute_calibration_metrics,
    plot_calibration_curve,
)
from src.data.multi_resolution_features import (
    MultiResolutionFeatureBuilder,
    create_training_dataset,
)

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class TrainingConfig:
    """Configuration for probability model training."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-5
    
    # Data splits (strict time-based)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model
    model_type: str = "lstm"  # 'lstm' or 'gru'
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    
    # Calibration
    calibrate_temperature: bool = True
    
    # Saving
    save_every: int = 10


class TimeSeriesDataset(Dataset):
    """Dataset for time series probability prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ProbabilityModelTrainer:
    """
    Trainer for probabilistic LSTM/GRU models.
    
    Key features:
    - Strict time-based train/val/test splits
    - Brier score as primary optimization target
    - Temperature scaling calibration
    - Early stopping on validation Brier score
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(
            "ProbabilityModelTrainer initialized",
            device=self.device,
            config=self.config,
        )
    
    def prepare_data(
        self,
        btc_data: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        eurusd_data: Optional[pd.DataFrame] = None,
        candle_minutes: int = 15,
        sequence_length: int = 180,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders with strict time-based splits.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        console.print("[bold blue]Preparing training data...[/bold blue]")
        
        # Create dataset
        X, y = create_training_dataset(
            btc_data=btc_data,
            dxy_data=dxy_data,
            eurusd_data=eurusd_data,
            candle_minutes=candle_minutes,
            sequence_length=sequence_length,
        )
        
        n_samples = len(X)
        console.print(f"Total samples: {n_samples}")
        console.print(f"Positive ratio: {y.mean():.3f}")
        
        # Strict time-based splits (NO SHUFFLING)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        console.print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Shuffle within train set is OK
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = "./logs/probability_model",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train the probability model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory to save model and logs
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get input dimension from data
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[-1]
        sequence_length = sample_x.shape[1]
        
        # Create model
        model_config = ProbabilisticLSTMConfig(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            output_logits=True,
        )
        
        model = create_probability_model(self.config.model_type, model_config)
        model = model.to(self.device)
        
        # Loss function (Brier score)
        criterion = BrierScoreLoss()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_brier": [],
        }
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        console.print("[bold green]Starting training...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Training", total=self.config.epochs)
            
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_losses = []
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits)
                    loss = criterion(probs, y_batch)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                avg_train_loss = np.mean(train_losses)
                
                # Validation phase
                val_metrics = self._evaluate(model, val_loader, criterion)
                
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_brier"].append(val_metrics["brier"])
                
                # Update scheduler
                scheduler.step(val_metrics["loss"])
                
                # Early stopping check
                if val_metrics["loss"] < best_val_loss - self.config.min_delta:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), output_path / "best_model.pt")
                else:
                    patience_counter += 1
                
                # Update progress
                progress.update(
                    task,
                    advance=1,
                    description=f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                               f"val_brier={val_metrics['brier']:.4f}, "
                               f"val_acc={val_metrics['accuracy']:.3f}"
                )
                
                # Periodic saving
                if (epoch + 1) % self.config.save_every == 0:
                    torch.save(model.state_dict(), output_path / f"model_epoch_{epoch+1}.pt")
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(output_path / "best_model.pt"))
        
        # Save final model
        torch.save(model.state_dict(), output_path / "final_model.pt")
        
        # Save config
        with open(output_path / "config.json", "w") as f:
            json.dump({
                "model_config": model_config.__dict__,
                "training_config": self.config.__dict__,
            }, f, indent=2)
        
        # Save history
        with open(output_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        console.print(f"[bold green]Training complete! Model saved to {output_path}[/bold green]")
        console.print(f"Best validation Brier score: {best_val_loss:.4f}")
        
        return model, history
    
    def _evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        model.eval()
        
        all_probs = []
        all_targets = []
        losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                loss = criterion(probs, y_batch)
                
                losses.append(loss.item())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        
        metrics = compute_calibration_metrics(all_probs, all_targets)
        metrics["loss"] = np.mean(losses)
        
        return metrics
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        output_dir: str = "./logs/probability_model",
    ) -> CalibrationWrapper:
        """
        Apply temperature scaling calibration.
        
        Args:
            model: Trained probability model
            val_loader: Validation data loader for calibration
            output_dir: Directory to save calibrated model
            
        Returns:
            Calibrated model wrapper
        """
        console.print("[bold blue]Calibrating model with temperature scaling...[/bold blue]")
        
        wrapped_model = CalibrationWrapper(model)
        temperature = wrapped_model.calibrate(val_loader, device=self.device)
        
        console.print(f"Fitted temperature: {temperature:.3f}")
        
        # Save calibrated model
        output_path = Path(output_dir)
        torch.save(wrapped_model.state_dict(), output_path / "calibrated_model.pt")
        
        return wrapped_model
    
    def evaluate_and_report(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        output_dir: str = "./logs/probability_model",
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation with calibration analysis.
        
        Args:
            model: Trained (optionally calibrated) model
            test_loader: Test data loader
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        console.print("[bold blue]Evaluating on test set...[/bold blue]")
        
        model.eval()
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch)
                else:
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        
        # Compute metrics
        metrics = compute_calibration_metrics(all_probs, all_targets)
        
        # Print results
        console.print("\n[bold]Test Set Results:[/bold]")
        console.print(f"  Brier Score: {metrics['brier_score']:.4f} (baseline: {metrics['baseline_brier']:.4f})")
        console.print(f"  Brier Skill: {metrics['brier_skill']:.4f}")
        console.print(f"  Accuracy: {metrics['accuracy']:.3f}")
        console.print(f"  ECE: {metrics['ece']:.4f}")
        console.print(f"  Log Loss: {metrics['log_loss']:.4f}")
        
        # Generate calibration plot
        try:
            fig = plot_calibration_curve(all_probs, all_targets, title="Test Set Calibration")
            output_path = Path(output_dir)
            fig.savefig(output_path / "calibration_curve.png", dpi=150)
            console.print(f"Calibration curve saved to {output_path / 'calibration_curve.png'}")
        except ImportError:
            console.print("[yellow]Matplotlib not available, skipping calibration plot[/yellow]")
        
        # Save metrics
        with open(Path(output_dir) / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def train_probability_model(
    btc_data: pd.DataFrame,
    dxy_data: Optional[pd.DataFrame] = None,
    eurusd_data: Optional[pd.DataFrame] = None,
    output_dir: str = "./logs/probability_model",
    epochs: int = 100,
    learning_rate: float = 1e-4,
    model_type: str = "lstm",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to train a probability model.
    
    Args:
        btc_data: BTC price/volume data
        dxy_data: Optional DXY data
        eurusd_data: Optional EUR/USD data
        output_dir: Directory to save model
        epochs: Number of training epochs
        learning_rate: Learning rate
        model_type: 'lstm' or 'gru'
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        model_type=model_type,
    )
    
    trainer = ProbabilityModelTrainer(config)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        btc_data=btc_data,
        dxy_data=dxy_data,
        eurusd_data=eurusd_data,
    )
    
    # Train
    model, history = trainer.train(train_loader, val_loader, output_dir)
    
    # Calibrate
    if config.calibrate_temperature:
        model = trainer.calibrate(model, val_loader, output_dir)
    
    # Evaluate
    metrics = trainer.evaluate_and_report(model, test_loader, output_dir)
    
    return model, metrics
