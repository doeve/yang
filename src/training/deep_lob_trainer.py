"""
Training pipeline for DeepLOB 3-class prediction model.

Uses:
- Focal loss for class imbalance
- F1-score as primary evaluation metric
- Class-weighted training
- Rolling window normalization
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.data.deep_lob_features import (
    DeepLOBFeatureBuilder,
    DeepLOBConfig as FeatureConfig,
    create_training_dataset_v2,
)
from src.models.deep_lob_model import (
    DeepLOBModel,
    DeepLOBConfig,
    FocalLoss,
    compute_class_weights,
)

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class DeepLOBTrainingConfig:
    """Training configuration for DeepLOB model."""
    
    # Training
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    
    # Early stopping
    patience: int = 30
    min_delta: float = 1e-4
    
    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    sequence_length: int = 120
    balanced_sampling: bool = True  # Use balanced batch sampler
    
    # Model
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.4
    
    # Loss
    focal_gamma: float = 3.0  # Higher gamma for harder focus on difficult samples
    use_class_weights: bool = True
    
    # Labeling
    prediction_horizon: int = 10
    alpha_threshold: float = 0.001  # 0.1% threshold
    
    # Saving
    save_every: int = 20


class DeepLOBDataset(Dataset):
    """Dataset for DeepLOB 3-class prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class BalancedBatchSampler:
    """Sampler that ensures balanced classes in each batch."""
    
    def __init__(self, y: np.ndarray, batch_size: int, num_classes: int = 3):
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Get indices for each class
        self.class_indices = {}
        for c in range(num_classes):
            self.class_indices[c] = np.where(y == c)[0].tolist()
        
        # Calculate batches per epoch
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = max(1, (min_class_size * num_classes) // batch_size)
    
    def __iter__(self):
        # Shuffle class indices
        shuffled_indices = {}
        for c in range(self.num_classes):
            shuffled_indices[c] = np.random.permutation(self.class_indices[c]).tolist()
        
        # Generate balanced batches
        samples_per_class = self.batch_size // self.num_classes
        
        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                if len(shuffled_indices[c]) < samples_per_class:
                    # Reshuffle if exhausted
                    shuffled_indices[c] = np.random.permutation(self.class_indices[c]).tolist()
                
                batch.extend(shuffled_indices[c][:samples_per_class])
                shuffled_indices[c] = shuffled_indices[c][samples_per_class:]
            
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.num_batches


class DeepLOBTrainer:
    """Trainer for DeepLOB 3-class model."""
    
    def __init__(
        self,
        config: Optional[DeepLOBTrainingConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DeepLOBTrainingConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(
            "DeepLOBTrainer initialized",
            device=self.device,
        )
    
    def prepare_data(
        self,
        btc_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
        """
        Prepare data loaders with strict time-based splits.
        
        Returns:
            (train_loader, val_loader, test_loader, class_weights)
        """
        console.print("[bold blue]Preparing DeepLOB training data...[/bold blue]")
        
        # Create dataset with 3-class labels
        X, y = create_training_dataset_v2(
            btc_data=btc_data,
            trades_data=trades_data,
            candle_minutes=15,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            alpha_threshold=self.config.alpha_threshold,
        )
        
        n = len(X)
        console.print(f"Total samples: {n}")
        
        # Class distribution
        for c in range(3):
            pct = (y == c).sum() / len(y) * 100
            label = ["Down", "Hold", "Up"][c]
            console.print(f"  {label}: {(y == c).sum()} ({pct:.1f}%)")
        
        # Time-based splits
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Normalize features (sliding window z-score on train, apply to all)
        console.print("[dim]Normalizing features...[/dim]")
        n_train, seq_len, n_features = X_train.shape
        
        X_train_flat = X_train.reshape(-1, n_features)
        self.feature_mean = np.mean(X_train_flat, axis=0)
        self.feature_std = np.std(X_train_flat, axis=0) + 1e-8
        
        X_train = ((X_train.reshape(-1, n_features) - self.feature_mean) / self.feature_std).reshape(n_train, seq_len, n_features)
        X_val = ((X_val.reshape(-1, n_features) - self.feature_mean) / self.feature_std).reshape(len(X_val), seq_len, n_features)
        X_test = ((X_test.reshape(-1, n_features) - self.feature_mean) / self.feature_std).reshape(len(X_test), seq_len, n_features)
        
        # Clip extreme values
        X_train = np.clip(X_train, -5, 5)
        X_val = np.clip(X_val, -5, 5)
        X_test = np.clip(X_test, -5, 5)
        
        console.print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Compute class weights from training data
        class_weights = compute_class_weights(y_train, num_classes=3)
        
        # Create data loaders
        if getattr(self.config, 'balanced_sampling', True):
            # Use balanced batch sampler
            balanced_sampler = BalancedBatchSampler(
                y_train, 
                self.config.batch_size, 
                num_classes=3
            )
            train_loader = DataLoader(
                DeepLOBDataset(X_train, y_train),
                batch_sampler=balanced_sampler,
            )
            console.print("[dim]Using balanced batch sampling[/dim]")
        else:
            train_loader = DataLoader(
                DeepLOBDataset(X_train, y_train),
                batch_size=self.config.batch_size,
                shuffle=True,
            )
        
        val_loader = DataLoader(
            DeepLOBDataset(X_val, y_val),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            DeepLOBDataset(X_test, y_test),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        return train_loader, val_loader, test_loader, class_weights

    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        output_dir: str = "./logs/deep_lob_model",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train the DeepLOB model.
        
        Returns:
            (trained_model, training_history)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get input dimension
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[-1]
        seq_len = sample_x.shape[1]
        
        # Create model
        model_config = DeepLOBConfig(
            input_dim=input_dim,
            sequence_length=seq_len,
            lstm_hidden=self.config.lstm_hidden,
            lstm_layers=self.config.lstm_layers,
            dropout=self.config.dropout,
            num_classes=3,
        )
        
        model = DeepLOBModel(model_config).to(self.device)
        
        # Loss function (focal loss with class weights)
        if self.config.use_class_weights:
            criterion = FocalLoss(
                alpha=class_weights.to(self.device),
                gamma=self.config.focal_gamma,
            )
        else:
            criterion = FocalLoss(gamma=self.config.focal_gamma)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
        }
        
        best_val_f1 = 0
        patience_counter = 0
        
        console.print("[bold green]Starting DeepLOB training...[/bold green]")
        
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
                    loss = criterion(logits, y_batch)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                scheduler.step()
                avg_train_loss = np.mean(train_losses)
                
                # Validation phase
                val_metrics = self._evaluate(model, val_loader, criterion)
                
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_f1"].append(val_metrics["f1_macro"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                
                # Check for improvement (use F1 score, not loss)
                if val_metrics["f1_macro"] > best_val_f1 + self.config.min_delta:
                    best_val_f1 = val_metrics["f1_macro"]
                    patience_counter = 0
                    torch.save(model.state_dict(), output_path / "best_model.pt")
                else:
                    patience_counter += 1
                
                # Update progress
                progress.update(
                    task,
                    advance=1,
                    description=f"Epoch {epoch+1}: loss={avg_train_loss:.4f}, "
                               f"val_f1={val_metrics['f1_macro']:.3f}, "
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
        model.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))
        
        # Save final model and config
        torch.save(model.state_dict(), output_path / "final_model.pt")
        
        with open(output_path / "config.json", "w") as f:
            json.dump({
                "model_config": model_config.__dict__,
                "training_config": self.config.__dict__,
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
            }, f, indent=2, default=str)
        
        with open(output_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        console.print(f"[bold green]Training complete! Best F1: {best_val_f1:.3f}[/bold green]")
        
        return model, history
    
    def _evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate model with F1 score and accuracy."""
        model.eval()
        
        all_preds = []
        all_targets = []
        losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                preds = torch.argmax(logits, dim=-1)
                
                losses.append(loss.item())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Compute metrics
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        accuracy = (all_preds == all_targets).mean()
        
        return {
            "loss": np.mean(losses),
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
        }
    
    def evaluate_and_report(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        output_dir: str = "./logs/deep_lob_model",
    ) -> Dict[str, Any]:
        """Generate detailed evaluation report on test set."""
        console.print("[bold blue]Evaluating on test set...[/bold blue]")
        
        model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        # Metrics
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        accuracy = (all_preds == all_targets).mean()
        
        # Per-class metrics
        class_names = ["Down", "Hold", "Up"]
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        console.print("\n[bold]Test Set Results:[/bold]")
        console.print(f"  F1 Macro: {f1_macro:.3f}")
        console.print(f"  F1 Weighted: {f1_weighted:.3f}")
        console.print(f"  Accuracy: {accuracy:.3f}")
        console.print("\n[bold]Per-Class Performance:[/bold]")
        for name in class_names:
            console.print(f"  {name}: F1={report[name]['f1-score']:.3f}, "
                         f"Precision={report[name]['precision']:.3f}, "
                         f"Recall={report[name]['recall']:.3f}")
        
        console.print("\n[bold]Confusion Matrix:[/bold]")
        console.print(f"  {'':8} Down  Hold   Up")
        for i, name in enumerate(class_names):
            console.print(f"  {name:8} {cm[i, 0]:4}  {cm[i, 1]:4}  {cm[i, 2]:4}")
        
        # Save results
        results = {
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        
        with open(Path(output_dir) / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results


def train_deep_lob_model(
    btc_data: pd.DataFrame,
    trades_data: Optional[pd.DataFrame] = None,
    output_dir: str = "./logs/deep_lob_model",
    epochs: int = 200,
    alpha_threshold: float = 0.001,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to train DeepLOB model.
    
    Args:
        btc_data: BTC data with timestamp, price, volume, buy_pressure
        trades_data: Optional trades data with is_buyer_maker
        output_dir: Output directory
        epochs: Training epochs
        alpha_threshold: Threshold for Up/Down classification
        
    Returns:
        (trained_model, test_results)
    """
    config = DeepLOBTrainingConfig(
        epochs=epochs,
        alpha_threshold=alpha_threshold,
    )
    
    trainer = DeepLOBTrainer(config)
    
    # Prepare data
    train_loader, val_loader, test_loader, class_weights = trainer.prepare_data(
        btc_data=btc_data,
        trades_data=trades_data,
    )
    
    # Train
    model, history = trainer.train(
        train_loader, val_loader, class_weights, output_dir
    )
    
    # Evaluate
    results = trainer.evaluate_and_report(model, test_loader, output_dir)
    
    return model, results


def train_deep_lob_balanced(
    balanced_data: pd.DataFrame,
    output_dir: str = "./logs/deep_lob_balanced",
    epochs: int = 150,
    alpha_threshold: float = 0.0001,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train DeepLOB model on pre-balanced data.
    
    Args:
        balanced_data: DataFrame with interval_id, label, and price data
        output_dir: Output directory
        epochs: Training epochs
        alpha_threshold: Threshold (used for feature builder)
        
    Returns:
        (trained_model, test_results)
    """
    from src.data.deep_lob_features import DeepLOBFeatureBuilder
    
    console.print("[bold blue]Preparing balanced training data...[/bold blue]")
    
    # Group by interval_id
    interval_ids = balanced_data["interval_id"].unique()
    console.print(f"  Total intervals: {len(interval_ids)}")
    
    # Feature builder
    feature_builder = DeepLOBFeatureBuilder()
    
    # Build features and labels for each interval
    all_features = []
    all_labels = []
    
    for interval_id in interval_ids:
        interval_df = balanced_data[balanced_data["interval_id"] == interval_id]
        
        if len(interval_df) < 50:
            continue
        
        # Get label
        label = interval_df["label"].iloc[0]
        
        # Build features
        features = feature_builder.precompute_all_features(interval_df)
        
        # Take last 120 samples as sequence
        if len(features) >= 120:
            seq = features[-120:]
            all_features.append(seq)
            all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    console.print(f"  Built {len(X)} sequences")
    console.print(f"  Class distribution: Down={sum(y==0)}, Hold={sum(y==1)}, Up={sum(y==2)}")
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split (80/10/10)
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Normalize
    n_train, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    feature_mean = np.mean(X_train_flat, axis=0)
    feature_std = np.std(X_train_flat, axis=0) + 1e-8
    
    X_train = ((X_train.reshape(-1, n_features) - feature_mean) / feature_std).reshape(n_train, seq_len, n_features)
    X_val = ((X_val.reshape(-1, n_features) - feature_mean) / feature_std).reshape(len(X_val), seq_len, n_features)
    X_test = ((X_test.reshape(-1, n_features) - feature_mean) / feature_std).reshape(len(X_test), seq_len, n_features)
    
    X_train = np.clip(X_train, -5, 5)
    X_val = np.clip(X_val, -5, 5)
    X_test = np.clip(X_test, -5, 5)
    
    console.print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Class weights
    class_weights = compute_class_weights(y_train, num_classes=3)
    
    # Create data loaders
    train_loader = DataLoader(DeepLOBDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(DeepLOBDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(DeepLOBDataset(X_test, y_test), batch_size=64, shuffle=False)
    
    # Train
    config = DeepLOBTrainingConfig(epochs=epochs, alpha_threshold=alpha_threshold)
    trainer = DeepLOBTrainer(config)
    trainer.feature_mean = feature_mean
    trainer.feature_std = feature_std
    
    model, history = trainer.train(train_loader, val_loader, class_weights, output_dir)
    results = trainer.evaluate_and_report(model, test_loader, output_dir)
    
    return model, results
