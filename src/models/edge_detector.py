"""
Edge Detection Model for Polymarket Token Trading.

This model estimates P(YES wins at settlement) from token price features,
then computes the "edge" (mispricing) as:

    edge = P(YES wins) - current_YES_price

If edge > 0: YES is underpriced (buy opportunity)
If edge < 0: NO is underpriced (sell YES / buy NO)

Architecture:
- MLP with residual connections for tabular data
- Calibrated probability output (Platt scaling optional)
- Confidence estimation

Training:
- Binary classification: YES wins (1) vs NO wins (0)
- Focal loss for handling class imbalance
- Time-aware sampling (more weight on late-candle predictions)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EdgeDetectorConfig:
    """Configuration for edge detector model."""

    # Architecture
    input_dim: int = 49  # From TokenFeatureBuilder
    hidden_dims: Tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = True

    # Output
    num_classes: int = 2  # Binary: YES wins / NO wins

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0  # Focal loss focusing parameter


class ResidualBlock(nn.Module):
    """Residual block for MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.fc(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class EdgeDetectorModel(nn.Module):
    """
    Edge detection model for mispricing identification.

    Predicts P(YES wins at settlement) from token features.
    """

    def __init__(self, config: Optional[EdgeDetectorConfig] = None):
        super().__init__()
        self.config = config or EdgeDetectorConfig()

        # Build layers
        layers = []
        in_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            if self.config.use_residual:
                layers.append(ResidualBlock(
                    in_dim, hidden_dim,
                    dropout=self.config.dropout,
                    use_batch_norm=self.config.use_batch_norm,
                ))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                if self.config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.classifier = nn.Linear(in_dim, self.config.num_classes)
        self.confidence = nn.Linear(in_dim, 1)  # Confidence estimation

        # Initialize weights
        self._init_weights()

        # Log model info
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "EdgeDetectorModel initialized",
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            num_params=num_params,
        )

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Dictionary with:
                - logits: Raw logits (batch, 2)
                - probs: Softmax probabilities (batch, 2)
                - p_yes: P(YES wins) (batch,)
                - confidence: Confidence score (batch,)
        """
        # Encode features
        h = self.encoder(x)

        # Classification logits
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)

        # P(YES wins) = probability of class 1
        p_yes = probs[:, 1]

        # Confidence (separate from probability)
        confidence = torch.sigmoid(self.confidence(h)).squeeze(-1)

        return {
            'logits': logits,
            'probs': probs,
            'p_yes': p_yes,
            'confidence': confidence,
        }

    def predict_edge(
        self,
        features: torch.Tensor,
        current_yes_price: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict edge (mispricing) given features and current price.

        Args:
            features: Token features (batch, input_dim)
            current_yes_price: Current YES token price

        Returns:
            Dictionary with edge metrics
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features)

            # Edge = P(YES wins) - current_yes_price
            edge = outputs['p_yes'] - current_yes_price

            # Edge direction
            edge_direction = torch.sign(edge)

            # Recommended action
            # Positive edge = buy YES, Negative edge = buy NO
            action = torch.where(
                edge > 0.05,  # Threshold for significance
                torch.ones_like(edge),  # Buy YES
                torch.where(
                    edge < -0.05,
                    -torch.ones_like(edge),  # Buy NO
                    torch.zeros_like(edge),  # Hold
                )
            )

        return {
            'p_yes': outputs['p_yes'],
            'edge': edge,
            'edge_direction': edge_direction,
            'confidence': outputs['confidence'],
            'action': action,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p) = -α(1-p)^γ * log(p)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_weight = alpha[targets]
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EdgeDetectorDataset(Dataset):
    """Dataset for edge detector training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, time_weights: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.time_weights = torch.FloatTensor(time_weights) if time_weights is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.time_weights is not None:
            return self.X[idx], self.y[idx], self.time_weights[idx]
        return self.X[idx], self.y[idx]


class EdgeDetectorTrainer:
    """Trainer for edge detector model."""

    def __init__(
        self,
        config: Optional[EdgeDetectorConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or EdgeDetectorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("EdgeDetectorTrainer initialized", device=self.device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 20,
        output_dir: str = "./logs/edge_detector",
    ) -> Tuple[EdgeDetectorModel, Dict[str, Any]]:
        """
        Train the edge detector model.

        Args:
            X_train: Training features
            y_train: Training labels (0 = NO wins, 1 = YES wins)
            X_val: Validation features
            y_val: Validation labels
            epochs: Max training epochs
            batch_size: Batch size
            patience: Early stopping patience
            output_dir: Output directory

        Returns:
            (trained_model, training_history)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create datasets
        train_dataset = EdgeDetectorDataset(X_train, y_train)
        val_dataset = EdgeDetectorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        self.config.input_dim = X_train.shape[1]
        model = EdgeDetectorModel(self.config).to(self.device)

        # Loss and optimizer
        criterion = FocalLoss(gamma=self.config.focal_gamma)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_brier': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting edge detector training...", epochs=epochs)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []

            for batch in train_loader:
                X_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs['logits'], y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            scheduler.step()

            # Validation phase
            model.eval()
            val_losses = []
            all_preds = []
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    X_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

                    outputs = model(X_batch)
                    loss = criterion(outputs['logits'], y_batch)

                    val_losses.append(loss.item())
                    all_preds.append(torch.argmax(outputs['logits'], dim=-1).cpu())
                    all_probs.append(outputs['p_yes'].cpu())
                    all_targets.append(y_batch.cpu())

            all_preds = torch.cat(all_preds).numpy()
            all_probs = torch.cat(all_probs).numpy()
            all_targets = torch.cat(all_targets).numpy()

            # Metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            accuracy = (all_preds == all_targets).mean()
            brier_score = np.mean((all_probs - all_targets) ** 2)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(accuracy)
            history['val_brier'].append(brier_score)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_path / "best_model.pt")
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}",
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{avg_val_loss:.4f}",
                    accuracy=f"{accuracy:.3f}",
                    brier=f"{brier_score:.4f}",
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))

        # Save final model and config
        torch.save(model.state_dict(), output_path / "final_model.pt")
        with open(output_path / "config.json", "w") as f:
            json.dump({
                "model_config": self.config.__dict__,
                "input_dim": int(X_train.shape[1]),
            }, f, indent=2, default=str)

        logger.info(
            "Training complete",
            best_val_loss=f"{best_val_loss:.4f}",
            final_accuracy=f"{history['val_accuracy'][-1]:.3f}",
        )

        return model, history

    def evaluate(
        self,
        model: EdgeDetectorModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        model.eval()
        model.to(self.device)

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test)

        with torch.no_grad():
            outputs = model(X_tensor)
            probs = outputs['p_yes'].cpu().numpy()
            preds = (probs > 0.5).astype(int)

        accuracy = (preds == y_test).mean()
        brier_score = np.mean((probs - y_test) ** 2)

        # Calibration (binned)
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(probs, bins) - 1
        calibration_error = 0.0
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_prob = probs[mask].mean()
                bin_outcome = y_test[mask].mean()
                calibration_error += abs(bin_prob - bin_outcome) * mask.sum() / len(y_test)

        return {
            "accuracy": float(accuracy),
            "brier_score": float(brier_score),
            "calibration_error": float(calibration_error),
        }


def load_edge_detector(
    model_dir: str,
    device: Optional[str] = None,
) -> EdgeDetectorModel:
    """Load trained edge detector model."""
    model_path = Path(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(model_path / "config.json", "r") as f:
        config_dict = json.load(f)

    config = EdgeDetectorConfig(input_dim=config_dict["input_dim"])
    model = EdgeDetectorModel(config)
    model.load_state_dict(torch.load(model_path / "final_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model
