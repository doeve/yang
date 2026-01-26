"""
Probability Predictor for Polymarket Trading.

This model predicts P(YES | features) - a calibrated probability estimate
that the YES outcome will occur.

Key Design Principles:
1. Single output: P(YES) ∈ [0, 1]
2. No action classification - just probability estimation
3. Trained with Brier score for calibration
4. No hindsight information in features or targets
5. Post-hoc temperature scaling for calibration

The model does NOT predict:
- Actions (WAIT, BUY_YES, etc.)
- Expected returns
- Confidence in actions

Edge is computed externally: edge = P(YES) - market_price_yes
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ProbabilityPredictorConfig:
    """Configuration for probability predictor."""

    # Input dimension (from EnhancedFeatureBuilder)
    feature_dim: int = 51

    # Architecture
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    use_layer_norm: bool = True

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 100
    patience: int = 15

    # Calibration
    temperature: float = 1.0  # Post-hoc calibration parameter


class ProbabilityPredictorModel(nn.Module):
    """
    Simple probability predictor.

    Input: Features (batch, feature_dim)
    Output: P(YES) (batch, 1) in [0, 1]
    """

    def __init__(self, config: Optional[ProbabilityPredictorConfig] = None):
        super().__init__()
        self.config = config or ProbabilityPredictorConfig()

        # Build encoder
        layers = []
        in_dim = self.config.feature_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Single probability output
        self.prob_head = nn.Sequential(
            nn.Linear(self.config.hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Temperature for calibration (learnable or fixed)
        self.register_buffer('temperature', torch.tensor(self.config.temperature))

        self._init_weights()

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ProbabilityPredictorModel initialized",
            feature_dim=self.config.feature_dim,
            hidden_dims=self.config.hidden_dims,
            num_params=num_params,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, feature_dim)

        Returns:
            P(YES) (batch, 1) - probability in [0, 1]
        """
        h = self.encoder(x)
        logit = self.prob_head(h)

        # Apply temperature scaling for calibration
        scaled_logit = logit / self.temperature

        # Sigmoid to get probability
        prob = torch.sigmoid(scaled_logit)

        return prob

    def predict_probability(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get P(YES) estimate.

        Args:
            features: Market features (batch, feature_dim)

        Returns:
            P(YES) (batch,) - probability estimate
        """
        self.eval()
        with torch.no_grad():
            prob = self.forward(features)
            return prob.squeeze(-1)

    def get_logit(self, features: torch.Tensor) -> torch.Tensor:
        """Get raw logit before temperature scaling (for calibration)."""
        self.eval()
        with torch.no_grad():
            h = self.encoder(features)
            logit = self.prob_head(h)
            return logit.squeeze(-1)

    def set_temperature(self, temperature: float):
        """Set calibration temperature."""
        self.temperature.fill_(temperature)
        logger.info(f"Temperature set to {temperature:.4f}")


class ProbabilityDataset(Dataset):
    """Dataset for training probability predictor."""

    def __init__(
        self,
        features: np.ndarray,
        outcomes: np.ndarray,
    ):
        self.features = torch.FloatTensor(features)
        self.outcomes = torch.FloatTensor(outcomes)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.outcomes[idx]


def brier_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Brier score loss.

    Brier score = mean((p - y)^2)

    This directly optimizes probability calibration.
    """
    return torch.mean((predictions.squeeze() - targets) ** 2)


def focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Focal loss for handling class imbalance.

    Focuses on hard examples by down-weighting easy ones.
    """
    bce = F.binary_cross_entropy(predictions.squeeze(), targets, reduction='none')
    pt = torch.where(targets == 1, predictions.squeeze(), 1 - predictions.squeeze())
    focal_weight = (1 - pt) ** gamma
    return torch.mean(alpha * focal_weight * bce)


class ProbabilityPredictorTrainer:
    """Trainer for probability predictor."""

    def __init__(
        self,
        config: Optional[ProbabilityPredictorConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or ProbabilityPredictorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ProbabilityPredictorTrainer initialized", device=self.device)

    def train(
        self,
        train_features: np.ndarray,
        train_outcomes: np.ndarray,
        val_features: np.ndarray,
        val_outcomes: np.ndarray,
        output_dir: str = "./logs/probability_predictor",
        use_focal_loss: bool = False,
    ) -> Tuple["ProbabilityPredictorModel", Dict[str, Any]]:
        """Train the probability predictor model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create datasets
        train_dataset = ProbabilityDataset(train_features, train_outcomes)
        val_dataset = ProbabilityDataset(val_features, val_outcomes)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Create model
        self.config.feature_dim = train_features.shape[1]
        model = ProbabilityPredictorModel(self.config).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss function
        loss_fn = focal_loss if use_focal_loss else brier_loss

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_brier': [],
            'val_accuracy': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting training...", epochs=self.config.epochs)

        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []

            for features, outcomes in train_loader:
                features = features.to(self.device)
                outcomes = outcomes.to(self.device)

                optimizer.zero_grad()
                predictions = model(features)
                loss = loss_fn(predictions, outcomes)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for features, outcomes in val_loader:
                    features = features.to(self.device)
                    outcomes = outcomes.to(self.device)

                    predictions = model(features)
                    loss = loss_fn(predictions, outcomes)

                    val_losses.append(loss.item())
                    all_preds.extend(predictions.squeeze().cpu().numpy())
                    all_targets.extend(outcomes.cpu().numpy())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            # Compute metrics
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            brier = np.mean((all_preds - all_targets) ** 2)
            accuracy = np.mean((all_preds > 0.5) == all_targets)

            scheduler.step(avg_val_loss)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_brier'].append(brier)
            history['val_accuracy'].append(accuracy)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_path / "best_model.pt")
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}",
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{avg_val_loss:.4f}",
                    brier=f"{brier:.4f}",
                    accuracy=f"{accuracy:.3f}",
                )

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))

        # Save final model and config
        torch.save(model.state_dict(), output_path / "final_model.pt")
        with open(output_path / "config.json", "w") as f:
            json.dump({
                "feature_dim": int(train_features.shape[1]),
                "hidden_dims": list(self.config.hidden_dims),
                "dropout": self.config.dropout,
                "use_layer_norm": self.config.use_layer_norm,
                "temperature": float(model.temperature.item()),
            }, f, indent=2)

        logger.info(
            "Training complete",
            best_val_loss=f"{best_val_loss:.4f}",
            final_brier=f"{history['val_brier'][-1]:.4f}",
        )

        return model, history


def calibrate_temperature(
    model: ProbabilityPredictorModel,
    val_features: np.ndarray,
    val_outcomes: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Calibrate model temperature using validation set.

    Uses Platt scaling: find temperature T that minimizes NLL on validation set.

    Args:
        model: Trained model
        val_features: Validation features
        val_outcomes: Validation outcomes (0 or 1)
        device: Device to use

    Returns:
        Optimal temperature
    """
    model.eval()
    model.to(device)

    features_t = torch.FloatTensor(val_features).to(device)
    outcomes_t = torch.FloatTensor(val_outcomes).to(device)

    # Get raw logits (before temperature)
    with torch.no_grad():
        h = model.encoder(features_t)
        logits = model.prob_head(h).squeeze(-1)

    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        probs = torch.sigmoid(scaled_logits)
        # NLL loss
        loss = F.binary_cross_entropy(probs, outcomes_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temp = float(temperature.item())
    logger.info(f"Calibrated temperature: {optimal_temp:.4f}")

    return optimal_temp


def compute_calibration_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration metrics.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes (0 or 1)
        n_bins: Number of bins for ECE

    Returns:
        Dictionary with:
            - brier_score: Brier score
            - accuracy: Classification accuracy
            - ece: Expected Calibration Error
            - reliability_diagram: (bin_confidences, bin_accuracies, bin_counts)
    """
    # Brier score
    brier = np.mean((predictions - outcomes) ** 2)

    # Accuracy
    accuracy = np.mean((predictions > 0.5) == outcomes)

    # ECE and reliability diagram
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_conf = predictions[mask].mean()
            bin_acc = outcomes[mask].mean()
            bin_weight = mask.sum() / len(predictions)
            ece += bin_weight * abs(bin_acc - bin_conf)

            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            bin_counts.append(mask.sum())
        else:
            bin_confidences.append(None)
            bin_accuracies.append(None)
            bin_counts.append(0)

    # Correlation between predictions and outcomes
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, outcomes)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    return {
        "brier_score": brier,
        "accuracy": accuracy,
        "ece": ece,
        "correlation": correlation,
        "reliability_diagram": {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        },
    }


def load_probability_predictor(
    model_dir: str,
    device: Optional[str] = None,
) -> ProbabilityPredictorModel:
    """Load trained probability predictor model."""
    model_path = Path(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_path / "config.json", "r") as f:
        config_dict = json.load(f)

    config = ProbabilityPredictorConfig(
        feature_dim=config_dict["feature_dim"],
        hidden_dims=tuple(config_dict["hidden_dims"]),
        dropout=config_dict.get("dropout", 0.2),
        use_layer_norm=config_dict.get("use_layer_norm", True),
        temperature=config_dict.get("temperature", 1.0),
    )

    model = ProbabilityPredictorModel(config)
    model.load_state_dict(
        torch.load(model_path / "final_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Set calibrated temperature
    if "temperature" in config_dict:
        model.set_temperature(config_dict["temperature"])

    return model
