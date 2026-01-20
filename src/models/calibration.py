"""
Probability calibration utilities for trading models.

Provides:
- Temperature scaling for post-hoc probability calibration
- Brier score loss function for training
- Calibration curve plotting and Expected Calibration Error (ECE)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)


class BrierScoreLoss(nn.Module):
    """
    Brier score loss for probability calibration.
    
    Brier score = mean((predicted_prob - actual_outcome)^2)
    
    Lower is better. Perfect calibration: 0.0
    Random baseline (0.5 always): 0.25
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            probs: Predicted probabilities in [0, 1], shape (batch,)
            targets: Binary outcomes {0, 1}, shape (batch,)
            
        Returns:
            Brier score (scalar if reduction='mean')
        """
        # Ensure proper shapes
        probs = probs.view(-1)
        targets = targets.view(-1).float()
        
        # Brier score
        score = (probs - targets) ** 2
        
        if self.reduction == "mean":
            return score.mean()
        elif self.reduction == "sum":
            return score.sum()
        else:
            return score


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter T such that:
        calibrated_prob = sigmoid(logit / T)
    
    T > 1: softens probabilities (less confident)
    T < 1: sharpens probabilities (more confident)
    
    Training: Freeze base model, optimize T on validation set.
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            
        Returns:
            Calibrated probabilities
        """
        return torch.sigmoid(logits / self.temperature)
    
    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calibrate already-sigmoided probabilities.
        
        Converts probs back to logits, scales, and re-applies sigmoid.
        """
        # Clamp to avoid numerical issues
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        # Convert to logits
        logits = torch.log(probs / (1 - probs))
        
        # Apply temperature scaling
        return self.forward(logits)
    
    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_iterations: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Fit temperature parameter on held-out validation data.
        
        Args:
            logits: Model logits on validation set
            targets: True binary labels
            num_iterations: Optimization iterations
            lr: Learning rate
            
        Returns:
            Final temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=num_iterations)
        criterion = nn.BCELoss()
        
        def closure():
            optimizer.zero_grad()
            probs = self.forward(logits)
            loss = criterion(probs, targets.float())
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        logger.info(
            "Temperature scaling fitted",
            temperature=self.temperature.item(),
        )
        
        return self.temperature.item()


def compute_expected_calibration_error(
    probs: np.ndarray,
    outcomes: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum(|accuracy - confidence| * bin_size) for each bin
    
    Args:
        probs: Predicted probabilities
        outcomes: True binary outcomes
        num_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE, bin_accuracies, bin_confidences)
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracies[i] = outcomes[mask].mean()
            bin_confidences[i] = probs[mask].mean()
            bin_counts[i] = mask.sum()
    
    # Weighted ECE
    total_samples = len(probs)
    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * bin_counts) / total_samples
    
    return ece, bin_accuracies, bin_confidences


def compute_calibration_metrics(
    probs: np.ndarray,
    outcomes: np.ndarray,
) -> dict:
    """
    Compute comprehensive calibration metrics.
    
    Returns dict with:
    - brier_score: Mean squared error of probabilities
    - accuracy: Classification accuracy (threshold 0.5)
    - ece: Expected Calibration Error
    - log_loss: Negative log likelihood
    - reliability: How close to diagonal on reliability diagram
    """
    # Brier score
    brier_score = np.mean((probs - outcomes) ** 2)
    
    # Accuracy
    predictions = (probs >= 0.5).astype(int)
    accuracy = np.mean(predictions == outcomes)
    
    # ECE
    ece, _, _ = compute_expected_calibration_error(probs, outcomes)
    
    # Log loss (with clipping to avoid inf)
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    log_loss = -np.mean(
        outcomes * np.log(probs_clipped) + 
        (1 - outcomes) * np.log(1 - probs_clipped)
    )
    
    # Baseline Brier score (predicting class frequency)
    baseline_prob = outcomes.mean()
    baseline_brier = np.mean((baseline_prob - outcomes) ** 2)
    
    # Brier skill score (improvement over baseline)
    brier_skill = 1 - (brier_score / baseline_brier) if baseline_brier > 0 else 0
    
    return {
        "brier_score": float(brier_score),
        "accuracy": float(accuracy),
        "ece": float(ece),
        "log_loss": float(log_loss),
        "brier_skill": float(brier_skill),
        "baseline_brier": float(baseline_brier),
    }


def plot_calibration_curve(
    probs: np.ndarray,
    outcomes: np.ndarray,
    num_bins: int = 10,
    title: str = "Calibration Curve",
) -> "matplotlib.figure.Figure":
    """
    Plot reliability diagram (calibration curve).
    
    A well-calibrated model follows the diagonal.
    """
    import matplotlib.pyplot as plt
    
    ece, bin_accs, bin_confs = compute_expected_calibration_error(
        probs, outcomes, num_bins
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(bin_confs, bin_accs, "o-", label=f"Model (ECE={ece:.3f})")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(probs, bins=50, density=True, alpha=0.7)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


class CalibrationWrapper(nn.Module):
    """
    Wrapper that adds temperature scaling to any probability model.
    
    Usage:
        base_model = ProbabilisticLSTM(...)
        calibrated = CalibrationWrapper(base_model)
        calibrated.calibrate(val_logits, val_targets)
        
        # Now use calibrated model
        probs = calibrated(x)  # Returns calibrated probabilities
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        initial_temperature: float = 1.5,
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature_scaling = TemperatureScaling(initial_temperature)
        self.is_calibrated = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional calibration."""
        # Get base model output (logits or probs depending on model)
        output = self.base_model(x)
        
        if self.is_calibrated:
            # Assume output is logits, apply temperature scaling
            return self.temperature_scaling(output)
        else:
            # Return raw sigmoid of output
            return torch.sigmoid(output)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits from base model."""
        return self.base_model(x)
    
    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> float:
        """
        Calibrate temperature on validation data.
        
        Args:
            val_loader: DataLoader with validation data
            device: Device to run calibration on
            
        Returns:
            Fitted temperature value
        """
        self.base_model.eval()
        
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = self.base_model(x)
                all_logits.append(logits.cpu())
                all_targets.append(y)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        temp = self.temperature_scaling.fit(all_logits, all_targets)
        self.is_calibrated = True
        
        return temp
