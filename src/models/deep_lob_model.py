"""
DeepLOB-style CNN-LSTM model for limit order book prediction.

Based on:
- Zhang et al. 2019: "DeepLOB: Deep CNNs for Limit Order Books"
- Kolm et al. 2021: "Deep order flow imbalance"

Architecture:
1. CNN layers for spatial patterns (across feature dimensions)
2. LSTM for temporal dependencies
3. Multi-head attention for focus on important timesteps
4. 3-class output (Down, Hold, Up) with focal loss
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DeepLOBConfig:
    """Configuration for DeepLOB model."""
    
    input_dim: int = 34           # Number of features
    sequence_length: int = 120    # Sequence length
    
    # CNN parameters
    cnn_filters: list = None      # Filters for each conv layer
    cnn_kernel_size: int = 3
    
    # LSTM parameters
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    
    # Attention
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Output
    num_classes: int = 3          # Down, Hold, Up
    
    # Regularization
    dropout: float = 0.4
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 32, 64]


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str = 'same',
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    """
    Inception-like module for multi-scale pattern extraction.
    
    Uses parallel convolutions with different kernel sizes.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Split output channels across branches
        branch_channels = out_channels // 4
        
        # 1x1 conv (point-wise)
        self.branch1 = nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        
        # 1x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        
        # 1x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.activation(self.bn(out))


class DeepLOBModel(nn.Module):
    """
    DeepLOB-style CNN-LSTM model for 3-class prediction.
    
    Architecture:
    1. Input: (batch, seq_len, features)
    2. CNN layers: extract spatial patterns across features
    3. LSTM: capture temporal dependencies
    4. Multi-head attention: focus on important timesteps
    5. Output: 3-class logits (Down, Hold, Up)
    """
    
    def __init__(self, config: Optional[DeepLOBConfig] = None):
        super().__init__()
        self.config = config or DeepLOBConfig()
        
        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.cnn_filters[0])
        
        # CNN layers
        cnn_layers = []
        in_channels = self.config.cnn_filters[0]
        for out_channels in self.config.cnn_filters:
            cnn_layers.append(ConvBlock(in_channels, out_channels, self.config.cnn_kernel_size))
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Inception module for multi-scale patterns
        self.inception = InceptionModule(in_channels, in_channels)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.config.lstm_hidden,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.lstm_dropout if self.config.lstm_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.lstm_hidden,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.config.lstm_hidden)
        
        # Output layers
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc1 = nn.Linear(self.config.lstm_hidden, self.config.lstm_hidden // 2)
        self.fc2 = nn.Linear(self.config.lstm_hidden // 2, self.config.num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Log model info
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DeepLOBModel initialized",
            input_dim=self.config.input_dim,
            num_params=num_params,
            num_classes=self.config.num_classes,
        )
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Input projection: (batch, seq, features) -> (batch, seq, cnn_filters[0])
        x = self.input_proj(x)
        
        # CNN: transpose for Conv1d (batch, channels, seq)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.inception(x)
        x = x.transpose(1, 2)  # Back to (batch, seq, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Self-attention over LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)  # Residual connection
        
        # Take the last timestep
        out = attn_out[:, -1, :]  # (batch, hidden)
        
        # Output layers
        out = self.dropout(out)
        out = F.leaky_relu(self.fc1(out), 0.1)
        out = self.dropout(out)
        logits = self.fc2(out)  # (batch, num_classes)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses on hard examples by down-weighting easy ones.
    Particularly useful for "Hold" class which is often majority.
    
    FL(p) = -α(1-p)^γ * log(p)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (batch, num_classes)
            targets: Class indices of shape (batch,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
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
        else:
            return loss


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights inversely proportional to frequency.
    
    Helps balance training when classes are imbalanced (e.g., many "Hold").
    """
    valid_mask = y >= 0
    y_valid = y[valid_mask]
    
    counts = np.bincount(y_valid, minlength=num_classes)
    total = counts.sum()
    
    # Inverse frequency weighting
    weights = total / (num_classes * counts + 1)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    logger.info(
        "Computed class weights",
        counts=counts.tolist(),
        weights=weights.tolist(),
    )
    
    return torch.FloatTensor(weights)


def create_deep_lob_model(config: Optional[DeepLOBConfig] = None) -> DeepLOBModel:
    """Factory function to create DeepLOB model."""
    return DeepLOBModel(config)
