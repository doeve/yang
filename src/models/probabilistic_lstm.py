"""
Probabilistic LSTM for calibrated probability prediction.

Predicts P(BTC_close_15m > strike | current_info) with proper calibration.
Designed for Polymarket 15-minute crypto prediction markets.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ProbabilisticLSTMConfig:
    """Configuration for probabilistic LSTM model."""
    
    # Input dimensions
    input_dim: int = 24              # Multi-resolution features
    sequence_length: int = 180       # ~3 hours of 1-minute data
    
    # Architecture (kept small to prevent overfitting)
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3             # Heavy regularization
    bidirectional: bool = False      # Causal only for production
    
    # Output
    output_logits: bool = True       # Return logits for temperature scaling
    
    # Regularization
    weight_decay: float = 1e-4
    layer_norm: bool = True


class ProbabilisticLSTM(nn.Module):
    """
    LSTM-based probability estimator for binary outcomes.
    
    Key design principles:
    - Small architecture to prevent overfitting on noisy market data
    - Outputs logits for temperature scaling calibration
    - Heavy dropout and layer normalization
    - Causal (unidirectional) for production inference
    
    Target metric: Brier score, NOT accuracy
    Expected accuracy: 52-55% (realistic for noisy markets)
    """
    
    def __init__(self, config: Optional[ProbabilisticLSTMConfig] = None):
        super().__init__()
        self.config = config or ProbabilisticLSTMConfig()
        
        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        
        # Layer normalization
        if self.config.layer_norm:
            self.input_ln = nn.LayerNorm(self.config.hidden_dim)
        else:
            self.input_ln = nn.Identity()
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )
        
        # Output dimension after LSTM
        lstm_output_dim = self.config.hidden_dim * (2 if self.config.bidirectional else 1)
        
        # Output layers
        self.output_ln = nn.LayerNorm(lstm_output_dim) if self.config.layer_norm else nn.Identity()
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Two-layer output head for better representation
        self.fc1 = nn.Linear(lstm_output_dim, self.config.hidden_dim)
        self.fc2 = nn.Linear(self.config.hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            "ProbabilisticLSTM initialized",
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_params=sum(p.numel() for p in self.parameters()),
        )
    
    def _init_weights(self):
        """Initialize weights with Xavier/orthogonal initialization."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_hidden: If True, also return final hidden state
            
        Returns:
            Logits of shape (batch,) if output_logits=True
            Probabilities of shape (batch,) if output_logits=False
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.config.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]
        
        # Output layers
        hidden = self.output_ln(hidden)
        hidden = self.dropout(hidden)
        
        out = F.relu(self.fc1(hidden))
        out = self.dropout(out)
        logits = self.fc2(out).squeeze(-1)
        
        if return_hidden:
            return logits, hidden
        
        if self.config.output_logits:
            return logits
        else:
            return torch.sigmoid(logits)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probability predictions.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Probabilities in [0, 1], shape (batch,)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_uncertainty(self, x: torch.Tensor, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty via MC Dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes
            
        Returns:
            Tuple of (mean_prob, std_prob)
        """
        self.train()  # Enable dropout
        
        probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs.append(torch.sigmoid(logits))
        
        probs = torch.stack(probs, dim=0)
        mean_prob = probs.mean(dim=0)
        std_prob = probs.std(dim=0)
        
        return mean_prob, std_prob


class ProbabilisticGRU(nn.Module):
    """
    GRU variant of the probabilistic model.
    
    GRU is faster and often performs comparably to LSTM
    on shorter sequences.
    """
    
    def __init__(self, config: Optional[ProbabilisticLSTMConfig] = None):
        super().__init__()
        self.config = config or ProbabilisticLSTMConfig()
        
        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.input_ln = nn.LayerNorm(self.config.hidden_dim) if self.config.layer_norm else nn.Identity()
        
        # GRU
        self.gru = nn.GRU(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )
        
        gru_output_dim = self.config.hidden_dim * (2 if self.config.bidirectional else 1)
        
        self.output_ln = nn.LayerNorm(gru_output_dim) if self.config.layer_norm else nn.Identity()
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc1 = nn.Linear(gru_output_dim, self.config.hidden_dim)
        self.fc2 = nn.Linear(self.config.hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        gru_out, h_n = self.gru(x)
        
        if self.config.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]
        
        hidden = self.output_ln(hidden)
        hidden = self.dropout(hidden)
        
        out = F.relu(self.fc1(hidden))
        out = self.dropout(out)
        logits = self.fc2(out).squeeze(-1)
        
        if self.config.output_logits:
            return logits
        else:
            return torch.sigmoid(logits)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


def create_probability_model(
    model_type: str = "lstm",
    config: Optional[ProbabilisticLSTMConfig] = None,
) -> nn.Module:
    """
    Factory function to create probability models.
    
    Args:
        model_type: 'lstm' or 'gru'
        config: Model configuration
        
    Returns:
        Probability model instance
    """
    if model_type.lower() == "lstm":
        return ProbabilisticLSTM(config)
    elif model_type.lower() == "gru":
        return ProbabilisticGRU(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
