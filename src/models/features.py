"""
Temporal feature extractors for ML models.

Provides neural network modules for encoding price history:
- LSTM-based feature extractor
- Transformer-based feature extractor
- Attention mechanisms for important timesteps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Adds positional information to input sequences so the model
    can understand the temporal ordering of observations.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttention(nn.Module):
    """
    Attention mechanism for temporal sequences.
    
    Allows the model to focus on the most relevant timesteps
    in the price history.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Normalize to get weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Compute weighted sum
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return context, weights


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for temporal price data.
    
    Architecture:
    - Input projection to hidden dimension
    - 2-layer bidirectional LSTM
    - Temporal attention for importance weighting
    - Output projection to feature dimension
    
    This is the default feature extractor as it is:
    - More sample-efficient for short sequences
    - Faster training and inference
    - Well-suited for online trading scenarios
    
    Example:
        extractor = LSTMFeatureExtractor(input_dim=15, hidden_dim=128, output_dim=64)
        features = extractor(price_sequence)  # (batch, seq_len, input_dim) -> (batch, output_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: LSTM hidden dimension
            output_dim: Output feature dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Attention
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(lstm_output_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Feature tensor of shape (batch, output_dim)
            Optionally also attention weights of shape (batch, seq_len)
        """
        # Input projection
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * directions)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Output projection
        features = self.output_proj(context)
        
        if return_attention:
            return features, attention_weights
        return features


class TransformerFeatureExtractor(nn.Module):
    """
    Transformer-based feature extractor for temporal price data.
    
    Architecture:
    - Input projection with positional encoding
    - Multi-head self-attention layers
    - Mean pooling or CLS token for sequence representation
    - Output projection
    
    Advantages over LSTM:
    - Better at capturing long-range dependencies
    - More parallelizable for training
    - Better performance on longer sequences
    
    Example:
        extractor = TransformerFeatureExtractor(input_dim=15, output_dim=64)
        features = extractor(price_sequence)
    """
    
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        use_cls_token: bool = True,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: Transformer hidden dimension (d_model)
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_cls_token: Whether to use CLS token for pooling
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_cls_token = use_cls_token
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # CLS token (learnable)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        # Pooling
        if self.use_cls_token:
            # Use CLS token representation
            pooled = x[:, 0, :]
        else:
            # Mean pooling
            pooled = x.mean(dim=1)
        
        # Output projection
        features = self.output_proj(pooled)
        
        return features


class HybridFeatureExtractor(nn.Module):
    """
    Hybrid feature extractor combining LSTM and Transformer.
    
    Uses LSTM to capture local patterns and Transformer for
    global context, then combines both representations.
    """
    
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # LSTM branch
        self.lstm_extractor = LSTMFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=output_dim // 2,
            num_layers=1,
            dropout=dropout,
        )
        
        # Transformer branch
        self.transformer_extractor = TransformerFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=output_dim // 2,
            num_heads=2,
            num_layers=1,
            dropout=dropout,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        # Get features from both branches
        lstm_features = self.lstm_extractor(x)
        transformer_features = self.transformer_extractor(x)
        
        # Concatenate and fuse
        combined = torch.cat([lstm_features, transformer_features], dim=-1)
        features = self.fusion(combined)
        
        return features


def create_feature_extractor(
    extractor_type: str = "lstm",
    input_dim: int = 15,
    hidden_dim: int = 128,
    output_dim: int = 64,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create feature extractors.
    
    Args:
        extractor_type: One of "lstm", "transformer", "hybrid"
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        output_dim: Output feature dimension
        **kwargs: Additional arguments for the extractor
        
    Returns:
        Feature extractor module
    """
    if extractor_type == "lstm":
        return LSTMFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs,
        )
    elif extractor_type == "transformer":
        return TransformerFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs,
        )
    elif extractor_type == "hybrid":
        return HybridFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
