"""
Attention-Based Market Predictor with KAMA Spectrum.

This module implements a time-aware attention mechanism that dynamically weights
Kaufman Adaptive Moving Averages (KAMAs) based on market regime and time remaining.

Key Architecture:
1. TimeAwareKAMAAttention: Uses context (time, strike, volatility) to compute
   softmax attention weights over KAMA spectrum
2. AttentionMarketPredictorModel: Full model with attention-weighted feature fusion
3. AttentionMarketPredictorTrainer: Training loop with attention weight logging

For 15-minute binary BTC markets:
- Early in candle: Model learns to attend to slow KAMAs (50-80)
- Near expiration: Model learns to attend to fast KAMAs (5-10)
- Near strike (0.50): Model adapts based on volatility
"""

from dataclasses import dataclass, field
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


# Import Action class from existing module
from src.models.market_predictor import Action


@dataclass
class AttentionMarketPredictorConfig:
    """Configuration for attention-based market predictor."""
    
    # KAMA spectrum dimensions
    num_kamas: int = 12  # Number of KAMA periods
    kama_features_per_period: int = 2  # deviation + slope per KAMA
    
    # Context dimensions
    context_dim: int = 12  # Time(5) + Strike(4) + Volatility(3)
    
    # Base features (from EnhancedFeatureBuilder if used)
    base_feature_dim: int = 71  # Original features (can be 0 if not using)
    
    # Position state
    position_state_dim: int = 6
    
    # Attention network architecture
    attention_hidden_dim: int = 64
    attention_dropout: float = 0.1
    
    # Main network architecture
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    use_layer_norm: bool = True
    
    # Output
    num_actions: int = 5  # WAIT, BUY_YES, BUY_NO, EXIT, HOLD
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    
    # Loss weighting
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    
    @property
    def kama_spectrum_dim(self) -> int:
        """Total KAMA spectrum feature dimension."""
        return self.num_kamas * self.kama_features_per_period


class TimeAwareKAMAAttention(nn.Module):
    """
    Attention layer that uses time/strike/volatility context to weight KAMA features.
    
    The attention mechanism learns to:
    - Near expiration (low time_remaining): shift weights to fast KAMAs
    - Far from strike: focus on trending KAMAs
    - High volatility: prefer adaptive/medium KAMAs
    """
    
    def __init__(
        self,
        context_dim: int = 12,
        num_kamas: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_kamas = num_kamas
        
        # Context processing network
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        
        # Output attention weights for each KAMA
        self.attention_head = nn.Linear(hidden_dim // 2, num_kamas)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize attention head to produce uniform weights initially
        nn.init.zeros_(self.attention_head.weight)
        nn.init.zeros_(self.attention_head.bias)
    
    def forward(
        self,
        context: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention weights over KAMA features.
        
        Args:
            context: (batch, context_dim) - time/strike/vol features
            return_logits: If True, also return raw logits before softmax
            
        Returns:
            weights: (batch, num_kamas) - softmax attention weights
        """
        h = self.context_net(context)
        logits = self.attention_head(h)
        weights = F.softmax(logits, dim=-1)
        
        if return_logits:
            return weights, logits
        return weights


class AttentionMarketPredictorModel(nn.Module):
    """
    Market predictor with time-aware KAMA attention.
    
    Architecture:
    1. Extract KAMA spectrum and context from input
    2. Compute attention weights from context
    3. Apply attention to KAMA spectrum → weighted trend signal
    4. Concatenate with base features and position state
    5. Process through MLP prediction head
    
    Outputs:
    - action_logits: Logits for each action
    - expected_return: Predicted return for action
    - confidence: Model confidence in prediction
    - attention_weights: For analysis/logging
    """
    
    def __init__(self, config: Optional[AttentionMarketPredictorConfig] = None):
        super().__init__()
        self.config = config or AttentionMarketPredictorConfig()
        
        # Attention module
        self.attention = TimeAwareKAMAAttention(
            context_dim=self.config.context_dim,
            num_kamas=self.config.num_kamas,
            hidden_dim=self.config.attention_hidden_dim,
            dropout=self.config.attention_dropout,
        )
        
        # Calculate input dimension for MLP
        # After attention: num_kamas * features_per_period get weighted summed → features_per_period
        attended_dim = self.config.kama_features_per_period
        
        mlp_input_dim = (
            attended_dim +                      # Weighted KAMA signal
            self.config.context_dim +           # Context features
            self.config.base_feature_dim +      # Base features (optional)
            self.config.position_state_dim      # Position state
        )
        
        # Feature encoder with layer norm
        encoder_layers = []
        in_dim = mlp_input_dim
        
        for hidden_dim in self.config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if self.config.use_layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        final_dim = self.config.hidden_dims[-1]
        
        # Action classification head
        self.action_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.config.num_actions),
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Expected return head
        self.expected_return_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Returns capped to [-1, 1]
        )
        
        self._init_weights()
        
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "AttentionMarketPredictorModel initialized",
            num_kamas=self.config.num_kamas,
            context_dim=self.config.context_dim,
            mlp_input_dim=mlp_input_dim,
            hidden_dims=self.config.hidden_dims,
            num_params=num_params,
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        kama_spectrum: torch.Tensor,
        context: torch.Tensor,
        base_features: torch.Tensor,
        position_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention over KAMA spectrum.
        
        Args:
            kama_spectrum: (batch, num_kamas * 2) - KAMA features
            context: (batch, context_dim) - Time/strike/vol context
            base_features: (batch, base_feature_dim) - Other features
            position_state: (batch, position_state_dim) - Position info
            
        Returns:
            Dictionary with action_logits, confidence, expected_return, 
            action_probs, attention_weights
        """
        batch_size = kama_spectrum.shape[0]
        
        # 1. Compute attention weights from context
        attn_weights = self.attention(context)  # (batch, num_kamas)
        
        # 2. Reshape spectrum: (batch, num_kamas, features_per_period)
        spectrum_reshaped = kama_spectrum.view(
            batch_size, 
            self.config.num_kamas, 
            self.config.kama_features_per_period
        )
        
        # 3. Apply attention: weighted sum over KAMA variants
        # attn_weights: (batch, num_kamas) → (batch, num_kamas, 1)
        weighted_spectrum = spectrum_reshaped * attn_weights.unsqueeze(-1)
        attended_signal = weighted_spectrum.sum(dim=1)  # (batch, features_per_period)
        
        # 4. Concatenate all features
        combined = torch.cat([
            attended_signal,    # Attended KAMA signal
            context,            # Context features (also useful for prediction)
            base_features,      # Base features
            position_state,     # Position info
        ], dim=-1)
        
        # 5. Encode through MLP
        h = self.encoder(combined)
        
        # 6. Compute outputs
        action_logits = self.action_head(h)
        action_probs = F.softmax(action_logits, dim=-1)
        confidence = self.confidence_head(h)
        expected_return = self.expected_return_head(h)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'confidence': confidence,
            'expected_return': expected_return,
            'attention_weights': attn_weights,
        }
    
    def forward_combined(
        self,
        combined_features: torch.Tensor,
        position_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with combined feature tensor.
        
        For compatibility with existing data pipeline that produces
        concatenated [kama_spectrum, context, base_features] tensors.
        
        Args:
            combined_features: Concatenated feature tensor
            position_state: Position state tensor
            
        Returns:
            Same as forward()
        """
        kama_dim = self.config.kama_spectrum_dim
        context_dim = self.config.context_dim
        
        kama_spectrum = combined_features[:, :kama_dim]
        context = combined_features[:, kama_dim:kama_dim + context_dim]
        base_features = combined_features[:, kama_dim + context_dim:]
        
        return self.forward(kama_spectrum, context, base_features, position_state)
    
    def get_action(
        self,
        kama_spectrum: torch.Tensor,
        context: torch.Tensor,
        base_features: torch.Tensor,
        position_state: torch.Tensor,
        deterministic: bool = True,
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get optimal action with action masking.
        
        Invalid actions are masked to -inf before selection.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(kama_spectrum, context, base_features, position_state)
            
            # Extract has_position flag
            has_position = position_state[:, 0] > 0.5
            
            # Apply action masking
            action_logits = outputs['action_logits'].clone()
            masked_logits = self._apply_action_mask(action_logits, has_position)
            
            if deterministic:
                if np.random.random() < epsilon:
                    action = self._sample_valid_random(has_position)
                else:
                    action = masked_logits.argmax(dim=-1)
            else:
                masked_probs = F.softmax(masked_logits, dim=-1)
                action = torch.multinomial(masked_probs, 1).squeeze(-1)
        
        return {
            'action': action,
            'action_logits': outputs['action_logits'],
            'masked_logits': masked_logits,
            'confidence': outputs['confidence'],
            'expected_return': outputs['expected_return'],
            'action_probs': F.softmax(masked_logits, dim=-1),
            'attention_weights': outputs['attention_weights'],
        }
    
    def _apply_action_mask(
        self,
        logits: torch.Tensor,
        has_position: torch.Tensor,
    ) -> torch.Tensor:
        """Apply action mask: set invalid action logits to -inf."""
        masked = logits.clone()
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            if has_position[i]:
                # In position: can only EXIT or HOLD
                masked[i, Action.WAIT] = float('-inf')
                masked[i, Action.BUY_YES] = float('-inf')
                masked[i, Action.BUY_NO] = float('-inf')
            else:
                # No position: can only WAIT, BUY_YES, or BUY_NO
                masked[i, Action.EXIT] = float('-inf')
                masked[i, Action.HOLD] = float('-inf')
        
        return masked
    
    def _sample_valid_random(self, has_position: torch.Tensor) -> torch.Tensor:
        """Sample uniformly from valid actions only."""
        batch_size = has_position.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=has_position.device)
        
        for i in range(batch_size):
            valid = Action.get_valid_actions(bool(has_position[i]))
            actions[i] = valid[np.random.randint(len(valid))]
        
        return actions


class AttentionMarketPredictorDataset(Dataset):
    """Dataset for training attention market predictor."""
    
    def __init__(
        self,
        kama_spectrum: np.ndarray,
        context_features: np.ndarray,
        base_features: np.ndarray,
        position_states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
    ):
        self.kama_spectrum = torch.FloatTensor(kama_spectrum)
        self.context_features = torch.FloatTensor(context_features)
        self.base_features = torch.FloatTensor(base_features)
        self.position_states = torch.FloatTensor(position_states)
        self.actions = torch.LongTensor(actions)
        self.returns = torch.FloatTensor(returns)
    
    def __len__(self) -> int:
        return len(self.kama_spectrum)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.kama_spectrum[idx],
            self.context_features[idx],
            self.base_features[idx],
            self.position_states[idx],
            self.actions[idx],
            self.returns[idx],
        )


class AttentionMarketPredictorTrainer:
    """
    Trainer for attention-based market predictor.
    
    Includes attention weight logging for analysis.
    """
    
    def __init__(
        self,
        config: Optional[AttentionMarketPredictorConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or AttentionMarketPredictorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("AttentionMarketPredictorTrainer initialized", device=self.device)
    
    def _apply_training_mask(
        self,
        logits: torch.Tensor,
        has_position: torch.Tensor,
    ) -> torch.Tensor:
        """Apply action mask during training."""
        masked = logits.clone()
        
        for i in range(len(logits)):
            if has_position[i]:
                masked[i, Action.WAIT] = float('-inf')
                masked[i, Action.BUY_YES] = float('-inf')
                masked[i, Action.BUY_NO] = float('-inf')
            else:
                masked[i, Action.EXIT] = float('-inf')
                masked[i, Action.HOLD] = float('-inf')
        
        return masked
    
    def _check_action_validity(
        self,
        actions: torch.Tensor,
        has_position: torch.Tensor,
    ) -> torch.Tensor:
        """Check which actions are invalid."""
        invalid = torch.zeros(len(actions), dtype=torch.bool, device=actions.device)
        
        for i in range(len(actions)):
            a = int(actions[i])
            hp = bool(has_position[i])
            if not Action.is_valid(a, hp):
                invalid[i] = True
        
        return invalid
    
    def train(
        self,
        train_kama: np.ndarray,
        train_context: np.ndarray,
        train_base: np.ndarray,
        train_position_states: np.ndarray,
        train_actions: np.ndarray,
        train_returns: np.ndarray,
        val_kama: np.ndarray,
        val_context: np.ndarray,
        val_base: np.ndarray,
        val_position_states: np.ndarray,
        val_actions: np.ndarray,
        val_returns: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 15,
        output_dir: str = "./logs/attention_market_predictor",
        attention_log_interval: int = 10,
    ) -> Tuple['AttentionMarketPredictorModel', Dict[str, Any]]:
        """
        Train the attention market predictor.
        
        Args:
            train_*/val_*: Training and validation data
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            output_dir: Output directory
            attention_log_interval: Log attention weights every N epochs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        train_dataset = AttentionMarketPredictorDataset(
            train_kama, train_context, train_base,
            train_position_states, train_actions, train_returns
        )
        val_dataset = AttentionMarketPredictorDataset(
            val_kama, val_context, val_base,
            val_position_states, val_actions, val_returns
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Update config with actual dimensions
        self.config.kama_features_per_period = train_kama.shape[1] // self.config.num_kamas
        self.config.context_dim = train_context.shape[1]
        self.config.base_feature_dim = train_base.shape[1]
        self.config.position_state_dim = train_position_states.shape[1]
        
        # Create model
        model = AttentionMarketPredictorModel(self.config).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss functions
        action_criterion = nn.CrossEntropyLoss(reduction='none')
        return_criterion = nn.SmoothL1Loss()
        confidence_criterion = nn.BCELoss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_action_accuracy': [],
            'val_return_mse': [],
            'attention_weights': [],  # Store attention weight snapshots
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting training...", epochs=epochs)
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch in train_loader:
                kama, context, base, pos_states, actions, returns = [
                    x.to(self.device) for x in batch
                ]
                
                has_position = pos_states[:, 0] > 0.5
                
                # Check for invalid actions
                invalid_mask = self._check_action_validity(actions, has_position)
                if invalid_mask.any():
                    valid_mask = ~invalid_mask
                    if not valid_mask.any():
                        continue
                    kama = kama[valid_mask]
                    context = context[valid_mask]
                    base = base[valid_mask]
                    pos_states = pos_states[valid_mask]
                    actions = actions[valid_mask]
                    returns = returns[valid_mask]
                    has_position = has_position[valid_mask]
                
                optimizer.zero_grad()
                
                outputs = model(kama, context, base, pos_states)
                
                # Apply action mask
                masked_logits = self._apply_training_mask(
                    outputs['action_logits'], has_position
                )
                
                # Losses
                action_loss = action_criterion(masked_logits, actions).mean()
                
                capped_returns = torch.clamp(returns, -1.0, 1.0)
                return_loss = return_criterion(
                    outputs['expected_return'].squeeze(-1),
                    capped_returns
                )
                
                with torch.no_grad():
                    pred_actions = masked_logits.argmax(dim=-1)
                    is_correct = (pred_actions == actions).float()
                
                confidence_loss = confidence_criterion(
                    outputs['confidence'].squeeze(-1),
                    is_correct
                )
                
                loss = (
                    self.config.policy_loss_weight * action_loss +
                    self.config.value_loss_weight * return_loss +
                    0.5 * confidence_loss
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            all_pred_actions = []
            all_true_actions = []
            all_pred_returns = []
            all_true_returns = []
            all_attention_weights = []
            all_time_remaining = []
            
            with torch.no_grad():
                for batch in val_loader:
                    kama, context, base, pos_states, actions, returns = [
                        x.to(self.device) for x in batch
                    ]
                    
                    has_position = pos_states[:, 0] > 0.5
                    
                    outputs = model(kama, context, base, pos_states)
                    
                    masked_logits = self._apply_training_mask(
                        outputs['action_logits'], has_position
                    )
                    
                    action_loss = action_criterion(masked_logits, actions).mean()
                    capped_returns = torch.clamp(returns, -1.0, 1.0)
                    return_loss = return_criterion(
                        outputs['expected_return'].squeeze(-1),
                        capped_returns
                    )
                    loss = action_loss + return_loss
                    
                    pred_actions = masked_logits.argmax(dim=-1)
                    
                    val_losses.append(loss.item())
                    all_pred_actions.extend(pred_actions.cpu().numpy())
                    all_true_actions.extend(actions.cpu().numpy())
                    all_pred_returns.extend(outputs['expected_return'].squeeze(-1).cpu().numpy())
                    all_true_returns.extend(returns.cpu().numpy())
                    all_attention_weights.extend(outputs['attention_weights'].cpu().numpy())
                    all_time_remaining.extend(context[:, 0].cpu().numpy())  # time_remaining_pct
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            action_accuracy = np.mean(np.array(all_pred_actions) == np.array(all_true_actions))
            capped_true = np.clip(np.array(all_true_returns), -1.0, 1.0)
            return_mse = np.mean((np.array(all_pred_returns) - capped_true) ** 2)
            
            scheduler.step(avg_val_loss)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_action_accuracy'].append(action_accuracy)
            history['val_return_mse'].append(return_mse)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_path / "best_model.pt")
            else:
                patience_counter += 1
            
            # Log attention weights periodically
            if (epoch + 1) % attention_log_interval == 0:
                self._log_attention_weights(
                    epoch + 1,
                    np.array(all_attention_weights),
                    np.array(all_time_remaining),
                    output_path,
                )
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}",
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{avg_val_loss:.4f}",
                    action_acc=f"{action_accuracy:.3f}",
                    return_mse=f"{return_mse:.4f}",
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))
        
        # Save final model and config
        torch.save(model.state_dict(), output_path / "final_model.pt")
        with open(output_path / "config.json", "w") as f:
            config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
            # Convert tuples to lists for JSON
            for key, value in config_dict.items():
                if isinstance(value, tuple):
                    config_dict[key] = list(value)
            json.dump(config_dict, f, indent=2)
        
        logger.info(
            "Training complete",
            best_val_loss=f"{best_val_loss:.4f}",
            final_action_accuracy=f"{history['val_action_accuracy'][-1]:.3f}",
        )
        
        return model, history
    
    def _log_attention_weights(
        self,
        epoch: int,
        attention_weights: np.ndarray,
        time_remaining: np.ndarray,
        output_path: Path,
    ):
        """Log attention weight analysis by time phase."""
        import csv
        
        # Define time phases
        phases = [
            ('early', 0.6, 1.0),
            ('mid', 0.3, 0.6),
            ('late', 0.0, 0.3),
        ]
        
        # KAMA periods for reference
        kama_periods = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80]
        
        analysis = {'epoch': epoch, 'phases': {}}
        
        logger.info(f"Epoch {epoch} - Attention Weight Analysis:")
        
        for phase_name, low, high in phases:
            mask = (time_remaining >= low) & (time_remaining < high)
            if mask.sum() == 0:
                continue
            
            phase_weights = attention_weights[mask].mean(axis=0)
            phase_std = attention_weights[mask].std(axis=0)
            
            analysis['phases'][phase_name] = {
                'mean_weights': phase_weights.tolist(),
                'std_weights': phase_std.tolist(),
                'sample_count': int(mask.sum()),
            }
            
            logger.info(f"  {phase_name.upper()} Phase ({int(low*100)}-{int(high*100)}% remaining):")
            for i, period in enumerate(kama_periods[:len(phase_weights)]):
                bar = '█' * int(phase_weights[i] * 20)
                logger.info(f"    KAMA_{period:2d}: {phase_weights[i]:.3f} {bar}")
        
        # Save to CSV
        csv_path = output_path / f"attention_weights_epoch_{epoch}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['phase', 'sample_count'] + [f'kama_{p}_mean' for p in kama_periods[:attention_weights.shape[1]]]
            writer.writerow(header)
            
            for phase_name, data in analysis['phases'].items():
                row = [phase_name, data['sample_count']] + data['mean_weights']
                writer.writerow(row)
        
        logger.info(f"  Saved attention weights to {csv_path}")


def load_attention_market_predictor(
    model_dir: str,
    device: Optional[str] = None,
) -> AttentionMarketPredictorModel:
    """Load trained attention market predictor model."""
    model_path = Path(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(model_path / "config.json", "r") as f:
        config_dict = json.load(f)
    
    # Convert lists back to tuples
    if 'hidden_dims' in config_dict and isinstance(config_dict['hidden_dims'], list):
        config_dict['hidden_dims'] = tuple(config_dict['hidden_dims'])
    
    config = AttentionMarketPredictorConfig(**config_dict)
    
    model = AttentionMarketPredictorModel(config)
    model.load_state_dict(
        torch.load(model_path / "final_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    
    return model
