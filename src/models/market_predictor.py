"""
Unified Market Predictor for Polymarket Token Trading.

This model replaces the separate EdgeDetector + SAC architecture with a single
end-to-end trained model that learns to predict OPTIMAL ACTIONS, not just
P(YES wins).

Key Design Principles:
1. Predict EXPECTED RETURN, not binary outcome
2. Time-aware: naturally penalize late entries without hardcoding
3. Trend-aware: learn from price momentum patterns
4. Convergence-aware: understand when prices approach certainty
5. No hardcoded thresholds: model learns optimal entry/exit points

Training Target:
- Given state S at time T, what is the EXPECTED VALUE of each action?
- Actions: WAIT, BUY_YES, BUY_NO, EXIT_YES, EXIT_NO
- Value = realized PnL if action taken and position held to optimal exit

Architecture:
- Input: Enhanced features (trend, time, convergence, position state)
- Output: Q-values for each action
- Training: On REAL historical data with computed optimal actions
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


# Action space
class Action:
    WAIT = 0
    BUY_YES = 1
    BUY_NO = 2
    EXIT = 3  # Exit current position
    HOLD = 4  # Continue holding

    @classmethod
    def num_actions(cls) -> int:
        return 5

    @classmethod
    def names(cls) -> List[str]:
        return ["WAIT", "BUY_YES", "BUY_NO", "EXIT", "HOLD"]


@dataclass
class MarketPredictorConfig:
    """Configuration for unified market predictor."""

    # Feature dimensions
    base_feature_dim: int = 51  # From TokenFeatureBuilder
    position_state_dim: int = 6  # Enhanced position state
    context_dim: int = 16  # Additional learned context

    # Architecture
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    attention_heads: int = 4
    dropout: float = 0.2
    use_layer_norm: bool = True

    # Output
    num_actions: int = 5  # WAIT, BUY_YES, BUY_NO, EXIT, HOLD

    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99  # Discount for future rewards

    # Loss weighting
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    entropy_weight: float = 0.01  # Encourage exploration


class TemporalAttention(nn.Module):
    """Self-attention for temporal patterns in price history."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim) or (batch, embed_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out).squeeze(1)


class MarketPredictorModel(nn.Module):
    """
    Unified predictor for optimal trading actions.

    Outputs:
    - Q-values for each action
    - Confidence score
    - Predicted expected return for current position
    """

    def __init__(self, config: Optional[MarketPredictorConfig] = None):
        super().__init__()
        self.config = config or MarketPredictorConfig()

        input_dim = self.config.base_feature_dim + self.config.position_state_dim

        # Feature encoder with layer norm
        encoder_layers = []
        in_dim = input_dim

        for hidden_dim in self.config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if self.config.use_layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        final_dim = self.config.hidden_dims[-1]

        # Temporal attention for pattern recognition
        self.temporal_attention = TemporalAttention(
            embed_dim=final_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout,
        )

        # Action Q-value heads (dueling architecture)
        self.value_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.config.num_actions),
        )

        # Auxiliary heads
        self.confidence_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.expected_return_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Returns between -1 and 1
        )

        self._init_weights()

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MarketPredictorModel initialized",
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            num_params=num_params,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Dictionary with:
                - q_values: Q-value for each action (batch, num_actions)
                - value: State value (batch, 1)
                - advantage: Advantage per action (batch, num_actions)
                - confidence: Prediction confidence (batch, 1)
                - expected_return: Expected return for current state (batch, 1)
                - action_probs: Softmax action probabilities (batch, num_actions)
        """
        # Encode features
        h = self.encoder(x)

        # Apply temporal attention
        h = self.temporal_attention(h)

        # Dueling Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(h)
        advantage = self.advantage_head(h)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # Action probabilities (for policy gradient)
        action_probs = F.softmax(q_values / 0.1, dim=-1)  # Temperature-scaled

        # Auxiliary outputs
        confidence = self.confidence_head(h)
        expected_return = self.expected_return_head(h)

        return {
            'q_values': q_values,
            'value': value,
            'advantage': advantage,
            'confidence': confidence,
            'expected_return': expected_return,
            'action_probs': action_probs,
        }

    def get_action(
        self,
        features: torch.Tensor,
        position_state: torch.Tensor,
        deterministic: bool = True,
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get optimal action for given state.

        Args:
            features: Market features (batch, feature_dim)
            position_state: Position state (batch, position_dim)
            deterministic: If True, take argmax action
            epsilon: Exploration probability

        Returns:
            Dictionary with action, q_values, confidence
        """
        self.eval()
        with torch.no_grad():
            x = torch.cat([features, position_state], dim=-1)
            outputs = self.forward(x)

            if deterministic:
                if np.random.random() < epsilon:
                    action = torch.randint(0, self.config.num_actions, (x.shape[0],))
                else:
                    action = outputs['q_values'].argmax(dim=-1)
            else:
                # Sample from action probabilities
                action = torch.multinomial(outputs['action_probs'], 1).squeeze(-1)

        return {
            'action': action,
            'q_values': outputs['q_values'],
            'confidence': outputs['confidence'],
            'expected_return': outputs['expected_return'],
            'action_probs': outputs['action_probs'],
        }


class EnhancedPositionState:
    """Build enhanced position state vector."""

    @staticmethod
    def compute(
        has_position: bool,
        position_side: Optional[str],  # "yes" or "no"
        entry_price: float,
        current_price: float,
        time_remaining: float,
        ticks_held: int,
        max_pnl_seen: float,
    ) -> np.ndarray:
        """
        Compute 6-dimensional position state.

        Features:
        1. has_position (0 or 1)
        2. position_direction (-1 for NO, 0 for none, +1 for YES)
        3. unrealized_pnl (normalized)
        4. holding_time_fraction (ticks_held normalized by typical hold time)
        5. drawdown_from_max (current vs best seen)
        6. time_pressure (exponential as time runs out)
        """
        if has_position and entry_price > 0:
            position_dir = 1.0 if position_side == "yes" else -1.0
            unrealized_pnl = (current_price - entry_price) / (entry_price + 1e-8)
            holding_time = min(ticks_held / 50.0, 1.0)  # Normalize to typical range
            drawdown = (max_pnl_seen - unrealized_pnl) if max_pnl_seen > unrealized_pnl else 0.0
        else:
            position_dir = 0.0
            unrealized_pnl = 0.0
            holding_time = 0.0
            drawdown = 0.0

        # Time pressure: exponential increase as settlement approaches
        time_pressure = np.exp(3.0 * (1.0 - time_remaining)) - 1.0
        time_pressure = min(time_pressure, 10.0) / 10.0  # Normalize

        return np.array([
            float(has_position),
            position_dir,
            np.clip(unrealized_pnl, -1.0, 1.0),
            holding_time,
            np.clip(drawdown, 0.0, 1.0),
            time_pressure,
        ], dtype=np.float32)


class OptimalActionLabeler:
    """
    Label historical data with optimal actions based on hindsight.

    For each state, compute what the BEST action would have been
    given knowledge of future prices.
    """

    def __init__(
        self,
        transaction_cost: float = 0.01,
        min_profit_threshold: float = 0.02,
    ):
        self.transaction_cost = transaction_cost
        self.min_profit_threshold = min_profit_threshold

    def compute_optimal_action(
        self,
        yes_prices: np.ndarray,
        current_idx: int,
        has_position: bool,
        position_side: Optional[str],
        entry_price: float,
        outcome: int,  # 0 = NO wins, 1 = YES wins
    ) -> Tuple[int, float]:
        """
        Compute optimal action at current_idx given future knowledge.

        Returns:
            (action, expected_value)
        """
        current_yes = yes_prices[current_idx]
        current_no = 1.0 - current_yes
        time_remaining = 1.0 - current_idx / len(yes_prices)

        # Settlement prices
        settlement_yes = float(outcome)
        settlement_no = 1.0 - settlement_yes

        if has_position:
            # Currently holding - evaluate EXIT vs HOLD
            if position_side == "yes":
                current_price = current_yes
                final_value = settlement_yes
            else:
                current_price = current_no
                final_value = settlement_no

            # Value of holding to settlement
            hold_value = (final_value - entry_price) / (entry_price + 1e-8)
            hold_value -= self.transaction_cost  # Exit cost at settlement

            # Value of exiting now
            exit_value = (current_price - entry_price) / (entry_price + 1e-8)
            exit_value -= self.transaction_cost

            # Check if there's a better exit point ahead
            best_exit_value = exit_value
            for future_idx in range(current_idx + 1, len(yes_prices)):
                future_price = yes_prices[future_idx] if position_side == "yes" else (1.0 - yes_prices[future_idx])
                future_exit = (future_price - entry_price) / (entry_price + 1e-8) - self.transaction_cost
                best_exit_value = max(best_exit_value, future_exit)

            if hold_value > exit_value and hold_value > best_exit_value - 0.01:
                return Action.HOLD, hold_value
            else:
                return Action.EXIT, max(exit_value, best_exit_value)

        else:
            # No position - evaluate WAIT vs BUY_YES vs BUY_NO
            wait_value = 0.0  # Opportunity cost

            # Value of buying YES now
            yes_entry_cost = current_yes + self.transaction_cost
            yes_final_value = settlement_yes - self.transaction_cost
            yes_pnl = (yes_final_value - yes_entry_cost) / (yes_entry_cost + 1e-8)

            # Check for better entry
            best_yes_pnl = yes_pnl
            for future_idx in range(current_idx + 1, len(yes_prices)):
                future_entry = yes_prices[future_idx] + self.transaction_cost
                future_pnl = (settlement_yes - self.transaction_cost - future_entry) / (future_entry + 1e-8)
                best_yes_pnl = max(best_yes_pnl, future_pnl)

            # Value of buying NO now
            no_entry_cost = current_no + self.transaction_cost
            no_final_value = settlement_no - self.transaction_cost
            no_pnl = (no_final_value - no_entry_cost) / (no_entry_cost + 1e-8)

            best_no_pnl = no_pnl
            for future_idx in range(current_idx + 1, len(yes_prices)):
                future_entry = (1.0 - yes_prices[future_idx]) + self.transaction_cost
                future_pnl = (settlement_no - self.transaction_cost - future_entry) / (future_entry + 1e-8)
                best_no_pnl = max(best_no_pnl, future_pnl)

            # Choose best action
            # If waiting leads to better entry, prefer waiting
            if best_yes_pnl > yes_pnl + self.min_profit_threshold:
                # Better YES entry available later
                if best_no_pnl > no_pnl + self.min_profit_threshold:
                    # Both could be better later, wait
                    return Action.WAIT, 0.0
                elif no_pnl > self.min_profit_threshold:
                    return Action.BUY_NO, no_pnl
                else:
                    return Action.WAIT, 0.0

            if best_no_pnl > no_pnl + self.min_profit_threshold:
                if yes_pnl > self.min_profit_threshold:
                    return Action.BUY_YES, yes_pnl
                else:
                    return Action.WAIT, 0.0

            # Current entry is optimal or close to it
            if yes_pnl > no_pnl and yes_pnl > self.min_profit_threshold:
                return Action.BUY_YES, yes_pnl
            elif no_pnl > yes_pnl and no_pnl > self.min_profit_threshold:
                return Action.BUY_NO, no_pnl
            else:
                return Action.WAIT, wait_value


class MarketPredictorDataset(Dataset):
    """Dataset for training market predictor."""

    def __init__(
        self,
        features: np.ndarray,
        position_states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
    ):
        self.features = torch.FloatTensor(features)
        self.position_states = torch.FloatTensor(position_states)
        self.actions = torch.LongTensor(actions)
        self.returns = torch.FloatTensor(returns)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.features[idx],
            self.position_states[idx],
            self.actions[idx],
            self.returns[idx],
        )


class MarketPredictorTrainer:
    """Trainer for unified market predictor."""

    def __init__(
        self,
        config: Optional[MarketPredictorConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or MarketPredictorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("MarketPredictorTrainer initialized", device=self.device)

    def train(
        self,
        train_features: np.ndarray,
        train_position_states: np.ndarray,
        train_actions: np.ndarray,
        train_returns: np.ndarray,
        val_features: np.ndarray,
        val_position_states: np.ndarray,
        val_actions: np.ndarray,
        val_returns: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 15,
        output_dir: str = "./logs/market_predictor",
    ) -> Tuple[MarketPredictorModel, Dict[str, Any]]:
        """Train the market predictor model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create datasets
        train_dataset = MarketPredictorDataset(
            train_features, train_position_states, train_actions, train_returns
        )
        val_dataset = MarketPredictorDataset(
            val_features, val_position_states, val_actions, val_returns
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        input_dim = train_features.shape[1] + train_position_states.shape[1]
        self.config.base_feature_dim = train_features.shape[1]
        self.config.position_state_dim = train_position_states.shape[1]

        model = MarketPredictorModel(self.config).to(self.device)

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
        action_criterion = nn.CrossEntropyLoss()
        return_criterion = nn.SmoothL1Loss()

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_action_accuracy': [],
            'val_return_mse': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting training...", epochs=epochs)

        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []

            for features, pos_states, actions, returns in train_loader:
                features = features.to(self.device)
                pos_states = pos_states.to(self.device)
                actions = actions.to(self.device)
                returns = returns.to(self.device)

                optimizer.zero_grad()

                x = torch.cat([features, pos_states], dim=-1)
                outputs = model(x)

                # Action loss (cross-entropy)
                action_loss = action_criterion(outputs['q_values'], actions)

                # Return prediction loss
                return_loss = return_criterion(
                    outputs['expected_return'].squeeze(-1),
                    returns
                )

                # Entropy bonus for exploration
                probs = outputs['action_probs']
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

                # Total loss
                loss = (
                    self.config.policy_loss_weight * action_loss +
                    self.config.value_loss_weight * return_loss -
                    self.config.entropy_weight * entropy
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

            with torch.no_grad():
                for features, pos_states, actions, returns in val_loader:
                    features = features.to(self.device)
                    pos_states = pos_states.to(self.device)
                    actions = actions.to(self.device)
                    returns = returns.to(self.device)

                    x = torch.cat([features, pos_states], dim=-1)
                    outputs = model(x)

                    action_loss = action_criterion(outputs['q_values'], actions)
                    return_loss = return_criterion(
                        outputs['expected_return'].squeeze(-1),
                        returns
                    )
                    loss = action_loss + return_loss

                    val_losses.append(loss.item())
                    all_pred_actions.extend(outputs['q_values'].argmax(dim=-1).cpu().numpy())
                    all_true_actions.extend(actions.cpu().numpy())
                    all_pred_returns.extend(outputs['expected_return'].squeeze(-1).cpu().numpy())
                    all_true_returns.extend(returns.cpu().numpy())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            action_accuracy = np.mean(np.array(all_pred_actions) == np.array(all_true_actions))
            return_mse = np.mean((np.array(all_pred_returns) - np.array(all_true_returns)) ** 2)

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
            json.dump({
                "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')},
                "feature_dim": int(train_features.shape[1]),
                "position_state_dim": int(train_position_states.shape[1]),
            }, f, indent=2, default=str)

        logger.info(
            "Training complete",
            best_val_loss=f"{best_val_loss:.4f}",
            final_action_accuracy=f"{history['val_action_accuracy'][-1]:.3f}",
        )

        return model, history


def load_market_predictor(
    model_dir: str,
    device: Optional[str] = None,
) -> MarketPredictorModel:
    """Load trained market predictor model."""
    model_path = Path(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_path / "config.json", "r") as f:
        config_dict = json.load(f)

    config = MarketPredictorConfig(
        base_feature_dim=config_dict["feature_dim"],
        position_state_dim=config_dict["position_state_dim"],
    )

    model = MarketPredictorModel(config)
    model.load_state_dict(
        torch.load(model_path / "final_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model
