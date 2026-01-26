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

    @classmethod
    def get_valid_actions(cls, has_position: bool) -> List[int]:
        """Return list of valid actions given position state."""
        if has_position:
            return [cls.EXIT, cls.HOLD]
        else:
            return [cls.WAIT, cls.BUY_YES, cls.BUY_NO]

    @classmethod
    def is_valid(cls, action: int, has_position: bool) -> bool:
        """Check if action is valid given position state."""
        return action in cls.get_valid_actions(has_position)


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

        # Action classification head (NOT Q-values - pure classification)
        self.action_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.config.num_actions),
        )

        # Confidence head - trained to predict action correctness
        # Used to gate trades and size positions
        self.confidence_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Expected return head - estimates tradable return (capped, normalized)
        # Used for position sizing, NOT as truth
        self.expected_return_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Returns capped to [-1, 1]
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
                - action_logits: Raw logits for each action (batch, num_actions)
                - confidence: Prediction confidence [0,1] (batch, 1)
                - expected_return: Capped expected return [-1,1] (batch, 1)
                - action_probs: Softmax action probabilities (batch, num_actions)
        """
        # Encode features
        h = self.encoder(x)

        # Apply temporal attention
        h = self.temporal_attention(h)

        # Action logits (pure classification, NOT Q-values)
        action_logits = self.action_head(h)

        # Action probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Auxiliary outputs
        confidence = self.confidence_head(h)
        expected_return = self.expected_return_head(h)

        return {
            'action_logits': action_logits,
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
        Get optimal action for given state WITH ACTION MASKING.

        Invalid actions are masked to -inf before selection.
        This is NON-NEGOTIABLE for correct behavior.

        Args:
            features: Market features (batch, feature_dim)
            position_state: Position state (batch, position_dim)
                           First element must be has_position flag (0 or 1)
            deterministic: If True, take argmax action
            epsilon: Exploration probability (only explores valid actions)

        Returns:
            Dictionary with action, action_logits, confidence, expected_return
        """
        self.eval()
        with torch.no_grad():
            x = torch.cat([features, position_state], dim=-1)
            outputs = self.forward(x)

            # Extract has_position flag (first element of position_state)
            has_position = position_state[:, 0] > 0.5  # Boolean tensor

            # CRITICAL: Apply action masking before any action selection
            action_logits = outputs['action_logits'].clone()
            masked_logits = self._apply_action_mask(action_logits, has_position)

            if deterministic:
                if np.random.random() < epsilon:
                    # Epsilon-greedy: sample uniformly from VALID actions only
                    action = self._sample_valid_random(has_position)
                else:
                    action = masked_logits.argmax(dim=-1)
            else:
                # Sample from masked probabilities
                masked_probs = F.softmax(masked_logits, dim=-1)
                action = torch.multinomial(masked_probs, 1).squeeze(-1)

            # Sanity check: verify action validity
            self._verify_action_validity(action, has_position)

        return {
            'action': action,
            'action_logits': outputs['action_logits'],  # Raw, unmasked for logging
            'masked_logits': masked_logits,  # After masking
            'confidence': outputs['confidence'],
            'expected_return': outputs['expected_return'],
            'action_probs': F.softmax(masked_logits, dim=-1),
        }

    def _apply_action_mask(
        self,
        logits: torch.Tensor,
        has_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply action mask: set invalid action logits to -inf.

        Args:
            logits: Action logits (batch, num_actions)
            has_position: Boolean tensor (batch,) indicating if position exists

        Returns:
            Masked logits with invalid actions set to -inf
        """
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

    def _verify_action_validity(self, action: torch.Tensor, has_position: torch.Tensor):
        """Verify all selected actions are valid. Raises error if not."""
        for i in range(len(action)):
            a = int(action[i])
            hp = bool(has_position[i])
            if not Action.is_valid(a, hp):
                raise ValueError(
                    f"INVALID ACTION BUG: action={Action.names()[a]}, "
                    f"has_position={hp}. This should never happen."
                )


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


class TradableActionLabeler:
    """
    Label historical data with TRADABLE actions and realistic expected returns.

    KEY CHANGES from hindsight labeler:
    1. Expected return uses market-implied probability, NOT outcome knowledge
    2. Returns are capped to realistic range [-1, 1]
    3. Actions are validated for position state
    4. Conservative: prefers WAIT over marginal trades
    """

    def __init__(
        self,
        transaction_cost: float = 0.02,  # 2% round-trip (conservative)
        min_edge_threshold: float = 0.05,  # Need 5% edge to trade
        return_cap: float = 0.5,  # Cap returns at 50%
    ):
        self.transaction_cost = transaction_cost
        self.min_edge_threshold = min_edge_threshold
        self.return_cap = return_cap

    def compute_tradable_action(
        self,
        yes_prices: np.ndarray,
        current_idx: int,
        has_position: bool,
        position_side: Optional[str],
        entry_price: float,
        outcome: int,  # Only used for labeling, NOT for expected return
    ) -> Tuple[int, float]:
        """
        Compute action using TRADABLE expected returns.

        The outcome is used to determine if the action WOULD HAVE worked,
        but expected_return is computed from market-implied probability only.

        Returns:
            (action, expected_return) where expected_return is in [-1, 1]
        """
        current_yes = yes_prices[current_idx]
        current_no = 1.0 - current_yes
        time_remaining = 1.0 - current_idx / len(yes_prices)

        # Market-implied probability (what we can actually estimate)
        implied_yes_prob = current_yes
        implied_no_prob = current_no

        if has_position:
            # MUST return EXIT or HOLD (action masking enforced)
            return self._compute_position_action(
                yes_prices, current_idx, position_side, entry_price,
                implied_yes_prob, outcome, time_remaining
            )
        else:
            # MUST return WAIT, BUY_YES, or BUY_NO (action masking enforced)
            return self._compute_entry_action(
                current_yes, current_no, implied_yes_prob, implied_no_prob,
                outcome, time_remaining
            )

    def _compute_position_action(
        self,
        yes_prices: np.ndarray,
        current_idx: int,
        position_side: str,
        entry_price: float,
        implied_yes_prob: float,
        outcome: int,
        time_remaining: float,
    ) -> Tuple[int, float]:
        """Compute EXIT vs HOLD for existing position."""
        if position_side == "yes":
            current_price = yes_prices[current_idx]
            implied_settlement = implied_yes_prob  # Market's estimate
            actual_settlement = float(outcome)
        else:
            current_price = 1.0 - yes_prices[current_idx]
            implied_settlement = 1.0 - implied_yes_prob
            actual_settlement = 1.0 - float(outcome)

        # Current unrealized PnL
        unrealized_pnl = (current_price - entry_price) / (entry_price + 1e-8)

        # Expected value of HOLD (using ACTUAL settlement for training target)
        actual_hold_return = (actual_settlement - entry_price) / (entry_price + 1e-8)
        # Note: No transaction cost deducted for holding (already paid/sunk)

        # Value of EXIT now
        exit_return = unrealized_pnl - self.transaction_cost

        # Decision: EXIT if current exit is better than expected hold (hindsight)
        # Also EXIT if time is very low and position is profitable
        should_exit = (
            exit_return > actual_hold_return or
            (time_remaining < 0.1 and exit_return > 0 and exit_return > actual_hold_return)
        )

        if should_exit:
            # Cap and normalize return
            capped_return = np.clip(exit_return, -self.return_cap, self.return_cap)
            return Action.EXIT, capped_return
        else:
            capped_return = np.clip(actual_hold_return, -self.return_cap, self.return_cap)
            return Action.HOLD, capped_return

    def _compute_entry_action(
        self,
        current_yes: float,
        current_no: float,
        implied_yes_prob: float,
        implied_no_prob: float,
        outcome: int,
        time_remaining: float,
    ) -> Tuple[int, float]:
        """Compute WAIT vs BUY_YES vs BUY_NO for no position."""

        # Compute expected returns using market-implied probabilities
        # This is what a trader can actually estimate

        # YES trade: buy at current_yes, expected settlement = implied_yes_prob
        yes_entry = current_yes + self.transaction_cost
        yes_expected_settle = implied_yes_prob
        yes_expected_return = (yes_expected_settle - yes_entry) / (yes_entry + 1e-8)

        # NO trade: buy at current_no, expected settlement = implied_no_prob
        no_entry = current_no + self.transaction_cost
        no_expected_settle = implied_no_prob
        no_expected_return = (no_expected_settle - no_entry) / (no_entry + 1e-8)

        # The ACTION label uses hindsight (what would have worked)
        # But RETURN is the tradable expectation
        actual_yes_return = (float(outcome) - yes_entry) / (yes_entry + 1e-8)
        actual_no_return = (1.0 - float(outcome) - no_entry) / (no_entry + 1e-8)

        # Determine best action based on actual outcome (for labeling)
        best_action = Action.WAIT
        best_return = 0.0

        # Only label as BUY if:
        # 1. It would have been profitable (hindsight)
        # 2. There was detectable edge (market mispricing)
        # 3. Enough time remaining

        yes_edge = actual_yes_return - yes_expected_return  # How wrong was market?
        no_edge = actual_no_return - no_expected_return

        if time_remaining > 0.1:  # Don't enter late
            if actual_yes_return > self.min_edge_threshold and yes_edge > 0:
                # YES would have worked and market underpriced it
                best_action = Action.BUY_YES
                # Return is ACTUAL realized return for training target
                best_return = np.clip(actual_yes_return, -self.return_cap, self.return_cap)

            elif actual_no_return > self.min_edge_threshold and no_edge > 0:
                # NO would have worked and market underpriced it
                best_action = Action.BUY_NO
                best_return = np.clip(actual_no_return, -self.return_cap, self.return_cap)

        return best_action, best_return


# Keep old name as alias for backward compatibility
OptimalActionLabeler = TradableActionLabeler


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

    def _check_action_validity(
        self,
        actions: torch.Tensor,
        has_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check which actions in batch are invalid given position state.
        Returns boolean mask where True = invalid.
        """
        invalid = torch.zeros(len(actions), dtype=torch.bool, device=actions.device)

        for i in range(len(actions)):
            a = int(actions[i])
            hp = bool(has_position[i])
            if not Action.is_valid(a, hp):
                invalid[i] = True

        return invalid

    def _apply_training_mask(
        self,
        logits: torch.Tensor,
        has_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply action mask during training.
        Invalid actions get -inf so they contribute 0 to softmax.
        """
        masked = logits.clone()

        for i in range(len(logits)):
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
        action_criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample for masking
        return_criterion = nn.SmoothL1Loss()
        confidence_criterion = nn.BCELoss()

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_action_accuracy': [],
            'val_return_mse': [],
            'val_confidence_auc': [],
            'invalid_action_count': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting training...", epochs=epochs)

        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            epoch_invalid_actions = 0

            for features, pos_states, actions, returns in train_loader:
                features = features.to(self.device)
                pos_states = pos_states.to(self.device)
                actions = actions.to(self.device)
                returns = returns.to(self.device)

                # Extract has_position flag for masking
                has_position = pos_states[:, 0] > 0.5

                # CRITICAL: Verify training labels are valid actions
                invalid_mask = self._check_action_validity(actions, has_position)
                if invalid_mask.any():
                    num_invalid = invalid_mask.sum().item()
                    epoch_invalid_actions += num_invalid
                    # Skip invalid samples - they indicate data generation bug
                    valid_mask = ~invalid_mask
                    if not valid_mask.any():
                        continue
                    features = features[valid_mask]
                    pos_states = pos_states[valid_mask]
                    actions = actions[valid_mask]
                    returns = returns[valid_mask]
                    has_position = has_position[valid_mask]

                optimizer.zero_grad()

                x = torch.cat([features, pos_states], dim=-1)
                outputs = model(x)

                # Apply action mask to logits for loss computation
                masked_logits = self._apply_training_mask(
                    outputs['action_logits'], has_position
                )

                # Action loss (cross-entropy on masked logits)
                action_loss = action_criterion(masked_logits, actions).mean()

                # Return prediction loss (on normalized returns)
                # Cap returns to [-1, 1] to match Tanh output
                capped_returns = torch.clamp(returns, -1.0, 1.0)
                return_loss = return_criterion(
                    outputs['expected_return'].squeeze(-1),
                    capped_returns
                )

                # Confidence loss: predict whether action prediction is correct
                with torch.no_grad():
                    pred_actions = masked_logits.argmax(dim=-1)
                    is_correct = (pred_actions == actions).float()

                confidence_loss = confidence_criterion(
                    outputs['confidence'].squeeze(-1),
                    is_correct
                )

                # Total loss (no entropy bonus - we want confident predictions)
                loss = (
                    self.config.policy_loss_weight * action_loss +
                    self.config.value_loss_weight * return_loss +
                    0.5 * confidence_loss  # Train confidence to gate trades
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Log invalid action count
            if epoch_invalid_actions > 0:
                logger.warning(
                    f"Epoch {epoch+1}: {epoch_invalid_actions} invalid action labels in training data!"
                )
            history['invalid_action_count'].append(epoch_invalid_actions)

            # Validation
            model.eval()
            val_losses = []
            all_pred_actions = []
            all_true_actions = []
            all_pred_returns = []
            all_true_returns = []
            all_confidences = []
            all_correct = []

            with torch.no_grad():
                for features, pos_states, actions, returns in val_loader:
                    features = features.to(self.device)
                    pos_states = pos_states.to(self.device)
                    actions = actions.to(self.device)
                    returns = returns.to(self.device)

                    has_position = pos_states[:, 0] > 0.5

                    x = torch.cat([features, pos_states], dim=-1)
                    outputs = model(x)

                    # Apply mask for validation too
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
                    is_correct = (pred_actions == actions).float()

                    val_losses.append(loss.item())
                    all_pred_actions.extend(pred_actions.cpu().numpy())
                    all_true_actions.extend(actions.cpu().numpy())
                    all_pred_returns.extend(outputs['expected_return'].squeeze(-1).cpu().numpy())
                    all_true_returns.extend(returns.cpu().numpy())
                    all_confidences.extend(outputs['confidence'].squeeze(-1).cpu().numpy())
                    all_correct.extend(is_correct.cpu().numpy())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            action_accuracy = np.mean(np.array(all_pred_actions) == np.array(all_true_actions))
            capped_true = np.clip(np.array(all_true_returns), -1.0, 1.0)
            return_mse = np.mean((np.array(all_pred_returns) - capped_true) ** 2)

            # Confidence calibration: correlation between confidence and correctness
            if len(all_confidences) > 0:
                conf_arr = np.array(all_confidences)
                correct_arr = np.array(all_correct)
                # Simple AUC proxy: mean confidence when correct vs incorrect
                correct_conf = conf_arr[correct_arr > 0.5].mean() if (correct_arr > 0.5).any() else 0
                incorrect_conf = conf_arr[correct_arr <= 0.5].mean() if (correct_arr <= 0.5).any() else 0
                conf_separation = correct_conf - incorrect_conf  # Should be positive
            else:
                conf_separation = 0.0

            scheduler.step(avg_val_loss)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_action_accuracy'].append(action_accuracy)
            history['val_return_mse'].append(return_mse)
            history['val_confidence_auc'].append(conf_separation)

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
                    conf_sep=f"{conf_separation:.3f}",
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
