#!/usr/bin/env python3
"""
Safeguarded Paper Trading for Polymarket.

This is a corrected execution layer that:
1. Uses edge-based entry decisions (not action classification)
2. Implements mandatory capital protection rules
3. Removes model control of EXIT decisions
4. Treats the model as a probability estimator

Key safeguards:
- Fixed per-trade stop loss (5%)
- Drawdown-scaled position sizing
- Loss-streak circuit breaker (5 consecutive losses)
- Minimum time remaining filter (35%)
- Rule-based exits only (stop loss, time expiry, settlement)

The model's job is ONLY to output P(YES | features).
Edge is computed as: edge = P(YES) - market_price_yes
"""

import os
os.environ["PYTORCH_NNPACK_ENABLED"] = "0"

import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")

from contextlib import contextmanager
import sys

@contextmanager
def suppress_stderr():
    """Suppress stderr at the file descriptor level."""
    try:
        original_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, original_stderr_fd)
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
        except Exception:
            pass

with suppress_stderr():
    import torch

import asyncio
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import httpx
import numpy as np
import structlog
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

from src.data.enhanced_features import EnhancedFeatureBuilder

logger = structlog.get_logger(__name__)
console = Console()

# SOCKS5 proxy for Polymarket
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False
    httpx_socks = None

SOCKS5_PROXY_URL = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:1080")


# =============================================================================
# EXECUTION SAFEGUARDS - HARD RULES (NON-NEGOTIABLE)
# =============================================================================

@dataclass
class SafeguardConfig:
    """Capital protection rules. These are HARD LIMITS."""

    # Edge thresholds
    edge_threshold: float = 0.07  # 7% minimum edge to enter (conservative start)
    edge_threshold_relaxed: float = 0.05  # Can relax to 5% after validation

    # Stop loss
    stop_loss_pct: float = 0.05  # 5% max loss per trade

    # Time filter
    min_time_remaining: float = 0.35  # Don't enter after 65% of candle elapsed

    # Position sizing
    base_position_size: float = 0.08  # 8% base position
    max_position_size: float = 0.15  # 15% max (reduced from 25%)
    min_position_size: float = 0.02  # 2% floor
    max_edge_scaling: float = 1.25  # Cap edge-based scaling at 1.25x

    # Drawdown scaling thresholds
    drawdown_tier1: float = 0.10  # >10% DD: scale to 50%
    drawdown_tier2: float = 0.20  # >20% DD: scale to 25%
    drawdown_tier3: float = 0.30  # >30% DD: stop trading

    # Loss streak circuit breaker
    max_consecutive_losses: int = 5
    cooldown_ticks: int = 60  # ~5 minutes at 5s refresh

    # Daily limits
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 0.15  # Stop if down 15% in a day


@dataclass
class SafeguardedTradingState:
    """Trading state with safeguard tracking."""

    # Account
    balance: float = 1000.0
    initial_balance: float = 1000.0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0

    # Peak tracking for drawdown
    peak_balance: float = 1000.0
    current_drawdown: float = 0.0

    # Loss streak tracking
    consecutive_losses: int = 0
    trading_paused: bool = False
    cooldown_remaining: int = 0

    # Daily tracking
    daily_trades: int = 0
    daily_pnl: float = 0.0
    trading_day: Optional[str] = None

    # Current Market Info
    current_candle_ts: Optional[int] = None
    active_yes_id: Optional[str] = None
    active_no_id: Optional[str] = None

    # Current position
    position_side: Optional[str] = None  # "yes" or "no"
    position_size: float = 0.0
    position_dollar_size: float = 0.0
    entry_price: float = 0.0
    entry_tick: int = 0
    ticks_held: int = 0

    # Market state
    last_yes_price: float = 0.5
    last_no_price: float = 0.5
    btc_open_price: Optional[float] = None
    btc_current_price: Optional[float] = None

    # Price history for features
    yes_price_history: List[float] = field(default_factory=list)
    no_price_history: List[float] = field(default_factory=list)
    btc_price_history: List[float] = field(default_factory=list)

    # Model outputs (for logging/display)
    last_p_yes: float = 0.5
    last_edge_yes: float = 0.0
    last_edge_no: float = 0.0

    # Trade history
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def update_drawdown(self):
        """Update peak balance and current drawdown."""
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.current_drawdown = (self.peak_balance - self.balance) / self.peak_balance

    def check_daily_reset(self):
        """Reset daily counters if new day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.trading_day != today:
            self.trading_day = today
            self.daily_trades = 0
            self.daily_pnl = 0.0


@dataclass
class SafeguardedPaperTradeConfig:
    """Configuration for safeguarded paper trading."""

    model_path: str = "./logs/market_predictor_v1"
    model_type: str = "legacy"  # "legacy" (old action model) or "probability" (new)

    initial_balance: float = 1000.0

    # Safeguard config
    safeguards: SafeguardConfig = field(default_factory=SafeguardConfig)

    # Polymarket API
    polymarket_api: str = "https://clob.polymarket.com"
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"

    # Refresh interval
    refresh_seconds: int = 5

    # Logging
    log_dir: str = "./logs/paper_trade_safeguarded"
    enable_ml_logging: bool = True


class SafeguardedPaperTrader:
    """
    Paper trader with mandatory capital protection.

    This trader:
    1. Uses model output as P(YES) estimate
    2. Computes edge = P(YES) - market_price
    3. Enters only when edge > threshold
    4. Exits only via rules (stop loss, time, settlement)
    5. Scales position size based on drawdown and edge
    """

    def __init__(self, config: SafeguardedPaperTradeConfig):
        self.config = config
        self.safeguards = config.safeguards
        self.state = SafeguardedTradingState(
            balance=config.initial_balance,
            initial_balance=config.initial_balance,
            peak_balance=config.initial_balance,
        )
        self.feature_builder = EnhancedFeatureBuilder()

        # Model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # HTTP clients
        self.polymarket_client: Optional[httpx.AsyncClient] = None
        self.binance_client: Optional[httpx.AsyncClient] = None

        # Tracking
        self._running = False
        self._tick_count = 0

        # Logging
        self.ml_log_file: Optional[Path] = None
        self.ml_log_handle = None

    # =========================================================================
    # SAFEGUARD METHODS
    # =========================================================================

    def calculate_position_size(self, edge: float) -> float:
        """
        Calculate position size with drawdown scaling and edge adjustment.

        Position size = base_size * drawdown_factor * edge_factor
        """
        dd = self.state.current_drawdown

        # Drawdown scaling
        if dd >= self.safeguards.drawdown_tier3:
            return 0.0  # Stop trading
        elif dd >= self.safeguards.drawdown_tier2:
            dd_factor = 0.25
        elif dd >= self.safeguards.drawdown_tier1:
            dd_factor = 0.50
        else:
            dd_factor = 1.0

        # Edge scaling (capped at max_edge_scaling)
        # At threshold edge (0.07), factor = 1.0
        # At 2x threshold (0.14), factor = 1.25 (capped)
        edge_ratio = abs(edge) / self.safeguards.edge_threshold
        edge_factor = min(edge_ratio, self.safeguards.max_edge_scaling)

        size = self.safeguards.base_position_size * dd_factor * edge_factor

        return float(np.clip(
            size,
            self.safeguards.min_position_size,
            self.safeguards.max_position_size
        ))

    def check_entry_allowed(self, time_remaining: float) -> Tuple[bool, str]:
        """
        Check if entry is allowed based on safeguards.

        Returns (allowed, reason) tuple.
        """
        # Circuit breaker
        if self.state.trading_paused:
            return False, "circuit_breaker"

        # Drawdown limit
        if self.state.current_drawdown >= self.safeguards.drawdown_tier3:
            return False, "max_drawdown"

        # Daily trade limit
        if self.state.daily_trades >= self.safeguards.max_daily_trades:
            return False, "daily_trade_limit"

        # Daily loss limit
        daily_loss_pct = -self.state.daily_pnl / self.state.initial_balance
        if daily_loss_pct >= self.safeguards.max_daily_loss_pct:
            return False, "daily_loss_limit"

        # Time remaining
        if time_remaining < self.safeguards.min_time_remaining:
            return False, "insufficient_time"

        # Already have position
        if self.state.position_side is not None:
            return False, "has_position"

        return True, "ok"

    def check_stop_loss(self) -> bool:
        """Check if stop loss is triggered. Returns True if should exit."""
        if self.state.position_side is None:
            return False

        if self.state.position_side == "yes":
            current_price = self.state.last_yes_price
        else:
            current_price = self.state.last_no_price

        unrealized_pnl_pct = (current_price - self.state.entry_price) / (self.state.entry_price + 1e-8)

        return unrealized_pnl_pct <= -self.safeguards.stop_loss_pct

    def update_circuit_breaker(self, trade_won: bool):
        """Update loss streak and circuit breaker state."""
        if trade_won:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

            if self.state.consecutive_losses >= self.safeguards.max_consecutive_losses:
                self.state.trading_paused = True
                self.state.cooldown_remaining = self.safeguards.cooldown_ticks
                console.print(
                    f"[bold red]CIRCUIT BREAKER: {self.state.consecutive_losses} "
                    f"consecutive losses. Pausing for {self.safeguards.cooldown_ticks} ticks.[/bold red]"
                )

    def tick_cooldown(self):
        """Decrement cooldown if paused."""
        if self.state.trading_paused and self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            if self.state.cooldown_remaining <= 0:
                self.state.trading_paused = False
                self.state.consecutive_losses = 0
                console.print("[green]Circuit breaker reset. Trading resumed.[/green]")

    # =========================================================================
    # MODEL INTERFACE
    # =========================================================================

    def load_model(self):
        """Load the model based on type."""
        console.print("[bold blue]Loading model...[/bold blue]")

        model_path = Path(self.config.model_path)

        if self.config.model_type == "probability":
            # New probability model
            from src.models.probability_predictor import load_probability_predictor
            self.model = load_probability_predictor(str(model_path), self.device)
            console.print(f"  [green]Probability model loaded from {model_path}[/green]")
        else:
            # Legacy action model - we'll extract probability from it
            from src.models.market_predictor import load_market_predictor
            self.model = load_market_predictor(str(model_path), self.device)
            console.print(f"  [green]Legacy model loaded from {model_path}[/green]")
            console.print("  [yellow]Note: Extracting P(YES) from action logits[/yellow]")

        self.model.eval()

    def get_probability_estimate(self) -> float:
        """
        Get P(YES) estimate from model.

        For legacy models: extract from action logits
        For probability models: direct output
        """
        if self.model is None:
            return 0.5

        # Build features
        time_remaining = self.get_time_remaining()

        yes_prices = np.array(self.state.yes_price_history[-300:] or [0.5])
        no_prices = np.array(self.state.no_price_history[-300:] or [0.5])
        btc_prices = np.array(self.state.btc_price_history[-300:]) if self.state.btc_price_history else None

        features = self.feature_builder.compute_features(
            yes_prices=yes_prices,
            no_prices=no_prices,
            time_remaining=time_remaining,
            btc_prices=btc_prices,
            btc_open=self.state.btc_open_price,
        )

        features_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.config.model_type == "probability":
                # Direct probability output
                p_yes = self.model.predict_probability(features_t)
                return float(p_yes.item())
            else:
                # Legacy model: extract from logits
                # We'll use the BUY_YES vs BUY_NO logit difference as a proxy
                # This is imperfect but better than using confidence/expected_return

                # Build dummy position state (no position)
                position_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                position_t = torch.FloatTensor(position_state).unsqueeze(0).to(self.device)

                x = torch.cat([features_t, position_t], dim=-1)
                outputs = self.model(x)

                # Get action logits (WAIT=0, BUY_YES=1, BUY_NO=2, EXIT=3, HOLD=4)
                logits = outputs['action_logits'].squeeze(0)

                # Use softmax over BUY_YES and BUY_NO only as proxy for P(YES)
                yes_logit = logits[1].item()  # BUY_YES
                no_logit = logits[2].item()   # BUY_NO

                # Softmax to get relative preference
                exp_yes = np.exp(yes_logit - max(yes_logit, no_logit))
                exp_no = np.exp(no_logit - max(yes_logit, no_logit))
                p_yes = exp_yes / (exp_yes + exp_no)

                return float(p_yes)

    def compute_edges(self, p_yes: float) -> Tuple[float, float]:
        """
        Compute edge for YES and NO sides explicitly.

        edge_yes = P(YES) - price_yes
        edge_no = P(NO) - price_no = (1 - P(YES)) - price_no

        Note: We compute edge_no explicitly from price_no, not as -edge_yes,
        to handle any market spread correctly.
        """
        price_yes = self.state.last_yes_price
        price_no = self.state.last_no_price

        edge_yes = p_yes - price_yes
        edge_no = (1.0 - p_yes) - price_no

        return edge_yes, edge_no

    # =========================================================================
    # TRADING LOGIC
    # =========================================================================

    def execute_trading_logic(self):
        """
        Main trading logic with safeguards.

        Entry: Based on edge threshold
        Exit: Based on rules only (stop loss, time, settlement)
        """
        time_remaining = self.get_time_remaining()

        # Update state tracking
        self.state.check_daily_reset()
        self.state.update_drawdown()
        self.tick_cooldown()

        # Get probability estimate
        p_yes = self.get_probability_estimate()
        self.state.last_p_yes = p_yes

        # Compute edges explicitly
        edge_yes, edge_no = self.compute_edges(p_yes)
        self.state.last_edge_yes = edge_yes
        self.state.last_edge_no = edge_no

        # Update position tracking
        if self.state.position_side:
            self.state.ticks_held += 1

            # CHECK STOP LOSS (Rule-based exit #1)
            if self.check_stop_loss():
                self.execute_exit("stop_loss")
                return

            # CHECK TIME EXPIRY (Rule-based exit #2)
            if time_remaining < 0.02:
                self.execute_exit("time_expiry")
                return

            # No model-controlled exits - position held until rule triggers
            return

        # NO POSITION - Check for entry
        allowed, reason = self.check_entry_allowed(time_remaining)

        if not allowed:
            # Log filtered signal if edge was sufficient
            if edge_yes >= self.safeguards.edge_threshold or edge_no >= self.safeguards.edge_threshold:
                self._log_filtered_signal(reason, edge_yes, edge_no, time_remaining)
            return

        # Entry decision based on edge
        threshold = self.safeguards.edge_threshold

        if edge_yes >= threshold:
            # BUY YES
            position_size = self.calculate_position_size(edge_yes)
            if position_size > 0:
                self.execute_entry("yes", position_size, edge_yes, p_yes)

        elif edge_no >= threshold:
            # BUY NO
            position_size = self.calculate_position_size(edge_no)
            if position_size > 0:
                self.execute_entry("no", position_size, edge_no, p_yes)

    def execute_entry(self, side: str, position_size: float, edge: float, p_yes: float):
        """Execute position entry."""
        price = self.state.last_yes_price if side == "yes" else self.state.last_no_price

        self.state.position_side = side
        self.state.position_size = position_size
        self.state.position_dollar_size = position_size * self.state.balance
        self.state.entry_price = price
        self.state.entry_tick = self._tick_count
        self.state.ticks_held = 0
        self.state.daily_trades += 1

        console.print(
            f"[green]ENTRY: {side.upper()} @ {price:.3f} | "
            f"Size={position_size:.1%} (${self.state.position_dollar_size:.0f}) | "
            f"Edge={edge:+.1%} | P(YES)={p_yes:.3f}[/green]"
        )

        self._log_trade("entry", {
            "side": side,
            "price": price,
            "size": position_size,
            "dollar_size": self.state.position_dollar_size,
            "edge": edge,
            "p_yes": p_yes,
            "time_remaining": self.get_time_remaining(),
            "drawdown": self.state.current_drawdown,
            "consecutive_losses": self.state.consecutive_losses,
        })

    def execute_exit(self, reason: str):
        """Execute position exit with safeguard updates."""
        if self.state.position_side is None:
            return

        side = self.state.position_side
        entry_price = self.state.entry_price

        if side == "yes":
            current_price = self.state.last_yes_price
        else:
            current_price = self.state.last_no_price

        # Calculate PnL
        invested = self.state.position_dollar_size
        shares = invested / entry_price
        exit_value = shares * current_price
        pnl = exit_value - invested

        # Update state
        self.state.balance += pnl
        self.state.total_pnl += pnl
        self.state.daily_pnl += pnl

        trade_won = pnl > 0
        if trade_won:
            self.state.wins += 1
        else:
            self.state.losses += 1

        # Update circuit breaker
        self.update_circuit_breaker(trade_won)

        # Update peak/drawdown
        self.state.update_drawdown()

        # Record trade
        trade = {
            "time": datetime.now(timezone.utc),
            "side": side,
            "entry": entry_price,
            "exit": current_price,
            "pnl": pnl,
            "pnl_pct": pnl / invested if invested > 0 else 0,
            "reason": reason,
            "ticks_held": self.state.ticks_held,
        }
        self.state.trades.append(trade)

        pnl_color = 'green' if pnl > 0 else 'red'
        console.print(
            f"[{pnl_color}]EXIT: {side.upper()} @ {current_price:.3f} | "
            f"PnL=${pnl:+.2f} ({pnl/invested*100:+.1f}%) | {reason} | "
            f"Held {self.state.ticks_held} ticks[/]"
        )

        self._log_trade("exit", {
            "side": side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl / invested if invested > 0 else 0,
            "reason": reason,
            "ticks_held": self.state.ticks_held,
            "balance_after": self.state.balance,
            "drawdown_after": self.state.current_drawdown,
            "consecutive_losses": self.state.consecutive_losses,
        })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.position_dollar_size = 0.0
        self.state.entry_price = 0.0
        self.state.entry_tick = 0
        self.state.ticks_held = 0

    def settle_position(self):
        """Settle position at candle end (Rule-based exit #3)."""
        if not self.state.position_side:
            return

        if not self.state.btc_open_price or not self.state.btc_current_price:
            return

        btc_move = self.state.btc_current_price - self.state.btc_open_price
        up_won = btc_move > 0

        # Settlement payout
        if self.state.position_side == "yes":
            payout = 1.0 if up_won else 0.0
        else:
            payout = 1.0 if not up_won else 0.0

        invested = self.state.position_dollar_size
        shares = invested / self.state.entry_price
        returned_capital = shares * payout
        pnl = returned_capital - invested

        # Update state
        self.state.balance += pnl
        self.state.total_pnl += pnl
        self.state.daily_pnl += pnl

        trade_won = pnl > 0
        if trade_won:
            self.state.wins += 1
        else:
            self.state.losses += 1

        self.update_circuit_breaker(trade_won)
        self.state.update_drawdown()

        trade = {
            "time": datetime.now(timezone.utc),
            "side": self.state.position_side,
            "entry": self.state.entry_price,
            "exit": payout,
            "pnl": pnl,
            "pnl_pct": pnl / invested if invested > 0 else 0,
            "reason": "settlement",
            "ticks_held": self.state.ticks_held,
            "outcome": "up" if up_won else "down",
        }
        self.state.trades.append(trade)

        result_str = "WIN" if pnl > 0 else "LOSS"
        pnl_color = "green" if pnl > 0 else "red"
        console.print(
            f"[bold purple]SETTLEMENT: {self.state.position_side.upper()} -> "
            f"[{pnl_color}]{result_str}[/] | PnL=${pnl:+.2f}[/bold purple]"
        )

        self._log_trade("settlement", {
            "side": self.state.position_side,
            "entry_price": self.state.entry_price,
            "payout": payout,
            "btc_open": self.state.btc_open_price,
            "btc_close": self.state.btc_current_price,
            "up_won": up_won,
            "pnl": pnl,
            "balance_after": self.state.balance,
        })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.position_dollar_size = 0.0
        self.state.entry_price = 0.0

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    async def _setup_clients(self):
        """Setup HTTP clients."""
        self.binance_client = httpx.AsyncClient(timeout=10)

        if SOCKS_AVAILABLE:
            transport = httpx_socks.AsyncProxyTransport.from_url(SOCKS5_PROXY_URL)
            self.polymarket_client = httpx.AsyncClient(transport=transport, timeout=30, verify=False)
        else:
            console.print("[yellow]SOCKS5 not available, using direct connection[/yellow]")
            self.polymarket_client = httpx.AsyncClient(timeout=30)

    def get_current_candle_timestamp(self) -> int:
        """Get current 15-min candle timestamp."""
        now = int(datetime.now(timezone.utc).timestamp())
        return (now // 900) * 900

    def get_time_remaining(self) -> float:
        """Get fraction of 15-min candle remaining."""
        now = datetime.now(timezone.utc)
        minute_block = (now.minute // 15) * 15
        start_of_candle = now.replace(minute=minute_block, second=0, microsecond=0)
        duration = timedelta(minutes=15)
        elapsed = now - start_of_candle
        remaining_seconds = (duration - elapsed).total_seconds()
        return max(0.0, min(1.0, remaining_seconds / duration.total_seconds()))

    async def fetch_active_market(self, timestamp: int) -> bool:
        """Fetch current active market IDs."""
        slug = f"btc-updown-15m-{timestamp}"
        try:
            url = f"{self.config.polymarket_gamma_api}/events/slug/{slug}"
            response = await self.polymarket_client.get(url)

            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                if markets:
                    market = markets[0]
                    clob_tokens = market.get("clobTokenIds", "[]")
                    if isinstance(clob_tokens, str):
                        clob_tokens = json.loads(clob_tokens)

                    if len(clob_tokens) >= 2:
                        self.state.active_yes_id = clob_tokens[0]
                        self.state.active_no_id = clob_tokens[1]
                        self.state.current_candle_ts = timestamp
                        console.print(f"[blue]Found market: {slug}[/blue]")
                        return True
        except Exception as e:
            logger.error(f"Market discovery error: {e}")
        return False

    async def fetch_polymarket_prices(self) -> Optional[Dict[str, float]]:
        """Fetch current YES/NO prices."""
        if not self.state.active_yes_id:
            return None

        try:
            url = f"{self.config.polymarket_api}/midpoint"
            params = {"token_id": self.state.active_yes_id}
            response = await self.polymarket_client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                yes_price = float(data.get("mid", 0.5))
                return {
                    "yes_price": yes_price,
                    "no_price": 1.0 - yes_price,
                }

            # Fallback to orderbook
            url = f"{self.config.polymarket_api}/book"
            params = {"token_id": self.state.active_yes_id}
            response = await self.polymarket_client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                if bids and asks:
                    yes_price = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
                elif asks:
                    yes_price = float(asks[0]["price"])
                elif bids:
                    yes_price = float(bids[0]["price"])
                else:
                    yes_price = 0.5

                return {"yes_price": yes_price, "no_price": 1.0 - yes_price}

        except Exception as e:
            logger.error(f"Polymarket fetch error: {e}")
        return None

    async def fetch_btc_price(self) -> Optional[float]:
        """Fetch current BTC price."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": "BTCUSDT"}
            response = await self.binance_client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                return float(data["price"])
        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
        return None

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _setup_ml_logging(self):
        """Initialize ML logging file."""
        if not self.config.enable_ml_logging:
            return

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.config.model_path).name
        self.ml_log_file = log_dir / f"{model_name}_safeguarded_{timestamp}.jsonl"

        self.ml_log_handle = open(self.ml_log_file, 'w')
        console.print(f"[blue]ML logging to: {self.ml_log_file}[/blue]")

        # Write metadata
        metadata = {
            "type": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": self.config.model_path,
            "model_type": self.config.model_type,
            "safeguards": {
                "edge_threshold": self.safeguards.edge_threshold,
                "stop_loss_pct": self.safeguards.stop_loss_pct,
                "min_time_remaining": self.safeguards.min_time_remaining,
                "base_position_size": self.safeguards.base_position_size,
                "max_position_size": self.safeguards.max_position_size,
                "max_consecutive_losses": self.safeguards.max_consecutive_losses,
                "drawdown_tiers": [
                    self.safeguards.drawdown_tier1,
                    self.safeguards.drawdown_tier2,
                    self.safeguards.drawdown_tier3,
                ],
            },
            "config": {
                "initial_balance": self.config.initial_balance,
            }
        }
        self.ml_log_handle.write(json.dumps(metadata) + "\n")
        self.ml_log_handle.flush()

    def _log_tick(self):
        """Log tick data."""
        if not self.config.enable_ml_logging or self.ml_log_handle is None:
            return

        self._tick_count += 1

        log_entry = {
            "type": "tick",
            "tick": self._tick_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candle_ts": self.state.current_candle_ts,
            "time_remaining": self.get_time_remaining(),
            "market": {
                "yes_price": self.state.last_yes_price,
                "no_price": self.state.last_no_price,
                "btc_price": self.state.btc_current_price,
                "btc_open": self.state.btc_open_price,
            },
            "model": {
                "p_yes": self.state.last_p_yes,
                "edge_yes": self.state.last_edge_yes,
                "edge_no": self.state.last_edge_no,
            },
            "position": {
                "side": self.state.position_side,
                "size": self.state.position_size,
                "entry_price": self.state.entry_price,
                "ticks_held": self.state.ticks_held,
            },
            "account": {
                "balance": self.state.balance,
                "total_pnl": self.state.total_pnl,
                "drawdown": self.state.current_drawdown,
                "consecutive_losses": self.state.consecutive_losses,
                "trading_paused": self.state.trading_paused,
            },
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")

        if self._tick_count % 10 == 0:
            self.ml_log_handle.flush()

    def _log_trade(self, action: str, details: Dict[str, Any]):
        """Log trade events."""
        if not self.config.enable_ml_logging or self.ml_log_handle is None:
            return

        log_entry = {
            "type": "trade",
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tick": self._tick_count,
            **details
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")
        self.ml_log_handle.flush()

    def _log_filtered_signal(self, reason: str, edge_yes: float, edge_no: float, time_remaining: float):
        """Log signals that were filtered by safeguards."""
        if not self.config.enable_ml_logging or self.ml_log_handle is None:
            return

        log_entry = {
            "type": "filtered_signal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tick": self._tick_count,
            "reason": reason,
            "edge_yes": edge_yes,
            "edge_no": edge_no,
            "time_remaining": time_remaining,
            "drawdown": self.state.current_drawdown,
            "consecutive_losses": self.state.consecutive_losses,
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")

    def _close_ml_logging(self):
        """Close ML logging file."""
        if self.ml_log_handle:
            summary = {
                "type": "summary",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_ticks": self._tick_count,
                "final_balance": self.state.balance,
                "total_pnl": self.state.total_pnl,
                "wins": self.state.wins,
                "losses": self.state.losses,
                "win_rate": self.state.wins / max(1, self.state.wins + self.state.losses),
                "max_drawdown": self.state.current_drawdown,
                "total_trades": len(self.state.trades),
            }
            self.ml_log_handle.write(json.dumps(summary) + "\n")
            self.ml_log_handle.close()
            console.print(f"[green]ML log saved: {self.ml_log_file}[/green]")

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def build_display(self) -> Panel:
        """Build display panel."""
        # Left: Status
        left_table = Table(show_header=False, box=None, padding=(0, 1))
        left_table.add_column("Key", style="dim", width=16)
        left_table.add_column("Value", style="bold")

        left_table.add_row("Balance", f"${self.state.balance:.2f}")
        pnl_color = "green" if self.state.total_pnl >= 0 else "red"
        left_table.add_row("Total PnL", f"[{pnl_color}]${self.state.total_pnl:+.2f}[/]")

        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        left_table.add_row("Win Rate", f"{win_rate:.1%} ({self.state.wins}/{self.state.wins + self.state.losses})")

        dd_color = "green" if self.state.current_drawdown < 0.1 else "yellow" if self.state.current_drawdown < 0.2 else "red"
        left_table.add_row("Drawdown", f"[{dd_color}]{self.state.current_drawdown:.1%}[/]")

        left_table.add_row("", "")
        left_table.add_row("YES Price", f"{self.state.last_yes_price:.3f}")
        left_table.add_row("NO Price", f"{self.state.last_no_price:.3f}")
        left_table.add_row("Time Left", f"{self.get_time_remaining():.1%}")

        # Model output
        left_table.add_row("", "")
        left_table.add_row("P(YES)", f"{self.state.last_p_yes:.3f}")

        edge_yes_color = "green" if self.state.last_edge_yes > 0 else "red"
        edge_no_color = "green" if self.state.last_edge_no > 0 else "red"
        left_table.add_row("Edge YES", f"[{edge_yes_color}]{self.state.last_edge_yes:+.1%}[/]")
        left_table.add_row("Edge NO", f"[{edge_no_color}]{self.state.last_edge_no:+.1%}[/]")

        # Safeguard status
        left_table.add_row("", "")
        if self.state.trading_paused:
            left_table.add_row("Status", f"[red]PAUSED ({self.state.cooldown_remaining})[/]")
        elif self.state.consecutive_losses > 0:
            left_table.add_row("Status", f"[yellow]Loss streak: {self.state.consecutive_losses}[/]")
        else:
            left_table.add_row("Status", "[green]Active[/]")

        # BTC
        left_table.add_row("", "")
        btc_open = self.state.btc_open_price or 0
        btc_current = self.state.btc_current_price or 0
        btc_change = ((btc_current - btc_open) / btc_open * 100) if btc_open > 0 else 0
        btc_color = "green" if btc_change > 0 else "red" if btc_change < 0 else "white"
        left_table.add_row("BTC Open", f"${btc_open:,.0f}")
        left_table.add_row("BTC Now", f"${btc_current:,.0f} [{btc_color}]({btc_change:+.2f}%)[/]")

        # Position
        if self.state.position_side:
            left_table.add_row("", "")
            color = "green" if self.state.position_side == "yes" else "red"
            current_price = (
                self.state.last_yes_price if self.state.position_side == "yes"
                else self.state.last_no_price
            )
            unrealized_pnl = (current_price - self.state.entry_price) / self.state.entry_price
            pnl_color = "green" if unrealized_pnl > 0 else "red"

            left_table.add_row(
                "Position",
                f"[{color}]{self.state.position_side.upper()}[/] @ {self.state.entry_price:.3f}"
            )
            left_table.add_row("Size", f"{self.state.position_size:.1%} (${self.state.position_dollar_size:.0f})")
            left_table.add_row("Unrealized", f"[{pnl_color}]{unrealized_pnl:+.1%}[/]")
            left_table.add_row("Ticks Held", f"{self.state.ticks_held}")

        # Right: Trades
        trades_table = Table(show_header=True, box=None, padding=(0, 1))
        trades_table.add_column("Time", style="dim", width=8)
        trades_table.add_column("Side", width=4)
        trades_table.add_column("Entry", width=6)
        trades_table.add_column("Exit", width=6)
        trades_table.add_column("PnL", width=10)
        trades_table.add_column("Reason", width=8)

        recent_trades = list(reversed(self.state.trades[-15:]))
        for trade in recent_trades:
            side_color = "green" if trade["side"] == "yes" else "red"
            pnl_color = "green" if trade["pnl"] > 0 else "red"
            time_str = trade["time"].strftime("%H:%M:%S")

            reason_short = {
                "stop_loss": "STOP",
                "time_expiry": "TIME",
                "settlement": "SETTLE",
            }.get(trade.get("reason", ""), trade.get("reason", "")[:6])

            trades_table.add_row(
                time_str,
                f"[{side_color}]{trade['side'].upper()[:3]}[/]",
                f"{trade['entry']:.3f}",
                f"{trade['exit']:.3f}",
                f"[{pnl_color}]${trade['pnl']:+.2f}[/]",
                reason_short,
            )

        if not recent_trades:
            trades_table.add_row("", "", "[dim]No trades yet[/]", "", "", "")

        left_panel = Panel(left_table, title="[bold cyan]Status[/]", border_style="cyan", width=40)
        right_panel = Panel(trades_table, title="[bold yellow]Recent Trades[/]", border_style="yellow")

        layout = Columns([left_panel, right_panel], expand=True)

        title = "[bold blue]Safeguarded Paper Trading[/bold blue]"
        if self.state.trading_paused:
            title += " [red][PAUSED][/red]"

        return Panel(layout, title=title, border_style="blue")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def trading_loop(self):
        """Main trading loop."""
        console.print("[bold green]Starting safeguarded paper trading...[/bold green]")
        console.print(f"[dim]Edge threshold: {self.safeguards.edge_threshold:.1%}[/dim]")
        console.print(f"[dim]Stop loss: {self.safeguards.stop_loss_pct:.1%}[/dim]")
        console.print(f"[dim]Min time remaining: {self.safeguards.min_time_remaining:.1%}[/dim]")

        while self._running:
            try:
                # Market discovery
                current_ts = self.get_current_candle_timestamp()
                if self.state.current_candle_ts != current_ts:
                    if self.state.current_candle_ts is not None:
                        self.settle_position()

                    found = await self.fetch_active_market(current_ts)
                    if found:
                        self.state.yes_price_history = []
                        self.state.no_price_history = []
                        self.state.btc_price_history = []
                        self.state.btc_open_price = None
                    else:
                        console.print("[yellow]Waiting for new market...[/yellow]")
                        await asyncio.sleep(5)
                        continue

                # Fetch prices
                poly_prices = await self.fetch_polymarket_prices()
                btc_price = await self.fetch_btc_price()

                if poly_prices:
                    self.state.last_yes_price = poly_prices["yes_price"]
                    self.state.last_no_price = poly_prices["no_price"]
                    self.state.yes_price_history.append(poly_prices["yes_price"])
                    self.state.no_price_history.append(poly_prices["no_price"])

                if btc_price:
                    self.state.btc_current_price = btc_price
                    self.state.btc_price_history.append(btc_price)
                    if self.state.btc_open_price is None:
                        self.state.btc_open_price = btc_price

                # Log tick
                self._log_tick()

                # Execute trading logic
                self.execute_trading_logic()

                # Bound history
                max_history = 1000
                if len(self.state.yes_price_history) > max_history:
                    self.state.yes_price_history = self.state.yes_price_history[-max_history:]
                    self.state.no_price_history = self.state.no_price_history[-max_history:]
                    self.state.btc_price_history = self.state.btc_price_history[-max_history:]

                await asyncio.sleep(self.config.refresh_seconds)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Run paper trading with live display."""
        self.load_model()
        self._setup_ml_logging()
        await self._setup_clients()

        self._running = True

        trading_task = asyncio.create_task(self.trading_loop())

        try:
            with Live(self.build_display(), refresh_per_second=1) as live:
                while self._running:
                    live.update(self.build_display())
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            self._running = False
            trading_task.cancel()
        finally:
            self._close_ml_logging()


async def main():
    parser = argparse.ArgumentParser(description="Safeguarded paper trading")
    parser.add_argument("--model", type=str, default="./logs/market_predictor_v2.5")
    parser.add_argument("--model-type", type=str, default="legacy", choices=["legacy", "probability"])
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--edge-threshold", type=float, default=0.07)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--min-time", type=float, default=0.35)
    parser.add_argument("--log-dir", type=str, default="./logs/paper_trade_safeguarded")
    parser.add_argument("--no-ml-log", action="store_true")

    args = parser.parse_args()

    safeguards = SafeguardConfig(
        edge_threshold=args.edge_threshold,
        stop_loss_pct=args.stop_loss,
        min_time_remaining=args.min_time,
    )

    config = SafeguardedPaperTradeConfig(
        model_path=args.model,
        model_type=args.model_type,
        initial_balance=args.balance,
        safeguards=safeguards,
        log_dir=args.log_dir,
        enable_ml_logging=not args.no_ml_log,
    )

    trader = SafeguardedPaperTrader(config)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
