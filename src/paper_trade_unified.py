#!/usr/bin/env python3
"""
Unified Paper Trading for Polymarket.

Uses the new MarketPredictorModel which:
1. Makes unified action decisions (WAIT, BUY_YES, BUY_NO, EXIT, HOLD)
2. Outputs expected returns and confidence
3. Was trained on REAL historical data with optimal action labels
4. Has NO hardcoded thresholds - everything is learned

Key differences from token-centric paper trader:
- Single model replaces EdgeDetector + SAC
- Model directly outputs actions, not separate edge + policy
- Uses enhanced 71-feature set with better trend/time awareness
- Position sizing based on predicted expected return
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
from typing import Optional, Dict, Any, List

import httpx
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.data.enhanced_features import EnhancedFeatureBuilder
from src.models.market_predictor import (
    MarketPredictorModel,
    EnhancedPositionState,
    Action,
    load_market_predictor,
)

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


@dataclass
class UnifiedTradingState:
    """Trading state for unified approach."""
    balance: float = 1000.0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0

    # Current Market Info
    current_candle_ts: Optional[int] = None
    active_yes_id: Optional[str] = None
    active_no_id: Optional[str] = None

    # Current position
    position_side: Optional[str] = None  # "yes" or "no"
    position_size: float = 0.0
    entry_price: float = 0.0
    entry_tick: int = 0
    ticks_held: int = 0
    max_pnl_seen: float = 0.0

    # Market state
    last_yes_price: float = 0.5
    last_no_price: float = 0.5
    btc_open_price: Optional[float] = None
    btc_current_price: Optional[float] = None

    # Price history
    yes_price_history: List[float] = field(default_factory=list)
    no_price_history: List[float] = field(default_factory=list)
    btc_price_history: List[float] = field(default_factory=list)

    # Model outputs
    last_action: int = 0  # Action.WAIT
    last_action_logits: List[float] = field(default_factory=lambda: [0.0] * 5)
    last_confidence: float = 0.0
    last_expected_return: float = 0.0

    # Trade history
    trades: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UnifiedPaperTradeConfig:
    """Configuration for unified paper trading."""
    model_path: str = "./logs/market_predictor_v1"

    initial_balance: float = 1000.0

    # Position sizing based on expected return
    base_position_size: float = 0.10  # Base 10% of balance
    max_position_size: float = 0.25  # Max 25%
    min_position_size: float = 0.05  # Min 5%

    # Entry filters (soft - model learns these, but we add safety)
    min_confidence: float = 0.3  # Only act on confident predictions
    min_expected_return: float = 0.02  # Only if model predicts >2% return
    min_time_remaining: float = 0.05  # Don't enter in last 5% of candle

    # Polymarket API
    polymarket_api: str = "https://clob.polymarket.com"
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"

    # Refresh interval
    refresh_seconds: int = 5

    # Logging
    log_dir: str = "./logs/paper_trade_unified"
    enable_ml_logging: bool = True


class UnifiedPaperTrader:
    """Paper trader using unified MarketPredictor model."""

    def __init__(self, config: UnifiedPaperTradeConfig):
        self.config = config
        self.state = UnifiedTradingState(balance=config.initial_balance)
        self.feature_builder = EnhancedFeatureBuilder()

        # Model
        self.model: Optional[MarketPredictorModel] = None
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

    def _setup_ml_logging(self):
        """Initialize ML logging file."""
        if not self.config.enable_ml_logging:
            return

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.config.model_path).name
        self.ml_log_file = log_dir / f"{model_name}_{timestamp}.jsonl"

        self.ml_log_handle = open(self.ml_log_file, 'w')
        console.print(f"[blue]ML logging to: {self.ml_log_file}[/blue]")

        # Write header/metadata
        metadata = {
            "type": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": self.config.model_path,
            "config": {
                "initial_balance": self.config.initial_balance,
                "base_position_size": self.config.base_position_size,
                "min_confidence": self.config.min_confidence,
                "min_expected_return": self.config.min_expected_return,
            }
        }
        self.ml_log_handle.write(json.dumps(metadata) + "\n")
        self.ml_log_handle.flush()

    def _log_tick(self, features: np.ndarray, position_state: np.ndarray, model_output: Dict):
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

            "model_output": {
                "action": model_output["action"],
                "action_name": Action.names()[model_output["action"]],
                "q_values": model_output["q_values"],
                "confidence": model_output["confidence"],
                "expected_return": model_output["expected_return"],
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
                "wins": self.state.wins,
                "losses": self.state.losses,
            },

            "features_dim": len(features),
            "position_state": position_state.tolist(),
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
                "total_trades": len(self.state.trades),
            }
            self.ml_log_handle.write(json.dumps(summary) + "\n")
            self.ml_log_handle.close()
            console.print(f"[green]ML log saved: {self.ml_log_file}[/green]")

    def load_model(self):
        """Load the unified market predictor model."""
        console.print("[bold blue]Loading model...[/bold blue]")

        model_path = Path(self.config.model_path)
        if model_path.exists():
            self.model = load_market_predictor(str(model_path), self.device)
            self.model.eval()
            console.print(f"  [green]Model loaded from {model_path}[/green]")
        else:
            console.print(f"  [red]Model not found at {model_path}[/red]")
            raise FileNotFoundError(f"Model not found: {model_path}")

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

    def get_model_action(self) -> Dict[str, Any]:
        """Get action from unified model."""
        if self.model is None:
            return {
                "action": Action.WAIT,
                "q_values": [0.0] * 5,
                "confidence": 0.0,
                "expected_return": 0.0,
            }

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

        # Build position state
        if self.state.position_side:
            current_price = (
                self.state.last_yes_price if self.state.position_side == "yes"
                else self.state.last_no_price
            )
        else:
            current_price = self.state.last_yes_price

        position_state = EnhancedPositionState.compute(
            has_position=self.state.position_side is not None,
            position_side=self.state.position_side,
            entry_price=self.state.entry_price,
            current_price=current_price,
            time_remaining=time_remaining,
            ticks_held=self.state.ticks_held,
            max_pnl_seen=self.state.max_pnl_seen,
        )

        # Get model prediction
        features_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        position_t = torch.FloatTensor(position_state).unsqueeze(0).to(self.device)

        result = self.model.get_action(features_t, position_t, deterministic=True)

        action = int(result["action"].item())
        q_values = result["q_values"].squeeze(0).cpu().numpy().tolist()
        confidence = float(result["confidence"].item())
        expected_return = float(result["expected_return"].item())

        # Update state
        self.state.last_action = action
        self.state.last_q_values = q_values
        self.state.last_confidence = confidence
        self.state.last_expected_return = expected_return

        # Log
        self._log_tick(features, position_state, {
            "action": action,
            "q_values": q_values,
            "confidence": confidence,
            "expected_return": expected_return,
        })

        return {
            "action": action,
            "q_values": q_values,
            "confidence": confidence,
            "expected_return": expected_return,
            "features": features,
            "position_state": position_state,
        }

    def calculate_position_size(self, expected_return: float, confidence: float) -> float:
        """Calculate position size based on model outputs."""
        # Scale position with expected return and confidence
        # Higher expected return = larger position
        # Higher confidence = larger position

        # Base size scales linearly with expected return
        return_factor = np.clip(expected_return / 0.10, 0.5, 2.0)  # 0.5x to 2x based on return

        # Confidence factor
        conf_factor = 0.5 + 0.5 * confidence  # 0.5x to 1x based on confidence

        size = self.config.base_position_size * return_factor * conf_factor

        return float(np.clip(size, self.config.min_position_size, self.config.max_position_size))

    async def execute_trading_logic(self, model_output: Dict):
        """Execute trading based on model output."""
        action = model_output["action"]
        confidence = model_output["confidence"]
        expected_return = model_output["expected_return"]
        time_remaining = self.get_time_remaining()

        # Update position tracking
        if self.state.position_side:
            self.state.ticks_held += 1
            current_price = (
                self.state.last_yes_price if self.state.position_side == "yes"
                else self.state.last_no_price
            )
            current_pnl = (current_price - self.state.entry_price) / (self.state.entry_price + 1e-8)
            self.state.max_pnl_seen = max(self.state.max_pnl_seen, current_pnl)

        # Handle model's action
        if action == Action.WAIT:
            # Do nothing - model says wait
            return

        elif action == Action.BUY_YES:
            if self.state.position_side is not None:
                return  # Already have position

            # Safety filters (soft - model should have learned these)
            if confidence < self.config.min_confidence:
                return
            if expected_return < self.config.min_expected_return:
                return
            if time_remaining < self.config.min_time_remaining:
                return

            size = self.calculate_position_size(expected_return, confidence)
            self.execute_entry("yes", size, model_output)

        elif action == Action.BUY_NO:
            if self.state.position_side is not None:
                return  # Already have position

            # Safety filters
            if confidence < self.config.min_confidence:
                return
            if expected_return < self.config.min_expected_return:
                return
            if time_remaining < self.config.min_time_remaining:
                return

            size = self.calculate_position_size(expected_return, confidence)
            self.execute_entry("no", size, model_output)

        elif action == Action.EXIT:
            if self.state.position_side is None:
                return  # No position to exit
            self.execute_exit("model_signal")

        elif action == Action.HOLD:
            # Model says hold current position - do nothing
            pass

        # Force exit near settlement (safety net)
        if self.state.position_side and time_remaining < 0.02:
            self.execute_exit("time_expiry")

    def execute_entry(self, side: str, position_size: float, model_output: Dict):
        """Execute position entry."""
        price = self.state.last_yes_price if side == "yes" else self.state.last_no_price

        self.state.position_side = side
        self.state.position_size = position_size
        self.state.entry_price = price
        self.state.entry_tick = self._tick_count
        self.state.ticks_held = 0
        self.state.max_pnl_seen = 0.0

        dollar_size = position_size * self.state.balance

        console.print(
            f"[green]ENTRY: {side.upper()} @ {price:.3f} | "
            f"Size={position_size:.1%} (${dollar_size:.0f}) | "
            f"E[R]={model_output['expected_return']:+.1%} | "
            f"Conf={model_output['confidence']:.1%}[/green]"
        )

        self._log_trade("entry", {
            "side": side,
            "price": price,
            "size": position_size,
            "dollar_size": dollar_size,
            "expected_return": model_output["expected_return"],
            "confidence": model_output["confidence"],
            "q_values": model_output["q_values"],
            "time_remaining": self.get_time_remaining(),
        })

    def execute_exit(self, reason: str = "manual"):
        """Execute position exit."""
        if self.state.position_side is None:
            return

        side = self.state.position_side
        entry_price = self.state.entry_price
        current_price = (
            self.state.last_yes_price if side == "yes"
            else self.state.last_no_price
        )

        invested = self.state.position_size * self.state.balance
        shares = invested / entry_price
        exit_value = shares * current_price
        pnl = exit_value - invested

        self.state.balance += pnl
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": side,
            "entry": entry_price,
            "exit": current_price,
            "pnl": pnl,
            "reason": reason,
            "ticks_held": self.state.ticks_held,
        })

        pnl_color = 'green' if pnl > 0 else 'red'
        console.print(
            f"[{pnl_color}]EXIT: {side.upper()} @ {current_price:.3f} | "
            f"PnL=${pnl:+.2f} | {reason} | Held {self.state.ticks_held} ticks[/]"
        )

        self._log_trade("exit", {
            "side": side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": pnl,
            "reason": reason,
            "ticks_held": self.state.ticks_held,
            "balance_after": self.state.balance,
        })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.entry_price = 0.0
        self.state.entry_tick = 0
        self.state.ticks_held = 0
        self.state.max_pnl_seen = 0.0

    def settle_position(self):
        """Settle position at candle end."""
        if not self.state.position_side:
            return

        if not self.state.btc_open_price or not self.state.btc_current_price:
            return

        btc_move = self.state.btc_current_price - self.state.btc_open_price
        up_won = btc_move > 0

        # Settlement payout
        payout = 0.0
        if self.state.position_side == "yes":
            payout = 1.0 if up_won else 0.0
        else:
            payout = 1.0 if not up_won else 0.0

        invested = self.state.position_size * self.state.balance
        shares = invested / self.state.entry_price
        returned_capital = shares * payout
        pnl = returned_capital - invested

        self.state.balance += pnl
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": self.state.position_side,
            "entry": self.state.entry_price,
            "exit": payout,
            "pnl": pnl,
            "reason": "settlement",
            "ticks_held": self.state.ticks_held,
        })

        console.print(
            f"[bold purple]SETTLEMENT: {self.state.position_side.upper()} -> "
            f"{'WIN' if pnl > 0 else 'LOSS'} | PnL=${pnl:+.2f}[/bold purple]"
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
        self.state.entry_price = 0.0

    async def trading_loop(self):
        """Main trading loop."""
        console.print("[bold green]Starting unified paper trading...[/bold green]")

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

                # Get model action
                model_output = self.get_model_action()

                # Execute trading
                await self.execute_trading_logic(model_output)

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

    def build_display(self) -> Panel:
        """Build display panel."""
        # Left: Status
        left_table = Table(show_header=False, box=None, padding=(0, 1))
        left_table.add_column("Key", style="dim", width=14)
        left_table.add_column("Value", style="bold")

        left_table.add_row("Balance", f"${self.state.balance:.2f}")
        pnl_color = "green" if self.state.total_pnl >= 0 else "red"
        left_table.add_row("Total PnL", f"[{pnl_color}]${self.state.total_pnl:+.2f}[/]")

        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        left_table.add_row("Win Rate", f"{win_rate:.1%} ({self.state.wins}/{self.state.wins + self.state.losses})")

        left_table.add_row("", "")
        left_table.add_row("YES Price", f"{self.state.last_yes_price:.3f}")
        left_table.add_row("NO Price", f"{self.state.last_no_price:.3f}")
        left_table.add_row("Time Left", f"{self.get_time_remaining():.1%}")

        # BTC
        left_table.add_row("", "")
        btc_open = self.state.btc_open_price or 0
        btc_current = self.state.btc_current_price or 0
        btc_change = ((btc_current - btc_open) / btc_open * 100) if btc_open > 0 else 0
        btc_color = "green" if btc_change > 0 else "red" if btc_change < 0 else "white"
        left_table.add_row("BTC Open", f"${btc_open:,.0f}")
        left_table.add_row("BTC Now", f"${btc_current:,.0f} [{btc_color}]({btc_change:+.2f}%)[/]")

        # Model output
        left_table.add_row("", "")
        action_name = Action.names()[self.state.last_action]
        action_color = {
            "WAIT": "dim",
            "BUY_YES": "green",
            "BUY_NO": "red",
            "EXIT": "yellow",
            "HOLD": "cyan",
        }.get(action_name, "white")

        left_table.add_row("Model Action", f"[{action_color}]{action_name}[/]")
        left_table.add_row("Confidence", f"{self.state.last_confidence:.1%}")
        left_table.add_row("E[Return]", f"{self.state.last_expected_return:+.1%}")

        # Q-values
        q_str = " ".join([f"{q:.2f}" for q in self.state.last_q_values])
        left_table.add_row("Q-values", f"[dim]{q_str}[/]")

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
            left_table.add_row("Size", f"{self.state.position_size:.1%}")
            left_table.add_row("Unrealized", f"[{pnl_color}]{unrealized_pnl:+.1%}[/]")
            left_table.add_row("Ticks Held", f"{self.state.ticks_held}")

        # Right: Trades
        trades_table = Table(show_header=True, box=None, padding=(0, 1))
        trades_table.add_column("Time", style="dim", width=8)
        trades_table.add_column("Side", width=4)
        trades_table.add_column("Entry", width=6)
        trades_table.add_column("Exit", width=6)
        trades_table.add_column("PnL", width=8)
        trades_table.add_column("Reason", width=6)

        recent_trades = list(reversed(self.state.trades[-10:]))
        for trade in recent_trades:
            side_color = "green" if trade["side"] == "yes" else "red"
            pnl_color = "green" if trade["pnl"] > 0 else "red"
            time_str = trade["time"].strftime("%H:%M:%S")

            reason_short = {
                "model_signal": "MODEL",
                "time_expiry": "TIME",
                "settlement": "SETT",
            }.get(trade.get("reason", ""), trade.get("reason", "")[:5])

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

        left_panel = Panel(left_table, title="[bold cyan]Status[/]", border_style="cyan", width=38)
        right_panel = Panel(trades_table, title="[bold yellow]Recent Trades[/]", border_style="yellow")

        layout = Columns([left_panel, right_panel], expand=True)
        return Panel(layout, title="[bold blue]Unified Paper Trading[/bold blue]", border_style="blue")

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


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    model_path: str = "./logs/market_predictor_v1"
    initial_balance: float = 1000.0

    # Position sizing
    base_position_size: float = 0.10
    max_position_size: float = 0.25
    min_position_size: float = 0.05

    # Entry filters
    min_confidence: float = 0.3
    min_expected_return: float = 0.02
    min_time_remaining: float = 0.05

    # Data source: "historical" (API parquet) or "onchain" (local node JSON)
    data_source: str = "historical"

    # Historical data paths (parquet from API)
    candles_path: str = "./data/historical/polymarket_candles_30d.parquet"
    prices_path: str = "./data/historical/polymarket_prices_30d.parquet"
    btc_path: str = "./data/historical/btc_1m_30d.parquet"

    # On-chain data path (JSON from local node)
    onchain_data_path: str = "./data/onchain/btc_15min_current.json"

    # Simulation settings
    tick_interval_seconds: int = 30  # Simulate every 30 seconds (faster backtesting)
    max_candles: int = 0  # 0 = process all candles

    # Output
    log_dir: str = "./logs/backtest"
    verbose: bool = False


class BacktestRunner:
    """Backtest using historical data."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.feature_builder = EnhancedFeatureBuilder()

        # Model
        self.model: Optional[MarketPredictorModel] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data (historical parquet)
        self.candles_df: Optional[pd.DataFrame] = None
        self.prices_df: Optional[pd.DataFrame] = None
        self.btc_df: Optional[pd.DataFrame] = None

        # Data (on-chain JSON)
        self.onchain_data: Optional[Dict[str, Any]] = None

        # Results
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

        # Logging
        self.ml_log_file: Optional[Path] = None
        self.ml_log_handle = None
        self._tick_count = 0

    def _setup_logging(self):
        """Initialize logging file."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.config.model_path).name
        self.ml_log_file = log_dir / f"backtest_log_{model_name}_{timestamp}.jsonl"

        self.ml_log_handle = open(self.ml_log_file, 'w')
        console.print(f"[blue]Logging to: {self.ml_log_file}[/blue]")

        # Write metadata
        metadata = {
            "type": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": self.config.model_path,
            "config": {
                "initial_balance": self.config.initial_balance,
                "base_position_size": self.config.base_position_size,
                "min_confidence": self.config.min_confidence,
                "min_expected_return": self.config.min_expected_return,
                "tick_interval_seconds": self.config.tick_interval_seconds,
            }
        }
        self.ml_log_handle.write(json.dumps(metadata) + "\n")
        self.ml_log_handle.flush()

    def _log_tick(self, state: UnifiedTradingState, model_output: Dict, time_remaining: float):
        """Log tick data."""
        if self.ml_log_handle is None:
            return

        self._tick_count += 1

        log_entry = {
            "type": "tick",
            "tick": self._tick_count,
            "candle_ts": state.current_candle_ts,
            "time_remaining": time_remaining,
            "market": {
                "yes_price": state.last_yes_price,
                "no_price": state.last_no_price,
                "btc_price": state.btc_current_price,
                "btc_open": state.btc_open_price,
            },
            "model_output": {
                "action": model_output["action"],
                "action_name": Action.names()[model_output["action"]],
                "q_values": model_output["q_values"],
                "confidence": model_output["confidence"],
                "expected_return": model_output["expected_return"],
            },
            "position": {
                "side": state.position_side,
                "size": state.position_size,
                "entry_price": state.entry_price,
                "ticks_held": state.ticks_held,
            },
            "account": {
                "balance": state.balance,
                "total_pnl": state.total_pnl,
            },
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")

        if self._tick_count % 100 == 0:
            self.ml_log_handle.flush()

    def _log_trade(self, action: str, details: Dict[str, Any]):
        """Log trade events."""
        if self.ml_log_handle is None:
            return

        log_entry = {
            "type": "trade",
            "action": action,
            "tick": self._tick_count,
            **details
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")
        self.ml_log_handle.flush()

    def _close_logging(self, state: UnifiedTradingState):
        """Close logging file with summary."""
        if self.ml_log_handle:
            summary = {
                "type": "summary",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_ticks": self._tick_count,
                "final_balance": state.balance,
                "total_pnl": state.total_pnl,
                "wins": state.wins,
                "losses": state.losses,
                "total_trades": len(self.trades),
            }
            self.ml_log_handle.write(json.dumps(summary) + "\n")
            self.ml_log_handle.close()
            console.print(f"[green]Log saved: {self.ml_log_file}[/green]")

    def load_model(self):
        """Load the market predictor model."""
        console.print("[bold blue]Loading model...[/bold blue]")
        model_path = Path(self.config.model_path)
        if model_path.exists():
            self.model = load_market_predictor(str(model_path), self.device)
            self.model.eval()
            console.print(f"  [green]Model loaded from {model_path}[/green]")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    def load_data(self):
        """Load historical data (parquet from API)."""
        console.print("[bold blue]Loading historical data...[/bold blue]")

        self.candles_df = pd.read_parquet(self.config.candles_path)
        console.print(f"  [green]Candles: {len(self.candles_df)} records[/green]")

        self.prices_df = pd.read_parquet(self.config.prices_path)
        console.print(f"  [green]Prices: {len(self.prices_df)} records[/green]")

        self.btc_df = pd.read_parquet(self.config.btc_path)
        console.print(f"  [green]BTC data: {len(self.btc_df)} records[/green]")

        # Convert timestamps
        if not pd.api.types.is_datetime64_any_dtype(self.btc_df['timestamp']):
            self.btc_df['timestamp'] = pd.to_datetime(self.btc_df['timestamp'], utc=True)

    def load_onchain_data(self):
        """Load on-chain data (JSON from local node)."""
        console.print("[bold blue]Loading on-chain data...[/bold blue]")

        with open(self.config.onchain_data_path) as f:
            self.onchain_data = json.load(f)

        yes_count = len(self.onchain_data.get('yes_prices', []))
        no_count = len(self.onchain_data.get('no_prices', []))
        console.print(f"  [green]Candle: {self.onchain_data.get('slug', 'unknown')}[/green]")
        console.print(f"  [green]YES prices: {yes_count} events[/green]")
        console.print(f"  [green]NO prices: {no_count} events[/green]")

        # Also load BTC data if available
        if Path(self.config.btc_path).exists():
            self.btc_df = pd.read_parquet(self.config.btc_path)
            if not pd.api.types.is_datetime64_any_dtype(self.btc_df['timestamp']):
                self.btc_df['timestamp'] = pd.to_datetime(self.btc_df['timestamp'], utc=True)
            console.print(f"  [green]BTC data: {len(self.btc_df)} records[/green]")

    def get_btc_price_at(self, ts: datetime) -> Optional[float]:
        """Get BTC price at a given timestamp."""
        if self.btc_df is None:
            return None
        # Handle timezone-aware datetimes
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = pd.Timestamp(ts)
        else:
            ts = pd.Timestamp(ts, tz='UTC')
        idx = self.btc_df['timestamp'].searchsorted(ts)
        if idx > 0 and idx <= len(self.btc_df):
            return float(self.btc_df.iloc[idx - 1]['close'])
        return None

    def get_btc_open_for_candle(self, candle_ts: int) -> Optional[float]:
        """Get BTC open price for a candle."""
        candle_start = datetime.fromtimestamp(candle_ts, tz=timezone.utc)
        return self.get_btc_price_at(candle_start)

    def simulate_candle(
        self,
        candle: pd.Series,
        prices: pd.DataFrame,
        state: UnifiedTradingState
    ) -> Dict[str, Any]:
        """Simulate trading through one candle."""
        candle_ts = int(candle['timestamp'])
        candle_start = datetime.fromtimestamp(candle_ts, tz=timezone.utc)
        candle_end = candle_start + timedelta(minutes=15)

        # Get BTC prices for this candle
        btc_open = self.get_btc_open_for_candle(candle_ts)

        # Filter prices for this candle
        candle_prices = prices[prices['candle_timestamp'] == candle_ts].copy()
        if len(candle_prices) == 0:
            return {"skipped": True, "reason": "no_prices"}

        # Sort by timestamp
        # Note: Prices span ~24h before candle (tokens created early for trading)
        # We use all available prices as they show how market evolves
        candle_prices = candle_prices.sort_values('timestamp')

        # Initialize state for this candle
        state.current_candle_ts = candle_ts
        state.btc_open_price = btc_open
        state.yes_price_history = []
        state.no_price_history = []
        state.btc_price_history = []

        # Get yes/no prices
        # NOTE: The collector ALWAYS fetches YES token prices (see historical_data_collector.py line 345)
        # The 'outcome' column indicates candle result, but prices are always for YES token
        raw_prices = candle_prices['price'].tolist()

        if not raw_prices:
            raw_prices = [0.5]

        # Raw prices are always YES token prices
        yes_prices = raw_prices
        no_prices = [1.0 - p for p in raw_prices]

        # Simulate ticks through the candle
        tick_interval = self.config.tick_interval_seconds
        candle_duration = 15 * 60  # 15 minutes in seconds
        num_ticks = candle_duration // tick_interval

        trade_result = None

        for tick_idx in range(num_ticks):
            elapsed_seconds = tick_idx * tick_interval
            time_remaining = 1.0 - (elapsed_seconds / candle_duration)

            # Interpolate prices
            price_idx = min(int(tick_idx / num_ticks * len(yes_prices)), len(yes_prices) - 1)
            yes_price = yes_prices[price_idx]
            no_price = no_prices[min(price_idx, len(no_prices) - 1)] if no_prices else 1.0 - yes_price

            # Update state
            state.last_yes_price = yes_price
            state.last_no_price = no_price
            state.yes_price_history.append(yes_price)
            state.no_price_history.append(no_price)

            # Get BTC price
            tick_time = candle_start + timedelta(seconds=elapsed_seconds)
            btc_price = self.get_btc_price_at(tick_time)
            if btc_price:
                state.btc_current_price = btc_price
                state.btc_price_history.append(btc_price)

            # Update position tracking
            if state.position_side:
                state.ticks_held += 1
                current_price = state.last_yes_price if state.position_side == "yes" else state.last_no_price
                current_pnl = (current_price - state.entry_price) / (state.entry_price + 1e-8)
                state.max_pnl_seen = max(state.max_pnl_seen, current_pnl)

            # Get model action
            model_output = self._get_model_action(state, time_remaining)
            action = model_output["action"]
            confidence = model_output["confidence"]
            expected_return = model_output["expected_return"]

            # Log tick
            self._log_tick(state, model_output, time_remaining)

            # Execute trading logic
            if action == Action.BUY_YES and state.position_side is None:
                if (confidence >= self.config.min_confidence and
                    expected_return >= self.config.min_expected_return and
                    time_remaining >= self.config.min_time_remaining):

                    size = self._calculate_position_size(expected_return, confidence)
                    state.position_side = "yes"
                    state.position_size = size
                    state.entry_price = yes_price
                    state.ticks_held = 0
                    state.max_pnl_seen = 0.0

                    self._log_trade("entry", {
                        "candle_ts": candle_ts,
                        "side": "yes",
                        "price": yes_price,
                        "size": size,
                        "confidence": confidence,
                        "expected_return": expected_return,
                        "time_remaining": time_remaining,
                    })

                    if self.config.verbose:
                        console.print(f"  [green]BUY YES @ {yes_price:.3f}[/green]")

            elif action == Action.BUY_NO and state.position_side is None:
                if (confidence >= self.config.min_confidence and
                    expected_return >= self.config.min_expected_return and
                    time_remaining >= self.config.min_time_remaining):

                    size = self._calculate_position_size(expected_return, confidence)
                    state.position_side = "no"
                    state.position_size = size
                    state.entry_price = no_price
                    state.ticks_held = 0
                    state.max_pnl_seen = 0.0

                    self._log_trade("entry", {
                        "candle_ts": candle_ts,
                        "side": "no",
                        "price": no_price,
                        "size": size,
                        "confidence": confidence,
                        "expected_return": expected_return,
                        "time_remaining": time_remaining,
                    })

                    if self.config.verbose:
                        console.print(f"  [red]BUY NO @ {no_price:.3f}[/red]")

            elif action == Action.EXIT and state.position_side is not None:
                # Exit before settlement
                current_price = state.last_yes_price if state.position_side == "yes" else state.last_no_price
                trade_result = self._execute_exit(state, current_price, "model_signal")

            # Force exit near settlement
            if state.position_side and time_remaining < 0.02:
                current_price = state.last_yes_price if state.position_side == "yes" else state.last_no_price
                trade_result = self._execute_exit(state, current_price, "time_expiry")

        # Settlement
        if state.position_side:
            trade_result = self._settle_position(state, candle)

        return {"skipped": False, "trade": trade_result}

    def _get_model_action(self, state: UnifiedTradingState, time_remaining: float) -> Dict[str, Any]:
        """Get action from model."""
        if self.model is None:
            return {"action": Action.WAIT, "q_values": [0]*5, "confidence": 0, "expected_return": 0}

        yes_prices = np.array(state.yes_price_history[-300:] or [0.5])
        no_prices = np.array(state.no_price_history[-300:] or [0.5])
        btc_prices = np.array(state.btc_price_history[-300:]) if state.btc_price_history else None

        features = self.feature_builder.compute_features(
            yes_prices=yes_prices,
            no_prices=no_prices,
            time_remaining=time_remaining,
            btc_prices=btc_prices,
            btc_open=state.btc_open_price,
        )

        if state.position_side:
            current_price = state.last_yes_price if state.position_side == "yes" else state.last_no_price
        else:
            current_price = state.last_yes_price

        position_state = EnhancedPositionState.compute(
            has_position=state.position_side is not None,
            position_side=state.position_side,
            entry_price=state.entry_price,
            current_price=current_price,
            time_remaining=time_remaining,
            ticks_held=state.ticks_held,
            max_pnl_seen=state.max_pnl_seen,
        )

        features_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        position_t = torch.FloatTensor(position_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model.get_action(features_t, position_t, deterministic=True)

        return {
            "action": int(result["action"].item()),
            "q_values": result["q_values"].squeeze(0).cpu().numpy().tolist(),
            "confidence": float(result["confidence"].item()),
            "expected_return": float(result["expected_return"].item()),
        }

    def _calculate_position_size(self, expected_return: float, confidence: float) -> float:
        """Calculate position size."""
        return_factor = np.clip(expected_return / 0.10, 0.5, 2.0)
        conf_factor = 0.5 + 0.5 * confidence
        size = self.config.base_position_size * return_factor * conf_factor
        return float(np.clip(size, self.config.min_position_size, self.config.max_position_size))

    def _execute_exit(self, state: UnifiedTradingState, exit_price: float, reason: str) -> Dict[str, Any]:
        """Execute exit and return trade result."""
        invested = state.position_size * state.balance
        shares = invested / state.entry_price
        exit_value = shares * exit_price
        pnl = exit_value - invested

        trade = {
            "candle_ts": state.current_candle_ts,
            "side": state.position_side,
            "entry_price": state.entry_price,
            "exit_price": exit_price,
            "size": state.position_size,
            "pnl": pnl,
            "pnl_pct": pnl / invested if invested > 0 else 0,
            "reason": reason,
            "ticks_held": state.ticks_held,
        }

        state.balance += pnl
        state.total_pnl += pnl
        if pnl > 0:
            state.wins += 1
        else:
            state.losses += 1

        # Clear position
        state.position_side = None
        state.position_size = 0.0
        state.entry_price = 0.0
        state.ticks_held = 0
        state.max_pnl_seen = 0.0

        self.trades.append(trade)
        self._log_trade("exit", trade)
        return trade

    def _settle_position(self, state: UnifiedTradingState, candle: pd.Series) -> Dict[str, Any]:
        """Settle position at candle end."""
        outcome = int(candle.get('outcome', -1))
        closed = candle.get('closed', True)

        # Determine payout
        if state.position_side == "yes":
            payout = 1.0 if outcome == 1 else 0.0
        else:
            payout = 1.0 if outcome == 0 else 0.0

        invested = state.position_size * state.balance
        shares = invested / state.entry_price
        returned = shares * payout
        pnl = returned - invested

        trade = {
            "candle_ts": state.current_candle_ts,
            "side": state.position_side,
            "entry_price": state.entry_price,
            "exit_price": payout,
            "size": state.position_size,
            "pnl": pnl,
            "pnl_pct": pnl / invested if invested > 0 else 0,
            "reason": "settlement",
            "outcome": outcome,
            "ticks_held": state.ticks_held,
        }

        state.balance += pnl
        state.total_pnl += pnl
        if pnl > 0:
            state.wins += 1
        else:
            state.losses += 1

        # Clear position
        state.position_side = None
        state.position_side = None
        state.position_size = 0.0
        state.entry_price = 0.0
        state.ticks_held = 0
        state.max_pnl_seen = 0.0

        self.trades.append(trade)
        self._log_trade("settlement", trade)
        return trade

    def run(self) -> Dict[str, Any]:
        """Run the backtest."""
        self.load_model()

        if self.config.data_source == "onchain":
            return self.run_onchain_backtest()
        else:
            return self.run_historical_backtest()

    def run_onchain_backtest(self) -> Dict[str, Any]:
        """Run backtest using on-chain data from local node."""
        self.load_onchain_data()
        self._setup_logging()

        state = UnifiedTradingState(balance=self.config.initial_balance)
        self.equity_curve = [state.balance]

        # Merge YES and NO prices into chronological order
        yes_prices = self.onchain_data.get('yes_prices', [])
        no_prices = self.onchain_data.get('no_prices', [])

        # Create unified price timeline
        all_events = []
        for p in yes_prices:
            if p.get('timestamp') and p.get('price'):
                all_events.append({
                    'timestamp': p['timestamp'],
                    'yes_price': p['price'],
                    'no_price': None,
                    'block': p.get('block_number'),
                })
        for p in no_prices:
            if p.get('timestamp') and p.get('price'):
                all_events.append({
                    'timestamp': p['timestamp'],
                    'yes_price': None,
                    'no_price': p['price'],
                    'block': p.get('block_number'),
                })

        # Sort by timestamp
        all_events.sort(key=lambda x: (x['timestamp'], x.get('block', 0)))

        if not all_events:
            console.print("[red]No price events found in on-chain data![/red]")
            return {}

        # Get candle info
        candle_ts = self.onchain_data.get('candle_ts', 0)
        candle_start = candle_ts
        candle_end = candle_ts + 900  # 15 minutes

        console.print(f"\n[bold cyan]Running on-chain backtest...[/bold cyan]")
        console.print(f"  Candle: {self.onchain_data.get('slug', 'unknown')}")
        console.print(f"  Price events: {len(all_events)}")
        console.print(f"  Time range: {all_events[0]['timestamp']} to {all_events[-1]['timestamp']}")

        # Initialize state
        state.current_candle_ts = candle_ts
        state.yes_price_history = []
        state.no_price_history = []
        state.btc_price_history = []

        # Track last known prices
        last_yes = 0.5
        last_no = 0.5

        # Sample every N events to reduce computation
        sample_interval = max(1, len(all_events) // 500)  # ~500 ticks max

        console.print(f"  Sampling every {sample_interval} events ({len(all_events) // sample_interval} ticks)\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Backtesting...", total=len(all_events) // sample_interval)

            for i, event in enumerate(all_events):
                # Update prices
                if event['yes_price'] is not None:
                    last_yes = event['yes_price']
                if event['no_price'] is not None:
                    last_no = event['no_price']

                # Only process at sample intervals
                if i % sample_interval != 0:
                    continue

                # Calculate time remaining
                event_ts = event['timestamp']
                if candle_start <= event_ts <= candle_end:
                    time_remaining = max(0, (candle_end - event_ts) / 900)
                else:
                    time_remaining = 1.0 if event_ts < candle_start else 0.0

                # Update state
                state.last_yes_price = last_yes
                state.last_no_price = last_no
                state.yes_price_history.append(last_yes)
                state.no_price_history.append(last_no)

                # Get BTC price if available
                if self.btc_df is not None:
                    btc_price = self.get_btc_price_at(datetime.fromtimestamp(event_ts, tz=timezone.utc))
                    if btc_price:
                        state.btc_current_price = btc_price
                        state.btc_price_history.append(btc_price)
                        if state.btc_open_price is None:
                            state.btc_open_price = btc_price

                # Update position tracking
                if state.position_side:
                    state.ticks_held += 1
                    current_price = last_yes if state.position_side == "yes" else last_no
                    current_pnl = (current_price - state.entry_price) / (state.entry_price + 1e-8)
                    state.max_pnl_seen = max(state.max_pnl_seen, current_pnl)

                # Get model action
                model_output = self._get_model_action(state, time_remaining)
                action = model_output["action"]
                confidence = model_output["confidence"]
                expected_return = model_output["expected_return"]

                # Log tick
                self._log_tick(state, model_output, time_remaining)

                # Execute trading logic
                self._execute_onchain_trading(state, model_output, time_remaining, last_yes, last_no)

                self.equity_curve.append(state.balance)
                progress.update(task, advance=1)

        # Close logging
        self._close_logging(state)

        # Calculate metrics
        results = self._calculate_metrics(state)
        self._print_results(results)
        self._save_results(results)

        return results

    def _execute_onchain_trading(
        self,
        state: UnifiedTradingState,
        model_output: Dict,
        time_remaining: float,
        yes_price: float,
        no_price: float,
    ):
        """Execute trading logic for on-chain backtest."""
        action = model_output["action"]
        confidence = model_output["confidence"]
        expected_return = model_output["expected_return"]

        if action == Action.BUY_YES and state.position_side is None:
            if (confidence >= self.config.min_confidence and
                expected_return >= self.config.min_expected_return and
                time_remaining >= self.config.min_time_remaining):

                size = self._calculate_position_size(expected_return, confidence)
                state.position_side = "yes"
                state.position_size = size
                state.entry_price = yes_price
                state.ticks_held = 0
                state.max_pnl_seen = 0.0

                self._log_trade("entry", {
                    "side": "yes",
                    "price": yes_price,
                    "size": size,
                    "confidence": confidence,
                    "expected_return": expected_return,
                    "time_remaining": time_remaining,
                })

                if self.config.verbose:
                    console.print(f"  [green]BUY YES @ {yes_price:.3f} conf={confidence:.2f}[/green]")

        elif action == Action.BUY_NO and state.position_side is None:
            if (confidence >= self.config.min_confidence and
                expected_return >= self.config.min_expected_return and
                time_remaining >= self.config.min_time_remaining):

                size = self._calculate_position_size(expected_return, confidence)
                state.position_side = "no"
                state.position_size = size
                state.entry_price = no_price
                state.ticks_held = 0
                state.max_pnl_seen = 0.0

                self._log_trade("entry", {
                    "side": "no",
                    "price": no_price,
                    "size": size,
                    "confidence": confidence,
                    "expected_return": expected_return,
                    "time_remaining": time_remaining,
                })

                if self.config.verbose:
                    console.print(f"  [red]BUY NO @ {no_price:.3f} conf={confidence:.2f}[/red]")

        elif action == Action.EXIT and state.position_side is not None:
            exit_price = yes_price if state.position_side == "yes" else no_price
            self._execute_exit(state, exit_price, "model_signal")

        # Force exit near end
        if state.position_side and time_remaining < 0.02:
            exit_price = yes_price if state.position_side == "yes" else no_price
            self._execute_exit(state, exit_price, "time_expiry")

    def run_historical_backtest(self) -> Dict[str, Any]:
        """Run backtest using historical parquet data."""
        self.load_data()
        self._setup_logging()

        state = UnifiedTradingState(balance=self.config.initial_balance)
        self.equity_curve = [state.balance]

        # Filter to closed candles with valid outcomes
        valid_candles = self.candles_df[
            (self.candles_df['closed'] == True) &
            (self.candles_df['outcome'].isin([0, 1]))
        ].copy()
        valid_candles = valid_candles.sort_values('timestamp')

        # Limit candles if specified
        if self.config.max_candles > 0:
            valid_candles = valid_candles.head(self.config.max_candles)

        console.print(f"\n[bold cyan]Running backtest on {len(valid_candles)} candles...[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Backtesting...", total=len(valid_candles))

            for idx, (_, candle) in enumerate(valid_candles.iterrows()):
                result = self.simulate_candle(candle, self.prices_df, state)
                self.equity_curve.append(state.balance)
                progress.update(task, advance=1)

        # Close logging
        self._close_logging(state)

        # Calculate metrics
        results = self._calculate_metrics(state)
        self._print_results(results)
        self._save_results(results)

        return results

    def _calculate_metrics(self, state: UnifiedTradingState) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Sharpe-like ratio (using trade returns)
        if total_trades > 1:
            returns = [t['pnl_pct'] for t in self.trades]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(total_trades)
        else:
            sharpe = 0

        # By exit reason
        by_reason = {}
        for t in self.trades:
            reason = t['reason']
            if reason not in by_reason:
                by_reason[reason] = {"count": 0, "pnl": 0, "wins": 0}
            by_reason[reason]["count"] += 1
            by_reason[reason]["pnl"] += t['pnl']
            if t['pnl'] > 0:
                by_reason[reason]["wins"] += 1

        return {
            "initial_balance": self.config.initial_balance,
            "final_balance": state.balance,
            "total_pnl": state.total_pnl,
            "total_return": (state.balance - self.config.initial_balance) / self.config.initial_balance,
            "total_trades": total_trades,
            "wins": state.wins,
            "losses": state.losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "by_reason": by_reason,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

    def _print_results(self, results: Dict[str, Any]):
        """Print backtest results."""
        console.print("\n")

        # Summary table
        table = Table(title="[bold cyan]Backtest Results[/bold cyan]", show_header=False, box=None)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", style="bold")

        pnl_color = "green" if results["total_pnl"] >= 0 else "red"
        ret_color = "green" if results["total_return"] >= 0 else "red"

        table.add_row("Initial Balance", f"${results['initial_balance']:,.2f}")
        table.add_row("Final Balance", f"${results['final_balance']:,.2f}")
        table.add_row("Total PnL", f"[{pnl_color}]${results['total_pnl']:+,.2f}[/]")
        table.add_row("Total Return", f"[{ret_color}]{results['total_return']:+.2%}[/]")
        table.add_row("", "")
        table.add_row("Total Trades", f"{results['total_trades']}")
        table.add_row("Win Rate", f"{results['win_rate']:.1%} ({results['wins']}/{results['losses']})")
        table.add_row("Avg Win", f"${results['avg_win']:+.2f}")
        table.add_row("Avg Loss", f"${results['avg_loss']:.2f}")
        table.add_row("Profit Factor", f"{results['profit_factor']:.2f}")
        table.add_row("", "")
        table.add_row("Max Drawdown", f"[red]{results['max_drawdown']:.1%}[/]")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

        console.print(table)

        # By reason
        if results["by_reason"]:
            console.print("\n[bold]Performance by Exit Reason:[/bold]")
            reason_table = Table(show_header=True)
            reason_table.add_column("Reason")
            reason_table.add_column("Count", justify="right")
            reason_table.add_column("Win Rate", justify="right")
            reason_table.add_column("PnL", justify="right")

            for reason, stats in results["by_reason"].items():
                wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
                pnl_color = "green" if stats["pnl"] >= 0 else "red"
                reason_table.add_row(
                    reason,
                    str(stats["count"]),
                    f"{wr:.1%}",
                    f"[{pnl_color}]${stats['pnl']:+.2f}[/]"
                )
            console.print(reason_table)

    def _save_results(self, results: Dict[str, Any]):
        """Save backtest results to file."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.config.model_path).name

        # Save summary
        summary_file = log_dir / f"backtest_{model_name}_{timestamp}.json"
        summary = {k: v for k, v in results.items() if k not in ['trades', 'equity_curve']}
        summary['config'] = {
            'model_path': self.config.model_path,
            'initial_balance': self.config.initial_balance,
            'min_confidence': self.config.min_confidence,
            'min_expected_return': self.config.min_expected_return,
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save trades
        trades_file = log_dir / f"backtest_trades_{model_name}_{timestamp}.json"
        with open(trades_file, 'w') as f:
            json.dump(results['trades'], f, indent=2, default=str)

        console.print(f"\n[green]Results saved to {log_dir}[/green]")


async def main():
    parser = argparse.ArgumentParser(description="Unified paper trading and backtesting")
    parser.add_argument("--model", type=str, default="./logs/market_predictor_v1")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-return", type=float, default=0.02)
    parser.add_argument("--log-dir", type=str, default="./logs/paper_trade_unified")
    parser.add_argument("--no-ml-log", action="store_true")

    # Backtest mode
    parser.add_argument("--backtest", action="store_true", help="Run backtest instead of live trading")
    parser.add_argument("--data-source", type=str, default="historical", choices=["historical", "onchain"],
                        help="Data source: 'historical' (API parquet) or 'onchain' (local node JSON)")
    parser.add_argument("--onchain-data", type=str, default="./data/onchain/btc_15min_current.json",
                        help="Path to on-chain data JSON file")
    parser.add_argument("--candles", type=str, default="./data/historical/polymarket_candles_30d.parquet",
                        help="Path to candles data for backtest")
    parser.add_argument("--prices", type=str, default="./data/historical/polymarket_prices_30d.parquet",
                        help="Path to prices data for backtest")
    parser.add_argument("--btc-data", type=str, default="./data/historical/btc_1m_30d.parquet",
                        help="Path to BTC data for backtest")
    parser.add_argument("--tick-interval", type=int, default=30,
                        help="Tick interval in seconds for backtest simulation")
    parser.add_argument("--max-candles", type=int, default=0,
                        help="Max candles to test (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Verbose backtest output")

    args = parser.parse_args()

    if args.backtest:
        # Run backtest
        config = BacktestConfig(
            model_path=args.model,
            initial_balance=args.balance,
            min_confidence=args.min_confidence,
            min_expected_return=args.min_return,
            data_source=args.data_source,
            onchain_data_path=args.onchain_data,
            candles_path=args.candles,
            prices_path=args.prices,
            btc_path=args.btc_data,
            tick_interval_seconds=args.tick_interval,
            max_candles=args.max_candles,
            log_dir=args.log_dir.replace("paper_trade", "backtest"),
            verbose=args.verbose,
        )

        runner = BacktestRunner(config)
        runner.run()
    else:
        # Run live paper trading
        config = UnifiedPaperTradeConfig(
            model_path=args.model,
            initial_balance=args.balance,
            min_confidence=args.min_confidence,
            min_expected_return=args.min_return,
            log_dir=args.log_dir,
            enable_ml_logging=not args.no_ml_log,
        )

        trader = UnifiedPaperTrader(config)
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
