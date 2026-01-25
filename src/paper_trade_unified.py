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
import structlog
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

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
    last_q_values: List[float] = field(default_factory=lambda: [0.0] * 5)
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


async def main():
    parser = argparse.ArgumentParser(description="Unified paper trading")
    parser.add_argument("--model", type=str, default="./logs/market_predictor_v1")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-return", type=float, default=0.02)
    parser.add_argument("--log-dir", type=str, default="./logs/paper_trade_unified")
    parser.add_argument("--no-ml-log", action="store_true")

    args = parser.parse_args()

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
