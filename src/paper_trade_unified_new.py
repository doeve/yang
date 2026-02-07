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
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt

from src.data.enhanced_features import EnhancedFeatureBuilder
from src.models.market_predictor import (
    MarketPredictorModel,
    EnhancedPositionState,
    Action,
    load_market_predictor,
)
from src.config import load_config, TradingConfig
from src.execution import LiveExecutor, Redeemer, OnchainExecutor, OnchainOrderExecutor
from src.risk import DailyLossTracker

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
    position_size: float = 0.0  # Position as % of balance (for paper trading)
    position_shares: float = 0.0  # Actual shares owned (for live trading)
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

    # Live trading state
    wallet_balance: float = 0.0  # Real USDC balance (live mode)
    daily_pnl: float = 0.0  # Today's realized PnL
    condition_id: Optional[str] = None  # Current market's condition ID


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

    # Live trading config
    trading_mode: str = "paper"  # "paper" or "live"
    max_daily_loss_pct: float = 5.0
    max_position_size_usdc: float = 100.0


class UnifiedPaperTrader:
    """Paper trader using unified MarketPredictor model."""

    def __init__(self, config: UnifiedPaperTradeConfig, trading_config: Optional[TradingConfig] = None):
        self.config = config
        self.trading_config = trading_config
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
        self._mode_switch_requested = False
        self._reset_daily_limit_requested = False
        self._config_editor_open = False
        self._live_display = None

        # Logging
        self.log_file: Optional[Path] = None

        # Live trading components (onchain execution to avoid fees)
        self.executor: Optional[OnchainOrderExecutor] = None
        self.onchain_executor: Optional[OnchainExecutor] = None  # For redemption
        self.loss_tracker = DailyLossTracker(starting_balance=config.initial_balance)

    @property
    def is_live_mode(self) -> bool:
        """Check if running in live trading mode."""
        return self.config.trading_mode == "live"

    def _setup_logging(self):
        """Initialize structured logging to file."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.config.model_path).name
        mode = "live" if self.is_live_mode else "paper"
        self.log_file = log_dir / f"{model_name}_{mode}_{timestamp}.jsonl"

        # Configure structlog to write JSON to file (no console output)
        import structlog as struct_module
        import logging

        # Setup standard logging to file only
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = []  # Clear any existing handlers

        file_handler = logging.FileHandler(str(self.log_file))
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(file_handler)

        # Configure structlog to write JSON only to file
        struct_module.configure(
            processors=[
                struct_module.processors.TimeStamper(fmt="iso"),
                struct_module.processors.add_log_level,
                struct_module.processors.StackInfoRenderer(),
                struct_module.processors.format_exc_info,
                struct_module.processors.JSONRenderer()
            ],
            logger_factory=struct_module.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        console.print(f"[blue]Logging to: {self.log_file}[/blue]")

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
        """Setup HTTP clients and live trading components."""
        self.binance_client = httpx.AsyncClient(timeout=10)

        if SOCKS_AVAILABLE:
            transport = httpx_socks.AsyncProxyTransport.from_url(SOCKS5_PROXY_URL)
            self.polymarket_client = httpx.AsyncClient(transport=transport, timeout=30, verify=False)
        else:
            console.print("[yellow]SOCKS5 not available, using direct connection[/yellow]")
            self.polymarket_client = httpx.AsyncClient(timeout=30)

        # Initialize live trading components if in live mode
        if self.is_live_mode and self.trading_config:
            logger.info(
                "initializing_live_mode",
                mode="LIVE",
                polygon_rpc=self.trading_config.polygon_rpc_url,
                public_rpc=self.trading_config.public_rpc_url,
                use_public_rpc=self.trading_config.execution.use_public_rpc_for_redeem,
                has_socks5=bool(self.trading_config.socks5_proxy)
            )
            console.print("[bold red]🔴 LIVE TRADING MODE (ONCHAIN - NO POLYMARKET FEES)[/bold red]")

            # Initialize onchain order executor (for fee-free trading)
            try:
                self.executor = OnchainOrderExecutor(
                    local_rpc_url=self.trading_config.polygon_rpc_url,
                    private_key=self.trading_config.eth_private_key,
                    public_rpc_url=self.trading_config.public_rpc_url,
                    use_public_rpc=self.trading_config.execution.use_public_rpc_for_redeem,
                    use_clob=self.trading_config.execution.use_clob,
                    socks5_proxy=self.trading_config.socks5_proxy,
                )

                logger.info("connecting_order_executor", mode="LIVE")
                if await self.executor.connect():
                    logger.info("order_executor_connected", mode="LIVE")

                    # Ensure approvals for trading
                    logger.info("ensuring_token_approvals", mode="LIVE")
                    await self.executor.ensure_approvals()

                    # Update wallet balance
                    self.state.wallet_balance = await self.executor.get_usdc_balance()
                    logger.info(
                        "live_mode_initialized",
                        mode="LIVE",
                        wallet_balance=self.state.wallet_balance,
                        executor_type="OnchainOrderExecutor"
                    )
                    console.print(f"  [green]Wallet connected: ${self.state.wallet_balance:.2f} USDC[/green]")
                    console.print(f"  [cyan]Using onchain execution (bypassing CLOB fees)[/cyan]")
                else:
                    logger.error(
                        "order_executor_connection_failed",
                        mode="LIVE",
                        reason="Failed to connect to blockchain"
                    )
                    console.print("  [red]Failed to connect executor, falling back to paper mode[/red]")
                    self.config.trading_mode = "paper"

                # Initialize onchain executor for redemption
                logger.info("connecting_redemption_executor", mode="LIVE")
                self.onchain_executor = OnchainExecutor(
                    local_rpc_url=self.trading_config.polygon_rpc_url,
                    private_key=self.trading_config.eth_private_key,
                    public_rpc_url=self.trading_config.public_rpc_url,
                    use_public_rpc=self.trading_config.execution.use_public_rpc_for_redeem,
                    use_clob=self.trading_config.execution.use_clob,
                    socks5_proxy=self.trading_config.socks5_proxy,
                )
                await self.onchain_executor.connect()
                logger.info("redemption_executor_connected", mode="LIVE")
                console.print(f"  [cyan]Auto-redemption enabled[/cyan]")

            except Exception as e:
                logger.error(
                    "live_mode_initialization_error",
                    mode="LIVE",
                    error=str(e),
                    error_type=type(e).__name__,
                    reason="Failed to initialize live trading components"
                )
                import traceback
                logger.error(
                    "live_mode_init_traceback",
                    traceback=traceback.format_exc()
                )
                console.print(f"  [red]Failed to initialize live mode: {e}[/red]")
                self.config.trading_mode = "paper"
        else:
            logger.info("initializing_paper_mode", mode="PAPER")
            console.print("[green]📝 PAPER TRADING MODE[/green]")

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
                        self.state.condition_id = market.get("conditionId")  # Store for redemption
                        console.print(f"[blue]Found market: {slug}[/blue]")
                        if self.state.condition_id:
                            console.print(f"  [dim]Condition ID: {self.state.condition_id[:16]}...[/dim]")
                        return True
        except Exception as e:
            logger.error(
                "market_discovery_error",
                timestamp=timestamp,
                error=str(e),
                error_type=type(e).__name__,
                reason="Failed to discover active market"
            )
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
            logger.error(
                "polymarket_price_fetch_error",
                yes_id=self.state.active_yes_id,
                no_id=self.state.active_no_id,
                error=str(e),
                error_type=type(e).__name__,
                reason="Failed to fetch Polymarket prices"
            )
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
            logger.error(
                "binance_price_fetch_error",
                error=str(e),
                error_type=type(e).__name__,
                reason="Failed to fetch BTC price from Binance"
            )
        return None

    async def fetch_btc_candle_open(self) -> Optional[float]:
        """Fetch the open price of the current 15m candle."""
        try:
            # Calculate start of current 15m candle
            now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
            candle_start = (now_ts // 900000) * 900000
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": "BTCUSDT",
                "interval": "15m",
                "limit": 1,
                "startTime": candle_start
            }
            response = await self.binance_client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Index 1 is Open price
                    # [Open Time, Open, High, Low, Close, Volume, Close Time, ...]
                    return float(data[0][1])
        except Exception as e:
            logger.error(f"Binance candle fetch error: {e}")
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

        # Old ML logging removed - using structured logs instead
        # self._log_tick(features, position_state, {
        #     "action": action,
        #     "q_values": q_values,
        #     "confidence": confidence,
        #     "expected_return": expected_return,
        # })

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

        # Risk check: pause trading if daily loss limit hit
        mode_str = "live" if self.is_live_mode else "paper"
        if self.loss_tracker.is_limit_hit(self.config.max_daily_loss_pct, mode=mode_str):
            if action in [Action.BUY_YES, Action.BUY_NO]:
                mode = "LIVE" if self.is_live_mode else "PAPER"
                logger.warning(
                    "loss_limit_hit_blocking_entry",
                    mode=mode,
                    action=Action(action).name,
                    daily_pnl=self.loss_tracker.get_daily_pnl(mode=mode_str),
                    daily_pnl_pct=self.loss_tracker.get_daily_pnl_pct(mode=mode_str),
                    max_loss_pct=self.config.max_daily_loss_pct,
                    reason="24H loss limit exceeded, blocking new positions"
                )
                # Don't open new positions, but allow exits
                return

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
            await self.execute_entry("yes", size, model_output)

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
            await self.execute_entry("no", size, model_output)

        elif action == Action.EXIT:
            if self.state.position_side is None:
                return  # No position to exit
            await self.execute_exit("model_signal")

        elif action == Action.HOLD:
            # Model says hold current position - do nothing
            pass

        # Force exit near settlement (safety net)
        if self.state.position_side and time_remaining < 0.02:
            await self.execute_exit("time_expiry")

    async def execute_entry(self, side: str, position_size: float, model_output: Dict):
        """Execute position entry."""
        mode = "LIVE" if self.is_live_mode else "PAPER"

        # Check if already in a position
        if self.state.position_side is not None:
            logger.warning(
                "entry_blocked_existing_position",
                mode=mode,
                existing_side=self.state.position_side,
                requested_side=side,
                reason="Cannot enter new position while already holding a position"
            )
            console.print(f"[yellow]Skipping entry: Already in {self.state.position_side.upper()} position[/yellow]")
            return

        logger.info(
            f"execute_entry",
            mode=mode,
            side=side,
            position_size=position_size,
            balance=self.state.balance,
            expected_return=model_output.get('expected_return'),
            confidence=model_output.get('confidence')
        )

        price = self.state.last_yes_price if side == "yes" else self.state.last_no_price

        # Use wallet balance in live mode, paper balance otherwise
        balance_to_use = self.state.wallet_balance if self.is_live_mode else self.state.balance
        dollar_size = position_size * balance_to_use

        # Enforce max position size limit
        if dollar_size > self.config.max_position_size_usdc:
            logger.warning(
                "position_size_capped",
                mode=mode,
                calculated_size=dollar_size,
                max_allowed=self.config.max_position_size_usdc,
                message=f"Position size ${dollar_size:.2f} exceeds max ${self.config.max_position_size_usdc:.2f}, capping"
            )
            dollar_size = self.config.max_position_size_usdc

        # Calculate quantity (shares) = investment / price
        if price <= 0:
            logger.error(
                "execute_entry_failed_zero_price",
                mode=mode,
                side=side,
                price=price,
                reason="Price is zero or negative"
            )
            return
        quantity = dollar_size / price

        # Live Execution
        if self.is_live_mode and self.executor:
            # Determine token ID
            token_id = self.state.active_yes_id if side == "yes" else self.state.active_no_id
            if not token_id:
                logger.error(
                    "execute_entry_failed_no_token",
                    mode=mode,
                    side=side,
                    yes_id=self.state.active_yes_id,
                    no_id=self.state.active_no_id,
                    reason="No active token ID available"
                )
                return

            logger.info(
                "live_order_placing",
                mode="LIVE",
                side=side,
                token_id=token_id,
                quantity=quantity,
                price=price,
                dollar_size=dollar_size
            )
            console.print(f"[bold yellow]Executing LIVE BUY {side.upper()}: ${dollar_size:.2f} ({quantity:.1f} shares) @ {price:.3f}[/bold yellow]")

            # Place order
            try:
                result = await self.executor.place_order(
                    token_id=token_id,
                    side="BUY",
                    size=quantity,
                    price=price
                )

                if not result.success:
                    logger.error(
                        "live_order_failed",
                        mode="LIVE",
                        side=side,
                        token_id=token_id,
                        error=result.error,
                        reason="Order placement failed"
                    )
                    console.print(f"[bold red]Order Execution Failed: {result.error}[/bold red]")
                    return

                logger.info(
                    "live_order_placed",
                    mode="LIVE",
                    side=side,
                    order_id=result.order_id,
                    token_id=token_id
                )

                # Wait for fill (required for live mode to track position accurately)
                if result.order_id:
                    filled = await self.executor.wait_for_fill(result.order_id)
                    if not filled:
                        # Timeout - but order might have filled anyway
                        # Check actual token balance to see if we got the tokens
                        logger.warning(
                            "live_order_timeout",
                            mode="LIVE",
                            side=side,
                            order_id=result.order_id,
                            reason="Order not confirmed within timeout - checking actual balance"
                        )
                        console.print("[yellow]Order timeout - checking if filled...[/yellow]")

                        # Check if we actually received the tokens
                        try:
                            actual_balance = await self.executor._get_token_balance(token_id)
                            balance_shares = actual_balance / 1e6

                            if balance_shares >= (quantity * 0.95):  # Allow 5% tolerance
                                logger.info(
                                    "live_order_filled_after_timeout",
                                    mode="LIVE",
                                    side=side,
                                    order_id=result.order_id,
                                    balance_shares=balance_shares,
                                    expected_shares=quantity,
                                    reason="Order filled after timeout (confirmed by balance check)"
                                )
                                console.print(f"[green]✓ Order filled ({balance_shares:.2f} shares received)[/green]")
                                # Update filled amount to actual received
                                result.filled_amount = balance_shares
                            else:
                                logger.error(
                                    "live_order_not_filled",
                                    mode="LIVE",
                                    side=side,
                                    order_id=result.order_id,
                                    balance_shares=balance_shares,
                                    expected_shares=quantity,
                                    reason="Order not filled - balance check confirms no tokens received"
                                )
                                console.print(f"[red]Order not filled - position NOT entered[/red]")
                                return
                        except Exception as balance_error:
                            logger.error(f"Balance check failed: {balance_error}")
                            console.print("[red]Could not verify fill - position NOT entered[/red]")
                            return
                    else:
                        logger.info(
                            "live_order_filled",
                            mode="LIVE",
                            side=side,
                            order_id=result.order_id
                        )

                # Update wallet balance
                old_balance = self.state.wallet_balance
                self.state.wallet_balance = await self.executor.get_usdc_balance()
                logger.info(
                    "wallet_balance_updated",
                    mode="LIVE",
                    old_balance=old_balance,
                    new_balance=self.state.wallet_balance,
                    change=self.state.wallet_balance - old_balance
                )

            except Exception as e:
                logger.error(
                    "live_order_exception",
                    mode="LIVE",
                    side=side,
                    token_id=token_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    reason="Exception during order execution"
                )
                console.print(f"[bold red]Order Exception: {e}[/bold red]")
                return

        self.state.position_side = side
        self.state.position_size = position_size
        self.state.entry_price = price
        self.state.entry_tick = self._tick_count
        self.state.ticks_held = 0
        self.state.max_pnl_seen = 0.0

        # Store actual shares for live mode (use filled_amount from order result)
        if self.is_live_mode and self.executor:
            # Use the actual filled amount from the order result
            self.state.position_shares = result.filled_amount if result and result.success else quantity
            logger.info(
                "position_shares_stored",
                mode="LIVE",
                shares=self.state.position_shares,
                side=side
            )
        else:
            # Paper mode: calculate shares from position size
            self.state.position_shares = quantity

        # Use correct balance for display (already calculated above)
        dollar_size = position_size * balance_to_use

        console.print(
            f"[green]ENTRY: {side.upper()} @ {price:.3f} | "
            f"Size={position_size:.1%} (${dollar_size:.0f}) | "
            f"E[R]={model_output['expected_return']:+.1%} | "
            f"Conf={model_output['confidence']:.1%}[/green]"
        )

        # Old ML logging removed - using structured logs instead
        # self._log_trade("entry", {
        #     "side": side,
        #     "price": price,
        #     "size": position_size,
        #     "dollar_size": dollar_size,
        #     "expected_return": model_output["expected_return"],
        #     "confidence": model_output["confidence"],
        #     "q_values": model_output["q_values"],
        #     "time_remaining": self.get_time_remaining(),
        # })

    async def execute_exit(self, reason: str = "manual"):
        """Execute position exit."""
        mode = "LIVE" if self.is_live_mode else "PAPER"

        if self.state.position_side is None:
            logger.warning(
                "execute_exit_no_position",
                mode=mode,
                reason=reason,
                message="Exit called but no position open"
            )
            return

        side = self.state.position_side
        entry_price = self.state.entry_price
        current_price = (
            self.state.last_yes_price if side == "yes"
            else self.state.last_no_price
        )

        # In live mode, use stored actual shares (don't recalculate from balance!)
        # In paper mode, calculate from position size
        if self.is_live_mode:
            shares = self.state.position_shares
            invested = shares * entry_price
        else:
            invested = self.state.position_size * self.state.balance
            shares = invested / entry_price if entry_price > 0 else 0

        logger.info(
            "execute_exit",
            mode=mode,
            side=side,
            reason=reason,
            entry_price=entry_price,
            current_price=current_price,
            shares=shares,
            invested=invested,
            ticks_held=self.state.ticks_held
        )

        # Live Execution
        if self.is_live_mode and self.executor:
            # Determine token ID
            token_id = self.state.active_yes_id if side == "yes" else self.state.active_no_id
            if token_id:
                logger.info(
                    "live_exit_placing",
                    mode="LIVE",
                    side=side,
                    token_id=token_id,
                    shares=shares,
                    price=current_price,
                    reason=reason
                )
                console.print(f"[bold yellow]Executing LIVE SELL {side.upper()}: {shares:.1f} shares @ {current_price:.3f}[/bold yellow]")

                try:
                    # Place order
                    result = await self.executor.place_order(
                        token_id=token_id,
                        side="SELL",
                        size=shares,
                        price=current_price
                    )

                    if not result.success:
                        logger.error(
                            "live_exit_failed",
                            mode="LIVE",
                            side=side,
                            token_id=token_id,
                            error=result.error,
                            reason=reason,
                            message="Exit order failed, position still open"
                        )
                        console.print(f"[bold red]Exit Order Failed: {result.error}[/bold red]")
                        # For exit, we might want to retry or mark as stuck?
                        # For now, we return and don't clear the position so we try again next tick
                        return

                    logger.info(
                        "live_exit_placed",
                        mode="LIVE",
                        side=side,
                        order_id=result.order_id,
                        token_id=token_id
                    )

                    # Wait for fill
                    if result.order_id:
                        filled = await self.executor.wait_for_fill(result.order_id)
                        if not filled:
                            logger.warning(
                                "live_exit_not_filled",
                                mode="LIVE",
                                side=side,
                                order_id=result.order_id,
                                reason="Exit order not filled within timeout"
                            )
                        else:
                            logger.info(
                                "live_exit_filled",
                                mode="LIVE",
                                side=side,
                                order_id=result.order_id
                            )

                    # Update wallet balance
                    old_balance = self.state.wallet_balance
                    self.state.wallet_balance = await self.executor.get_usdc_balance()
                    logger.info(
                        "wallet_balance_updated_exit",
                        mode="LIVE",
                        old_balance=old_balance,
                        new_balance=self.state.wallet_balance,
                        change=self.state.wallet_balance - old_balance
                    )

                except Exception as e:
                    logger.error(
                        "live_exit_exception",
                        mode="LIVE",
                        side=side,
                        token_id=token_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        reason="Exception during exit execution",
                        position_still_open=True
                    )
                    console.print(f"[bold red]Exit Exception: {e}[/bold red]")
                    return
            else:
                logger.error(
                    "live_exit_no_token",
                    mode="LIVE",
                    side=side,
                    yes_id=self.state.active_yes_id,
                    no_id=self.state.active_no_id,
                    reason="No token ID available for exit"
                )

        exit_value = shares * current_price
        pnl = exit_value - invested

        old_balance = self.state.balance
        self.state.balance += pnl
        self.state.total_pnl += pnl

        # Record in loss tracker
        mode_str = "live" if self.is_live_mode else "paper"
        self.loss_tracker.record_trade(pnl, mode=mode_str)
        old_daily_pnl = self.state.daily_pnl
        self.state.daily_pnl = self.loss_tracker.get_daily_pnl(mode=mode_str)

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        logger.info(
            "trade_pnl_calculated",
            mode=mode,
            side=side,
            reason=reason,
            pnl=pnl,
            pnl_pct=(pnl / invested * 100) if invested > 0 else 0,
            old_balance=old_balance,
            new_balance=self.state.balance,
            total_pnl=self.state.total_pnl,
            old_24h_pnl=old_daily_pnl,
            new_24h_pnl=self.state.daily_pnl,
            wins=self.state.wins,
            losses=self.state.losses,
            win_rate=(self.state.wins / max(1, self.state.wins + self.state.losses))
        )

        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": side,
            "entry": entry_price,
            "exit": current_price,
            "pnl": pnl,
            "reason": reason,
            "ticks_held": self.state.ticks_held,
            "mode": "live" if self.is_live_mode else "paper",
        })

        pnl_color = 'green' if pnl > 0 else 'red'
        console.print(
            f"[{pnl_color}]EXIT: {side.upper()} @ {current_price:.3f} | "
            f"PnL=${pnl:+.2f} | {reason} | Held {self.state.ticks_held} ticks[/]"
        )

        # Old ML logging removed - using structured logs instead
        # self._log_trade("exit", {
        #     "side": side,
        #     "entry_price": entry_price,
        #     "exit_price": current_price,
        #     "pnl": pnl,
        #     "reason": reason,
        #     "ticks_held": self.state.ticks_held,
        #     "balance_after": self.state.balance,
        # })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.position_shares = 0.0
        self.state.entry_price = 0.0
        self.state.entry_tick = 0
        self.state.ticks_held = 0
        self.state.max_pnl_seen = 0.0

        # Try immediate redemption (non-blocking)
        await self.try_immediate_redeem()

    async def settle_position(self):
        """Settle position at candle end."""
        mode = "LIVE" if self.is_live_mode else "PAPER"

        if not self.state.position_side:
            logger.debug(
                "settle_no_position",
                mode=mode,
                reason="No position to settle"
            )
            return

        if not self.state.btc_open_price or not self.state.btc_current_price:
            logger.error(
                "settle_missing_btc_price",
                mode=mode,
                position_side=self.state.position_side,
                btc_open=self.state.btc_open_price,
                btc_current=self.state.btc_current_price,
                reason="Missing BTC price data for settlement"
            )
            return

        btc_move = self.state.btc_current_price - self.state.btc_open_price
        btc_move_pct = (btc_move / self.state.btc_open_price) * 100 if self.state.btc_open_price > 0 else 0
        up_won = btc_move > 0

        # Settlement payout
        payout = 0.0
        if self.state.position_side == "yes":
            payout = 1.0 if up_won else 0.0
        else:
            payout = 1.0 if not up_won else 0.0

        # In live mode, use stored actual shares (don't recalculate from balance!)
        # In paper mode, calculate from position size
        if self.is_live_mode:
            shares = self.state.position_shares
            invested = shares * self.state.entry_price
        else:
            invested = self.state.position_size * self.state.balance
            shares = invested / self.state.entry_price
        returned_capital = shares * payout
        pnl = returned_capital - invested

        logger.info(
            "settlement_start",
            mode=mode,
            position_side=self.state.position_side,
            btc_open=self.state.btc_open_price,
            btc_close=self.state.btc_current_price,
            btc_move=btc_move,
            btc_move_pct=btc_move_pct,
            up_won=up_won,
            payout=payout,
            invested=invested,
            shares=shares,
            returned=returned_capital,
            pnl=pnl,
            ticks_held=self.state.ticks_held
        )

        old_balance = self.state.balance
        self.state.balance += pnl
        self.state.total_pnl += pnl

        # Record in loss tracker
        mode_str = "live" if self.is_live_mode else "paper"
        self.loss_tracker.record_trade(pnl, mode=mode_str)
        old_daily_pnl = self.state.daily_pnl
        self.state.daily_pnl = self.loss_tracker.get_daily_pnl(mode=mode_str)

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        logger.info(
            "settlement_pnl",
            mode=mode,
            position_side=self.state.position_side,
            won=pnl > 0,
            pnl=pnl,
            pnl_pct=(pnl / invested * 100) if invested > 0 else 0,
            old_balance=old_balance,
            new_balance=self.state.balance,
            total_pnl=self.state.total_pnl,
            old_24h_pnl=old_daily_pnl,
            new_24h_pnl=self.state.daily_pnl,
            wins=self.state.wins,
            losses=self.state.losses
        )

        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": self.state.position_side,
            "entry": self.state.entry_price,
            "exit": payout,
            "pnl": pnl,
            "reason": "settlement",
            "ticks_held": self.state.ticks_held,
            "mode": "live" if self.is_live_mode else "paper",
        })

        console.print(
            f"[bold purple]SETTLEMENT: {self.state.position_side.upper()} -> "
            f"{'WIN' if pnl > 0 else 'LOSS'} | PnL=${pnl:+.2f}[/bold purple]"
        )

        # Old ML logging removed - using structured logs instead
        # self._log_trade("settlement", {
        #     "side": self.state.position_side,
        #     "entry_price": self.state.entry_price,
        #     "payout": payout,
        #     "btc_open": self.state.btc_open_price,
        #     "btc_close": self.state.btc_current_price,
        #     "up_won": up_won,
        #     "pnl": pnl,
        #     "balance_after": self.state.balance,
        # })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.position_shares = 0.0
        self.state.entry_price = 0.0

        # Try immediate redemption (non-blocking)
        await self.try_immediate_redeem()

    async def switch_trading_mode(self):
        """Switch between paper and live trading mode."""
        old_mode = "LIVE" if self.is_live_mode else "PAPER"
        new_mode = "PAPER" if self.is_live_mode else "LIVE"

        # Don't allow switching if there's an open position
        if self.state.position_side is not None:
            logger.warning(
                "mode_switch_blocked",
                old_mode=old_mode,
                new_mode=new_mode,
                position_side=self.state.position_side,
                position_size=self.state.position_size,
                reason="Cannot switch mode with open position"
            )
            console.print("[bold red]❌ Cannot switch mode with open position! Exit position first.[/bold red]")
            return

        logger.info(
            "mode_switch_start",
            old_mode=old_mode,
            new_mode=new_mode,
            balance=self.state.balance,
            total_pnl=self.state.total_pnl
        )

        console.print(f"[bold yellow]🔄 Switching from {old_mode} to {new_mode} mode...[/bold yellow]")

        # Update config (use lowercase for config value)
        self.config.trading_mode = new_mode.lower()

        # If switching to live mode, initialize executor
        if self.config.trading_mode == "live":
            if not self.trading_config:
                console.print("[bold red]❌ No trading config available! Cannot switch to live mode.[/bold red]")
                self.config.trading_mode = "paper"
                return

            try:
                # Validate config
                self.trading_config.validate()

                # Initialize executor if not already done
                if not self.executor:
                    self.executor = OnchainOrderExecutor(
                        local_rpc_url=self.trading_config.polygon_rpc_url,
                        private_key=self.trading_config.eth_private_key,
                        public_rpc_url=self.trading_config.public_rpc_url,
                        use_public_rpc=self.trading_config.execution.use_public_rpc_for_redeem,
                        use_clob=self.trading_config.execution.use_clob,
                        socks5_proxy=self.trading_config.socks5_proxy,
                    )
                    if await self.executor.connect():
                        await self.executor.ensure_approvals()
                        self.state.wallet_balance = await self.executor.get_usdc_balance()
                        console.print(f"[bold green]✅ Connected to live executor. Wallet: ${self.state.wallet_balance:.2f}[/bold green]")
                    else:
                        console.print("[bold red]❌ Failed to connect executor! Staying in paper mode.[/bold red]")
                        self.config.trading_mode = "paper"
                        return

                # Initialize onchain executor for redemption if needed
                if not self.onchain_executor:
                    self.onchain_executor = OnchainExecutor(
                        local_rpc_url=self.trading_config.polygon_rpc_url,
                        private_key=self.trading_config.eth_private_key,
                        public_rpc_url=self.trading_config.public_rpc_url,
                        use_public_rpc=self.trading_config.execution.use_public_rpc_for_redeem,
                        use_clob=self.trading_config.execution.use_clob,
                        socks5_proxy=self.trading_config.socks5_proxy,
                    )
                    await self.onchain_executor.connect()
                    logger.info("redemption_executor_connected_on_switch", mode="LIVE")

                logger.info(
                    "mode_switch_success",
                    old_mode="PAPER",
                    new_mode="LIVE",
                    wallet_balance=self.state.wallet_balance,
                    executor_connected=self.executor is not None,
                    onchain_executor_connected=self.onchain_executor is not None
                )
                console.print("[bold green]✅ Switched to LIVE mode[/bold green]")
                console.print(f"[bold green]   Wallet Balance: ${self.state.wallet_balance:.2f} USDC[/bold green]")
                console.print(f"[bold green]   Paper Balance: ${self.state.balance:.2f}[/bold green]")
                console.print("[yellow]   (Wallet balance is your real USDC, paper balance tracks performance)[/yellow]")

            except ValueError as e:
                logger.error(
                    "mode_switch_config_error",
                    old_mode="PAPER",
                    new_mode="LIVE",
                    error=str(e),
                    error_type="ValueError",
                    reason="Configuration error during mode switch"
                )
                console.print(f"[bold red]❌ Config error: {e}[/bold red]")
                self.config.trading_mode = "paper"
                return
            except Exception as e:
                logger.error(
                    "mode_switch_error",
                    old_mode="PAPER",
                    new_mode="LIVE",
                    error=str(e),
                    error_type=type(e).__name__,
                    reason="Failed to switch to live mode"
                )
                import traceback
                logger.error(
                    "mode_switch_traceback",
                    traceback=traceback.format_exc()
                )
                console.print(f"[bold red]❌ Error switching to live mode: {e}[/bold red]")
                self.config.trading_mode = "paper"
                return
        else:
            logger.info(
                "mode_switch_success",
                old_mode="LIVE",
                new_mode="PAPER"
            )
            console.print("[bold green]✅ Switched to PAPER mode[/bold green]")

    def reset_daily_limit(self):
        """Reset the rolling loss tracker for the current mode only."""
        mode_str = "live" if self.is_live_mode else "paper"
        mode_display = "LIVE" if self.is_live_mode else "PAPER"
        console.print(f"[bold yellow]🔄 Resetting 24H loss tracker for {mode_display} mode...[/bold yellow]")
        self.loss_tracker.reset(mode=mode_str)
        self.state.daily_pnl = self.loss_tracker.get_daily_pnl(mode=mode_str)
        console.print(f"[bold green]✅ {mode_display} loss tracker reset! Trading resumed.[/bold green]")

    def run_config_editor_sync(self):
        """Synchronous config editor (runs outside Live context)."""
        import yaml

        console.clear()

        while True:
            # Build config menu
            menu_table = Table(title="[bold cyan]⚙️  Configuration Editor[/bold cyan]", show_header=True, header_style="bold magenta")
            menu_table.add_column("#", style="cyan", width=4)
            menu_table.add_column("Setting", style="yellow")
            menu_table.add_column("Current Value", style="green")

            # RPC Settings
            menu_table.add_row("1", "Local RPC URL", self.trading_config.polygon_rpc_url[:50] + "..." if self.trading_config and len(self.trading_config.polygon_rpc_url) > 50 else (self.trading_config.polygon_rpc_url if self.trading_config else "N/A"))
            menu_table.add_row("2", "Public RPC URL", self.trading_config.public_rpc_url[:50] + "..." if self.trading_config and len(self.trading_config.public_rpc_url) > 50 else (self.trading_config.public_rpc_url if self.trading_config else "N/A"))

            # Risk Settings
            menu_table.add_row("3", "Max Daily Loss %", f"{self.config.max_daily_loss_pct:.1f}%")
            menu_table.add_row("4", "Max Position Size (USDC)", f"${self.config.max_position_size_usdc:.2f}")
            menu_table.add_row("5", "Base Position Size %", f"{self.config.base_position_size:.1%}")

            # Model Settings
            menu_table.add_row("6", "Min Confidence", f"{self.config.min_confidence:.2f}")
            menu_table.add_row("7", "Min Expected Return", f"{self.config.min_expected_return:.2%}")

            # Timing
            menu_table.add_row("8", "Refresh Interval (sec)", f"{self.config.refresh_seconds}")

            # Execution
            use_clob_str = "✓ Enabled" if (self.trading_config and self.trading_config.execution.use_clob) else "✗ Disabled"
            menu_table.add_row("9", "Use CLOB API (vs onchain)", use_clob_str)

            # Actions
            menu_table.add_row("", "", "")
            menu_table.add_row("S", "[bold]Save to config.yaml[/bold]", "")
            menu_table.add_row("X", "[bold]Exit (without saving)[/bold]", "")

            console.print(menu_table)

            choice = Prompt.ask("\n[cyan]Select option[/cyan]", default="X")

            try:
                if choice.upper() == "X":
                    console.print("[yellow]Exiting config editor...[/yellow]")
                    break

                elif choice.upper() == "S":
                    if self.save_config_sync():
                        console.print("[bold green]✅ Configuration saved to config.yaml[/bold green]")
                        import time
                        time.sleep(1.5)
                    break

                elif choice == "1":
                    if not self.trading_config:
                        console.print("[red]No trading config available[/red]")
                        continue
                    new_url = Prompt.ask("Enter Local RPC URL", default=self.trading_config.polygon_rpc_url)
                    self.trading_config.polygon_rpc_url = new_url
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.5)

                elif choice == "2":
                    if not self.trading_config:
                        console.print("[red]No trading config available[/red]")
                        continue
                    new_url = Prompt.ask("Enter Public RPC URL", default=self.trading_config.public_rpc_url)
                    self.trading_config.public_rpc_url = new_url
                    # Update execution config too
                    if self.trading_config:
                        self.trading_config.execution.public_rpc_url = new_url
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.3)

                elif choice == "3":
                    new_val = FloatPrompt.ask("Enter Max Daily Loss %", default=self.config.max_daily_loss_pct)
                    self.config.max_daily_loss_pct = new_val
                    if self.trading_config:
                        self.trading_config.risk.max_daily_loss_pct = new_val
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.3)

                elif choice == "4":
                    current_val = self.config.max_position_size_usdc
                    logger.debug(f"Current max_position_size_usdc: {current_val}")
                    new_val = FloatPrompt.ask("Enter Max Position Size (USDC)", default=current_val if current_val > 0 else 10.0)
                    self.config.max_position_size_usdc = new_val
                    if self.trading_config:
                        self.trading_config.risk.max_position_size_usdc = new_val
                    console.print(f"[green]✓ Updated to ${new_val:.2f}[/green]")
                    logger.info(f"Updated max_position_size_usdc to {new_val}")
                    import time
                    time.sleep(0.3)

                elif choice == "5":
                    new_val = FloatPrompt.ask("Enter Base Position Size (0.0-1.0)", default=self.config.base_position_size)
                    self.config.base_position_size = max(0.0, min(1.0, new_val))
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.3)

                elif choice == "6":
                    new_val = FloatPrompt.ask("Enter Min Confidence (0.0-1.0)", default=self.config.min_confidence)
                    self.config.min_confidence = max(0.0, min(1.0, new_val))
                    if self.trading_config:
                        self.trading_config.model.min_confidence = new_val
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.3)

                elif choice == "7":
                    new_val = FloatPrompt.ask("Enter Min Expected Return (0.0-1.0)", default=self.config.min_expected_return)
                    self.config.min_expected_return = max(0.0, min(1.0, new_val))
                    if self.trading_config:
                        self.trading_config.model.min_expected_return = new_val
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.3)

                elif choice == "8":
                    new_val = IntPrompt.ask("Enter Refresh Interval (seconds)", default=self.config.refresh_seconds)
                    self.config.refresh_seconds = max(1, new_val)
                    console.print("[green]✓ Updated[/green]")
                    import time
                    time.sleep(0.3)

                elif choice == "9":
                    if not self.trading_config:
                        console.print("[red]No trading config available[/red]")
                        continue
                    current_val = self.trading_config.execution.use_clob
                    new_val = Confirm.ask(
                        f"Use CLOB API for all operations?\n"
                        f"  • Enabled: BUY and SELL via CLOB API (~2% fees, full functionality)\n"
                        f"  • Disabled: BUY via onchain split (free), hold until resolution for exit (free)\n"
                        f"Current: {'Enabled' if current_val else 'Disabled'}",
                        default=current_val
                    )
                    self.trading_config.execution.use_clob = new_val
                    console.print(f"[green]✓ Updated to {'Enabled' if new_val else 'Disabled'}[/green]")
                    import time
                    time.sleep(0.5)

                else:
                    console.print("[red]Invalid option[/red]")
                    import time
                    time.sleep(0.3)

                console.clear()

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Cancelled[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import time
                time.sleep(1)

        console.clear()

    async def open_config_editor(self):
        """Open config editor (wrapper for async context)."""
        # Stop Live display
        if self._live_display:
            self._live_display.stop()

        await asyncio.sleep(0.3)

        # Run synchronous editor
        await asyncio.to_thread(self.run_config_editor_sync)

        # Restart Live display
        if self._live_display:
            self._live_display.start()

        console.clear()

    def save_config_sync(self) -> bool:
        """Save current configuration to config.yaml."""
        try:
            config_path = Path("config.yaml")

            # Validate max_position_size_usdc before saving
            max_pos_size = self.config.max_position_size_usdc
            if max_pos_size <= 0:
                logger.warning(f"Invalid max_position_size_usdc: {max_pos_size}, using default 10.0")
                max_pos_size = 10.0
                self.config.max_position_size_usdc = 10.0

            # Build config dict with ALL settings
            config_data = {
                "trading_mode": self.config.trading_mode,
                "risk": {
                    "max_daily_loss_pct": self.config.max_daily_loss_pct,
                    "max_position_size_usdc": max_pos_size,
                    "min_balance_usdc": getattr(self.config, 'min_balance_usdc', 10.0),  # Default if missing
                },
                "model": {
                    "path": self.config.model_path,
                    "min_confidence": self.config.min_confidence,
                    "min_expected_return": self.config.min_expected_return,
                },
                "execution": {
                    "use_clob": False,  # Default
                    "order_timeout_seconds": 30,
                    "poll_interval_seconds": 5,
                    "use_public_rpc_for_redeem": True,
                    "public_rpc_url": "https://polygon-rpc.com",
                },
                "logging": {
                    "dir": self.config.log_dir,
                    "enable_ml_logging": self.config.enable_ml_logging,
                },
            }

            # Override execution settings from trading_config if available
            if self.trading_config:
                config_data["execution"]["use_clob"] = self.trading_config.execution.use_clob
                config_data["execution"]["order_timeout_seconds"] = self.trading_config.execution.order_timeout_seconds
                config_data["execution"]["poll_interval_seconds"] = self.trading_config.execution.poll_interval_seconds
                config_data["execution"]["use_public_rpc_for_redeem"] = self.trading_config.execution.use_public_rpc_for_redeem
                config_data["execution"]["public_rpc_url"] = self.trading_config.execution.public_rpc_url

            # Write to file with comments
            with open(config_path, 'w') as f:
                # Write header comments
                f.write("# Yang Trading Configuration\n")
                f.write("# This file controls trading behavior and risk parameters\n\n")

                # Write trading mode section
                f.write("# Trading Mode\n")
                f.write('# "paper" - Simulated trading (default, safe for testing)\n')
                f.write('# "live"  - Real on-chain trading with actual funds\n')
                f.write(f"trading_mode: {config_data['trading_mode']}\n\n")

                # Write risk section
                f.write("# Risk Controls\n")
                f.write("risk:\n")
                f.write("  # Maximum daily loss as percentage of starting balance\n")
                f.write("  # Trading pauses automatically when this limit is hit\n")
                f.write(f"  max_daily_loss_pct: {config_data['risk']['max_daily_loss_pct']}\n")
                f.write("\n")
                f.write("  # Maximum position size in USDC for a single trade\n")
                f.write(f"  max_position_size_usdc: {config_data['risk']['max_position_size_usdc']}\n")
                f.write("\n")
                f.write("  # Minimum balance to maintain (don't trade below this)\n")
                f.write(f"  min_balance_usdc: {config_data['risk']['min_balance_usdc']}\n\n")

                # Write execution section
                f.write("# Execution Settings\n")
                f.write("execution:\n")
                f.write("  # Use CLOB API for all operations (BUY and SELL)\n")
                f.write("  # true  - Use CLOB API for both entry and exit (~2% fees, full functionality)\n")
                f.write("  # false - Use onchain split for entry (free), hold until resolution for exit (free)\n")
                f.write(f"  use_clob: {str(config_data['execution']['use_clob']).lower()}\n")
                f.write("\n")
                f.write("  # Order timeout in seconds (cancel unfilled orders after this)\n")
                f.write(f"  order_timeout_seconds: {config_data['execution']['order_timeout_seconds']}\n")
                f.write("\n")
                f.write("  # Polling interval for trade confirmation (seconds)\n")
                f.write(f"  poll_interval_seconds: {config_data['execution']['poll_interval_seconds']}\n")
                f.write("\n")
                f.write("  # Use public RPC for redemption (more reliable)\n")
                f.write(f"  use_public_rpc_for_redeem: {str(config_data['execution']['use_public_rpc_for_redeem']).lower()}\n")
                f.write("\n")
                f.write("  # Public RPC URL (fallback for redemption)\n")
                f.write(f'  public_rpc_url: "{config_data["execution"]["public_rpc_url"]}"\n\n')

                # Write model section
                f.write("# Model Settings (can be overridden by CLI)\n")
                f.write("model:\n")
                f.write(f'  path: "{config_data["model"]["path"]}"\n')
                f.write(f"  min_confidence: {config_data['model']['min_confidence']}\n")
                f.write(f"  min_expected_return: {config_data['model']['min_expected_return']}\n\n")

                # Write logging section
                f.write("# Logging\n")
                f.write("logging:\n")
                f.write(f'  dir: "{config_data["logging"]["dir"]}"\n')
                f.write(f"  enable_ml_logging: {str(config_data['logging']['enable_ml_logging']).lower()}\n")

            return True

        except Exception as e:
            console.print(f"[red]Failed to save config: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    async def is_position_redeemable(self, condition_id: str) -> bool:
        """
        Check if a position is ready for redemption using the Data API.

        Returns True if the position can be redeemed, False otherwise.
        Uses the Data API to check the 'redeemable' field for this specific wallet.
        """
        try:
            if not self.onchain_executor:
                return False

            wallet_address = self.onchain_executor.address
            if not wallet_address:
                return False

            # Use SOCKS proxy if available
            if SOCKS_AVAILABLE and self.trading_config.socks5_proxy:
                from httpx_socks import SyncProxyTransport
                transport = SyncProxyTransport.from_url(self.trading_config.socks5_proxy)
                client = httpx.Client(transport=transport, verify=False, timeout=10)
            else:
                client = httpx.Client(verify=False, timeout=10)

            # Fetch positions for this wallet
            base_url = "https://data-api.polymarket.com/positions"
            params = {
                "user": wallet_address,
                "limit": 100,
                "offset": 0,
                "sizeThreshold": 0.0001,
            }

            response = client.get(base_url, params=params)
            response.raise_for_status()
            positions = response.json()
            client.close()

            if not isinstance(positions, list):
                return False

            # Find position with this condition_id
            for pos in positions:
                if pos.get('conditionId') == condition_id:
                    # Check if redeemable AND has value > 0
                    is_redeemable = pos.get('redeemable', False)
                    has_value = float(pos.get('currentValue', 0)) > 0
                    return is_redeemable and has_value

            return False

        except Exception as e:
            logger.debug(f"Position not redeemable yet: {e}")
            return False

    async def try_immediate_redeem(self):
        """
        Schedule delayed redemption with polling.

        Waits 1 minute, then polls every 30 seconds to check if position is redeemable.
        Only redeems when position is actually ready.
        """
        if not self.is_live_mode:
            logger.debug(
                "redeem_skipped_paper_mode",
                mode="PAPER",
                reason="Redemption only available in live mode"
            )
            return

        if not self.onchain_executor:
            logger.error(
                "redeem_skipped_no_executor",
                mode="LIVE",
                reason="Onchain executor not initialized"
            )
            return

        if not self.state.condition_id:
            logger.debug(
                "redeem_skipped_no_condition",
                mode="LIVE",
                reason="No condition_id set"
            )
            return

        # Schedule background redemption polling
        condition_id = self.state.condition_id
        logger.info(
            "redeem_scheduled",
            mode="LIVE",
            condition_id=condition_id[:16] + "...",
            wait_time=60,
            poll_interval=30,
            message="Will check redeemability in 60 seconds, then every 30 seconds"
        )
        console.print("[dim]📅 Redemption scheduled (1 min wait, then polling every 30s)[/dim]")

        # Start background task
        asyncio.create_task(self._poll_and_redeem(condition_id))

    async def _poll_and_redeem(self, condition_id: str):
        """
        Background task: Wait 1 minute, then poll every 30s until redeemable, then redeem.
        """
        try:
            # Wait 1 minute before first check
            await asyncio.sleep(60)

            logger.info(
                "redeem_polling_start",
                mode="LIVE",
                condition_id=condition_id[:16] + "...",
                message="Starting redemption polling"
            )

            # Poll every 30 seconds until redeemable
            max_attempts = 20  # Stop after 10 minutes of polling
            for attempt in range(max_attempts):
                logger.debug(
                    "redeem_poll_check",
                    mode="LIVE",
                    condition_id=condition_id[:16] + "...",
                    attempt=attempt + 1,
                    max_attempts=max_attempts
                )

                # Check if redeemable
                if await self.is_position_redeemable(condition_id):
                    logger.info(
                        "redeem_ready",
                        mode="LIVE",
                        condition_id=condition_id[:16] + "...",
                        message="Position is now redeemable"
                    )
                    console.print("[green]✓ Position ready for redemption[/green]")

                    # Redeem now
                    await self._execute_redeem(condition_id)
                    return
                else:
                    logger.debug(
                        "redeem_not_ready",
                        mode="LIVE",
                        condition_id=condition_id[:16] + "...",
                        attempt=attempt + 1
                    )

                # Wait 30 seconds before next check
                if attempt < max_attempts - 1:
                    await asyncio.sleep(30)

            logger.warning(
                "redeem_poll_timeout",
                mode="LIVE",
                condition_id=condition_id[:16] + "...",
                message="Stopped polling after max attempts - position may not be resolved"
            )

        except Exception as e:
            logger.error(
                "redeem_poll_error",
                mode="LIVE",
                condition_id=condition_id[:16] + "..." if condition_id else None,
                error=str(e),
                error_type=type(e).__name__
            )

    async def _execute_redeem(self, condition_id: str):
        """
        Execute the actual redemption transaction.
        """
        try:
            from web3 import Web3

            logger.info(
                "redeem_attempt_start",
                mode="LIVE",
                condition_id=condition_id[:16] + "..."
            )

            # CTF contract
            ctf = self.onchain_executor.public_w3.eth.contract(
                address=Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"),
                abi=[
                    {
                        "inputs": [
                            {"name": "collateralToken", "type": "address"},
                            {"name": "parentCollectionId", "type": "bytes32"},
                            {"name": "conditionId", "type": "bytes32"},
                            {"name": "indexSets", "type": "uint256[]"},
                        ],
                        "name": "redeemPositions",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function",
                    },
                ],
            )

            # USDC contract
            usdc = self.onchain_executor.public_w3.eth.contract(
                address=Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
                abi=[
                    {
                        "constant": True,
                        "inputs": [{"name": "_owner", "type": "address"}],
                        "name": "balanceOf",
                        "outputs": [{"name": "balance", "type": "uint256"}],
                        "type": "function",
                    }
                ],
            )

            # Get balance before
            balance_before = usdc.functions.balanceOf(self.onchain_executor.address).call()

            # Prepare redemption parameters
            condition_id = self.state.condition_id
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            condition_bytes = bytes.fromhex(condition_id[2:])

            parent_collection_id = bytes(32)
            index_sets = [1, 2]  # YES, NO

            # Build transaction
            nonce = self.onchain_executor.public_w3.eth.get_transaction_count(
                self.onchain_executor.address, "latest"
            )

            latest_block = self.onchain_executor.public_w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)
            max_priority_fee = self.onchain_executor.public_w3.to_wei(50, 'gwei')
            max_fee = base_fee * 2 + max_priority_fee

            tx = ctf.functions.redeemPositions(
                Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
                parent_collection_id,
                condition_bytes,
                index_sets,
            ).build_transaction({
                "from": self.onchain_executor.address,
                "nonce": nonce,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
                "gas": 300000,
            })

            # Sign and send
            logger.info(
                "redeem_tx_building",
                mode="LIVE",
                condition_id=self.state.condition_id[:16] + "...",
                nonce=nonce,
                max_fee=max_fee,
                gas_limit=300000
            )
            console.print(f"[dim]🔄 Attempting redemption...[/dim]")

            signed_tx = self.onchain_executor.account.sign_transaction(tx)
            tx_hash = self.onchain_executor.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(
                "redeem_tx_sent",
                mode="LIVE",
                tx_hash=tx_hash_hex,
                condition_id=self.state.condition_id[:16] + "..."
            )

            # Wait for receipt
            receipt = self.onchain_executor.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                # Calculate redeemed amount
                balance_after = usdc.functions.balanceOf(self.onchain_executor.address).call()
                usdc_redeemed = (balance_after - balance_before) / 1e6

                logger.info(
                    "redeem_success",
                    mode="LIVE",
                    tx_hash=tx_hash_hex,
                    condition_id=self.state.condition_id[:16] + "...",
                    balance_before=balance_before / 1e6,
                    balance_after=balance_after / 1e6,
                    usdc_redeemed=usdc_redeemed,
                    gas_used=receipt.get("gasUsed")
                )

                if usdc_redeemed > 0.01:  # Only show if meaningful amount
                    console.print(f"[green]✅ Redeemed ${usdc_redeemed:.2f} USDC[/green]")
                    # Update wallet balance
                    if self.executor:
                        old_wallet = self.state.wallet_balance
                        self.state.wallet_balance = await self.executor.get_usdc_balance()
                        logger.info(
                            "wallet_balance_updated_redeem",
                            mode="LIVE",
                            old_balance=old_wallet,
                            new_balance=self.state.wallet_balance,
                            change=self.state.wallet_balance - old_wallet
                        )
                else:
                    logger.warning(
                        "redeem_zero_payout",
                        mode="LIVE",
                        tx_hash=tx_hash_hex,
                        condition_id=self.state.condition_id[:16] + "...",
                        reason="Position redeemed but payout was zero"
                    )
            else:
                logger.warning(
                    "redeem_tx_reverted",
                    mode="LIVE",
                    tx_hash=tx_hash_hex,
                    condition_id=self.state.condition_id[:16] + "...",
                    reason="Transaction reverted - position not ready for redemption"
                )

        except Exception as e:
            # Silently fail - position probably not resolved yet
            logger.debug(
                "redeem_exception",
                mode="LIVE",
                condition_id=self.state.condition_id[:16] + "..." if self.state.condition_id else None,
                error=str(e),
                error_type=type(e).__name__,
                reason="Exception during redemption attempt"
            )

    async def auto_redeem(self):
        """
        Auto-redeem winning shares in live mode when market closes.

        Called after each candle settlement to automatically redeem resolved positions.
        This ensures we get our USDC back immediately after market resolution.

        Note: When use_clob=True, positions are exited via CLOB SELL (tokens → USDC),
        so redemption is unnecessary (user doesn't hold tokens anymore).
        """
        if not self.is_live_mode or not self.onchain_executor:
            return

        # Skip redemption when using CLOB - positions are exited via SELL orders
        if self.trading_config and self.trading_config.execution.use_clob:
            logger.debug("Skipping redemption check (CLOB mode - positions exited via SELL)")
            return

        if not self.state.condition_id:
            logger.debug("No condition_id set, skipping redemption check")
            return

        try:
            # Check if resolved
            if await self.onchain_executor.check_resolution(self.state.condition_id):
                console.print("[bold cyan]🎁 Auto-redeeming resolved position...[/bold cyan]")

                # Get token IDs for redemption
                yes_token_id = self.state.active_yes_id
                no_token_id = self.state.active_no_id

                if not yes_token_id or not no_token_id:
                    logger.warning("Token IDs not available for redemption")
                    return

                # Redeem position
                result = await self.onchain_executor.redeem_position(
                    condition_id=self.state.condition_id,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                )

                if result.success:
                    if result.skipped_reason:
                        console.print(f"[dim]⏭️  {result.skipped_reason}[/dim]")
                    elif result.tx_hash:
                        console.print(
                            f"[bold green]✅ Redeemed ${result.usdc_redeemed:.2f} USDC[/bold green] "
                            f"(tx: {result.tx_hash[:12]}...)"
                        )
                        # Update wallet balance
                        if self.executor:
                            self.state.wallet_balance = await self.executor.get_usdc_balance()
                            console.print(f"  [green]Updated balance: ${self.state.wallet_balance:.2f}[/green]")
                else:
                    console.print(f"[red]❌ Redemption failed: {result.error}[/red]")
            else:
                logger.debug("Market not yet resolved, skipping redemption")
        except Exception as e:
            logger.error(f"Auto-redeem error: {e}")

    async def fetch_all_positions(self) -> List[Dict]:
        """Fetch all positions from Data API."""
        if not self.is_live_mode or not self.onchain_executor:
            console.print("[yellow]Force redeem only available in live mode[/yellow]")
            return []

        wallet_address = self.onchain_executor.address
        if not wallet_address:
            console.print("[red]No wallet address available[/red]")
            return []

        try:
            # Use SOCKS proxy if available
            if SOCKS_AVAILABLE and self.trading_config.socks5_proxy:
                from httpx_socks import SyncProxyTransport
                transport = SyncProxyTransport.from_url(self.trading_config.socks5_proxy)
                client = httpx.Client(transport=transport, verify=False, timeout=30)
            else:
                client = httpx.Client(verify=False, timeout=30)

            base_url = "https://data-api.polymarket.com/positions"
            all_positions = []
            offset = 0
            limit = 500

            console.print(f"[cyan]Fetching positions for {wallet_address[:8]}...{wallet_address[-6:]}[/cyan]")

            while True:
                params = {
                    "user": wallet_address,
                    "limit": limit,
                    "offset": offset,
                    "sizeThreshold": 0.0001,
                    "sortBy": "CURRENT",
                    "sortDirection": "DESC",
                }

                response = client.get(base_url, params=params)
                response.raise_for_status()
                positions = response.json()

                if not isinstance(positions, list):
                    raise ValueError("Unexpected response format")

                num_positions = len(positions)
                all_positions.extend(positions)

                if num_positions < limit:
                    break

                offset += limit

            client.close()
            console.print(f"[green]✓ Found {len(all_positions)} positions[/green]")
            return all_positions

        except Exception as e:
            console.print(f"[red]Failed to fetch positions: {e}[/red]")
            return []

    async def force_redeem_single_position(
        self,
        condition_id: str,
        title: str,
        value: float,
    ) -> Dict[str, Any]:
        """
        Force redemption attempt without checks (uses direct contract call).

        This bypasses all resolution checks and attempts redemption directly.
        """
        if not self.onchain_executor or not self.onchain_executor.public_w3:
            return {"success": False, "error": "Not connected"}

        console.print(f"[cyan]  {title[:60]}...[/cyan]")
        console.print(f"[dim]    Value: ${value:.2f} | Condition: {condition_id[:16]}...[/dim]")

        try:
            from web3 import Web3

            # CTF contract
            ctf = self.onchain_executor.public_w3.eth.contract(
                address=Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"),
                abi=[
                    {
                        "inputs": [
                            {"name": "collateralToken", "type": "address"},
                            {"name": "parentCollectionId", "type": "bytes32"},
                            {"name": "conditionId", "type": "bytes32"},
                            {"name": "indexSets", "type": "uint256[]"},
                        ],
                        "name": "redeemPositions",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function",
                    },
                ],
            )

            # USDC contract
            usdc = self.onchain_executor.public_w3.eth.contract(
                address=Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
                abi=[
                    {
                        "constant": True,
                        "inputs": [{"name": "_owner", "type": "address"}],
                        "name": "balanceOf",
                        "outputs": [{"name": "balance", "type": "uint256"}],
                        "type": "function",
                    }
                ],
            )

            # Get balance before
            balance_before = usdc.functions.balanceOf(self.onchain_executor.address).call()

            # Prepare redemption parameters
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            condition_bytes = bytes.fromhex(condition_id[2:])

            parent_collection_id = bytes(32)
            index_sets = [1, 2]  # YES, NO

            # Build transaction
            nonce = self.onchain_executor.public_w3.eth.get_transaction_count(
                self.onchain_executor.address, "latest"
            )

            latest_block = self.onchain_executor.public_w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)
            max_priority_fee = self.onchain_executor.public_w3.to_wei(50, 'gwei')
            max_fee = base_fee * 2 + max_priority_fee

            tx = ctf.functions.redeemPositions(
                Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
                parent_collection_id,
                condition_bytes,
                index_sets,
            ).build_transaction({
                "from": self.onchain_executor.address,
                "nonce": nonce,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
                "gas": 300000,
            })

            # Sign and send
            console.print(f"[yellow]    → Sending transaction...[/yellow]")
            signed_tx = self.onchain_executor.account.sign_transaction(tx)
            tx_hash = self.onchain_executor.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            console.print(f"[dim]    TX: {tx_hash_hex[:16]}...[/dim]")
            console.print(f"[yellow]    → Waiting for confirmation...[/yellow]")

            # Wait for receipt
            receipt = self.onchain_executor.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                # Calculate redeemed amount
                balance_after = usdc.functions.balanceOf(self.onchain_executor.address).call()
                usdc_redeemed = (balance_after - balance_before) / 1e6

                console.print(f"[green]    ✓ Redeemed ${usdc_redeemed:.2f}[/green]")
                return {
                    "success": True,
                    "tx_hash": tx_hash_hex,
                    "usdc_redeemed": usdc_redeemed,
                }
            else:
                console.print(f"[red]    ✗ Transaction reverted[/red]")
                return {
                    "success": False,
                    "tx_hash": tx_hash_hex,
                    "error": "Transaction reverted",
                }

        except Exception as e:
            error_msg = str(e)

            # Parse common errors
            if "execution reverted" in error_msg.lower():
                if "payout is zero" in error_msg.lower():
                    console.print(f"[yellow]    ⊘ No payout (already redeemed or no value)[/yellow]")
                else:
                    console.print(f"[yellow]    ⊘ Reverted: {error_msg[:60]}...[/yellow]")
            else:
                console.print(f"[red]    ✗ Error: {error_msg[:60]}...[/red]")

            return {"success": False, "error": error_msg}

    async def force_redeem_all_positions(self):
        """
        Fetch all positions and attempt force redemption on redeemable ones.

        This is useful for cleaning up all resolved positions at once.
        Only attempts redemption on positions that are marked as redeemable
        and have value > 0.
        """
        if not self.is_live_mode or not self.onchain_executor:
            console.print("[yellow]Force redeem only available in live mode[/yellow]")
            return

        console.print("\n[bold yellow]⚡ FORCE REDEEM ALL POSITIONS ⚡[/bold yellow]")

        # Fetch all positions
        positions = await self.fetch_all_positions()

        if not positions:
            console.print("[yellow]No positions found[/yellow]")
            return

        # Filter positions: only redeemable and value > 0
        redeemable_positions = [
            pos for pos in positions
            if pos.get('redeemable', False) and float(pos.get('currentValue', 0)) > 0
        ]

        if not redeemable_positions:
            console.print("[yellow]No redeemable positions with value > 0 found[/yellow]")
            return

        skipped_count = len(positions) - len(redeemable_positions)
        if skipped_count > 0:
            console.print(f"[dim]Skipping {skipped_count} positions (not redeemable or value = 0)[/dim]\n")

        # Display summary
        total_value = sum(float(p.get('currentValue', 0)) for p in redeemable_positions)
        console.print(f"[bold]Found {len(redeemable_positions)} redeemable positions (Total Value: ${total_value:.2f})[/bold]")

        # Ask for confirmation
        console.print("\n[yellow]Press Enter to continue or Ctrl+C to cancel...[/yellow]")
        try:
            await asyncio.get_event_loop().run_in_executor(None, input)
        except KeyboardInterrupt:
            console.print("[yellow]Cancelled[/yellow]")
            return

        # Get initial balance
        initial_balance = await self.onchain_executor.get_usdc_balance()
        console.print(f"\n[cyan]Initial Balance: ${initial_balance:.2f}[/cyan]\n")

        # Attempt redemptions
        results = {"success": 0, "failed": 0, "total_redeemed": 0.0}

        for i, pos in enumerate(redeemable_positions, 1):
            console.print(f"[bold][{i}/{len(redeemable_positions)}][/bold]")

            result = await self.force_redeem_single_position(
                condition_id=pos.get('conditionId', ''),
                title=pos.get('title', 'Unknown'),
                value=float(pos.get('currentValue', 0)),
            )

            if result.get("success"):
                results["success"] += 1
                results["total_redeemed"] += result.get("usdc_redeemed", 0)
            else:
                results["failed"] += 1

            console.print()

            # Delay between attempts
            if i < len(redeemable_positions):
                await asyncio.sleep(2)

        # Get final balance
        final_balance = await self.onchain_executor.get_usdc_balance()

        # Update wallet balance
        if self.executor:
            self.state.wallet_balance = await self.executor.get_usdc_balance()

        # Display results
        console.print(f"[cyan]{'═'*70}[/cyan]")
        console.print(f"[bold green]REDEMPTION COMPLETE[/bold green]")
        console.print(f"[cyan]{'═'*70}[/cyan]")
        console.print(f"  Successful: [green]{results['success']}[/green]")
        console.print(f"  Failed: [red]{results['failed']}[/red]")
        console.print(f"  Initial Balance: ${initial_balance:.2f}")
        console.print(f"  Final Balance: [green]${final_balance:.2f}[/green]")
        console.print(f"  Total Redeemed: [bold green]${results['total_redeemed']:+.2f}[/bold green]")
        console.print(f"[cyan]{'═'*70}[/cyan]\n")

    async def keyboard_handler(self):
        """Handle keyboard input for interactive controls using a thread."""
        import threading
        import queue

        key_queue = queue.Queue()

        def read_keys():
            """Read keyboard input in a separate thread."""
            try:
                import sys
                import termios
                import tty

                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())  # Use cbreak instead of raw mode

                    while self._running:
                        try:
                            import select
                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                key = sys.stdin.read(1).lower()
                                key_queue.put(key)
                        except Exception:
                            pass
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception as e:
                logger.debug(f"Keyboard thread error: {e}")

        # Start keyboard reading thread
        kbd_thread = threading.Thread(target=read_keys, daemon=True)
        kbd_thread.start()

        # Process keys from queue
        while self._running:
            try:
                # Non-blocking queue check
                try:
                    key = key_queue.get_nowait()

                    if key == 'l':
                        await self.switch_trading_mode()
                    elif key == 'c':
                        await self.open_config_editor()
                    elif key == 'r':
                        self.reset_daily_limit()
                    elif key == 'f':
                        await self.force_redeem_all_positions()
                    elif key == 'q':
                        console.print("\n[yellow]Quit requested...[/yellow]")
                        self._running = False
                        break

                except queue.Empty:
                    pass

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.debug(f"Key processing error: {e}")
                await asyncio.sleep(0.1)

    async def trading_loop(self):
        """Main trading loop."""
        console.print("[bold green]Starting unified paper trading...[/bold green]")

        while self._running:
            try:
                # Market discovery
                current_ts = self.get_current_candle_timestamp()
                if self.state.current_candle_ts != current_ts:
                    if self.state.current_candle_ts is not None:
                        await self.settle_position()

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
                        # Try to fetch true open price
                        true_open = await self.fetch_btc_candle_open()
                        if true_open:
                            self.state.btc_open_price = true_open
                            console.print(f"[bold cyan]Synced BTC Candle Open: ${true_open:.2f}[/bold cyan]")
                        else:
                            # Fallback to current if fetch fails
                            self.state.btc_open_price = btc_price
                            logger.warning("Using current BTC price as open (fetch failed)")

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
                mode = "LIVE" if self.is_live_mode else "PAPER"
                logger.error(
                    "trading_loop_error",
                    mode=mode,
                    error=str(e),
                    error_type=type(e).__name__,
                    position_side=self.state.position_side,
                    position_size=self.state.position_size,
                    balance=self.state.balance,
                    total_pnl=self.state.total_pnl,
                    reason="Unexpected error in trading loop"
                )
                import traceback
                logger.error(
                    "trading_loop_traceback",
                    mode=mode,
                    traceback=traceback.format_exc()
                )
                await asyncio.sleep(5)

    def build_display(self) -> Panel:
        """Build display panel with mode indicator and enhanced sections."""
        from rich.layout import Layout
        from rich.text import Text

        # Mode indicator
        if self.is_live_mode:
            mode_text = Text("🔴 LIVE", style="bold red")
        else:
            mode_text = Text("📝 PAPER", style="bold green")

        # Left: Status table
        left_table = Table(show_header=False, box=None, padding=(0, 1))
        left_table.add_column("Key", style="dim", width=14)
        left_table.add_column("Value", style="bold")

        # Balance section
        if self.is_live_mode:
            left_table.add_row("Wallet", f"${self.state.wallet_balance:.2f} USDC")
        left_table.add_row("Balance", f"${self.state.balance:.2f}")
        pnl_color = "green" if self.state.total_pnl >= 0 else "red"
        left_table.add_row("Total PnL", f"[{pnl_color}]${self.state.total_pnl:+.2f}[/]")

        # 24H Rolling PnL with warning (mode-specific)
        mode_str = "live" if self.is_live_mode else "paper"
        daily_pnl = self.loss_tracker.get_daily_pnl(mode=mode_str)
        daily_pnl_pct = self.loss_tracker.get_daily_pnl_pct(mode=mode_str)
        if daily_pnl >= 0:
            daily_color = "green"
        elif abs(daily_pnl_pct) >= self.config.max_daily_loss_pct * 0.8:
            daily_color = "yellow"  # Approaching limit
        else:
            daily_color = "red"
        left_table.add_row("24H PnL", f"[{daily_color}]${daily_pnl:+.2f} ({daily_pnl_pct:+.1f}%)[/]")

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

        # Position (with absolute value unrealized PnL)
        if self.state.position_side:
            left_table.add_row("", "")
            color = "green" if self.state.position_side == "yes" else "red"
            current_price = (
                self.state.last_yes_price if self.state.position_side == "yes"
                else self.state.last_no_price
            )
            unrealized_pnl_pct = (current_price - self.state.entry_price) / self.state.entry_price

            # Calculate absolute unrealized PnL
            # In live mode: use actual stored shares (don't recalculate from balance!)
            # In paper mode: calculate from position size
            if self.is_live_mode:
                shares = self.state.position_shares
                invested = shares * self.state.entry_price
            else:
                invested = self.state.position_size * self.state.balance
                shares = invested / self.state.entry_price if self.state.entry_price > 0 else 0

            unrealized_pnl_abs = shares * (current_price - self.state.entry_price)
            pnl_color = "green" if unrealized_pnl_pct > 0 else "red"

            left_table.add_row(
                "Position",
                f"[{color}]{self.state.position_side.upper()}[/] @ {self.state.entry_price:.3f}"
            )
            left_table.add_row("Size", f"{self.state.position_size:.1%} (${invested:.0f})")
            left_table.add_row("Unrealized", f"[{pnl_color}]{unrealized_pnl_pct:+.1%} (${unrealized_pnl_abs:+.2f})[/]")
            left_table.add_row("Ticks Held", f"{self.state.ticks_held}")

        # Right: Trades table (with entry price column and mode)
        trades_table = Table(show_header=True, box=None, padding=(0, 1))
        trades_table.add_column("Time", style="dim", width=8)
        trades_table.add_column("Mode", width=5)
        trades_table.add_column("Side", width=4)
        trades_table.add_column("Entry", width=6)
        trades_table.add_column("Exit", width=6)
        trades_table.add_column("PnL", width=10)
        trades_table.add_column("Reason", width=6)

        recent_trades = list(reversed(self.state.trades[-10:]))
        for trade in recent_trades:
            side_color = "green" if trade["side"] == "yes" else "red"
            pnl_color = "green" if trade["pnl"] > 0 else "red"
            time_str = trade["time"].strftime("%H:%M:%S")

            # Mode indicator
            mode = trade.get("mode", "paper")
            mode_icon = "🔴" if mode == "live" else "📝"
            mode_style = "red" if mode == "live" else "green"

            reason_short = {
                "model_signal": "MODEL",
                "time_expiry": "TIME",
                "settlement": "SETT",
            }.get(trade.get("reason", ""), trade.get("reason", "")[:5])

            trades_table.add_row(
                time_str,
                f"[{mode_style}]{mode_icon}[/]",
                f"[{side_color}]{trade['side'].upper()[:3]}[/]",
                f"{trade['entry']:.3f}",
                f"{trade['exit']:.3f}",
                f"[{pnl_color}]${trade['pnl']:+.2f}[/]",
                reason_short,
            )

        if not recent_trades:
            trades_table.add_row("", "", "", "[dim]No trades yet[/]", "", "", "")

        # Build panels
        status_title = f"[bold cyan]Status[/] {mode_text}"
        left_panel = Panel(left_table, title=status_title, border_style="cyan", width=42)
        right_panel = Panel(trades_table, title="[bold yellow]Recent Trades[/]", border_style="yellow")

        # Controls panel
        controls_table = Table(show_header=False, box=None, padding=(0, 1))
        controls_table.add_column("Key", style="cyan", width=8)
        controls_table.add_column("Action", style="dim")
        controls_table.add_row("[L]", "Switch Paper/Live mode")
        controls_table.add_row("[C]", "Open Config Editor")
        controls_table.add_row("[R]", "Reset daily loss limit")
        if self.is_live_mode:
            controls_table.add_row("[F]", "Force redeem all positions")
        controls_table.add_row("[Q]", "Quit")
        controls_panel = Panel(controls_table, title="[bold cyan]⌨️  Controls[/]", border_style="cyan")

        layout = Columns([left_panel, right_panel], expand=True)
        from rich.console import Group
        layout = Group(controls_panel, layout)

        # Add loss limit warning banner if needed
        mode_str = "live" if self.is_live_mode else "paper"
        if self.loss_tracker.is_limit_hit(self.config.max_daily_loss_pct, mode=mode_str):
            warning = Panel(
                "[bold yellow]⚠ 24H LOSS LIMIT HIT — NEW ENTRIES PAUSED (Press [R] to reset)[/bold yellow]",
                border_style="yellow",
            )
            layout = Group(warning, layout)

        title = "[bold blue]Unified Trading[/bold blue]"
        if self.is_live_mode:
            title = "[bold red]🔴 LIVE TRADING[/bold red]"

        return Panel(layout, title=title, border_style="red" if self.is_live_mode else "blue")

    async def run(self):
        """Run paper trading with live display."""
        mode = "LIVE" if self.is_live_mode else "PAPER"

        logger.info(
            "session_start",
            mode=mode,
            model_path=self.config.model_path,
            initial_balance=self.state.balance,
            starting_balance=self.config.initial_balance,
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            base_position_size=self.config.base_position_size,
            min_position_size=self.config.min_position_size,
            max_position_size=self.config.max_position_size,
            enable_ml_logging=self.config.enable_ml_logging
        )

        self.load_model()
        self._setup_logging()
        await self._setup_clients()

        self._running = True

        logger.info(
            "session_initialized",
            mode=mode,
            model_loaded=self.model is not None,
            clients_setup=True,
            executor_connected=self.executor is not None if self.is_live_mode else None,
            onchain_executor_connected=self.onchain_executor is not None if self.is_live_mode else None
        )

        # Start both trading and keyboard handler tasks
        trading_task = asyncio.create_task(self.trading_loop())
        keyboard_task = asyncio.create_task(self.keyboard_handler())

        try:
            with Live(self.build_display(), refresh_per_second=1) as live:
                self._live_display = live
                while self._running:
                    try:
                        # Update display (will be paused when config editor stops Live)
                        live.update(self.build_display())
                    except Exception:
                        # Live might be stopped, skip update
                        pass
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            self._running = False
        finally:
            # Cancel tasks
            trading_task.cancel()
            keyboard_task.cancel()
            try:
                await asyncio.gather(trading_task, keyboard_task, return_exceptions=True)
            except:
                pass
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
        self.log_file: Optional[Path] = None
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

    # Live trading mode
    parser.add_argument("--live", action="store_true", help="Enable live trading mode (requires .env credentials)")
    parser.add_argument("--clob", action="store_true", help="Use CLOB API for all operations (BUY and SELL) instead of onchain split/merge")

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
        # Load trading config from file + env + CLI
        cli_args = {
            "live": args.live,
            "clob": args.clob,
            "model": args.model,
            "min_confidence": args.min_confidence,
            "min_return": args.min_return,
            "log_dir": args.log_dir,
            "no_ml_log": args.no_ml_log,
            "balance": args.balance,
        }
        trading_config = load_config(cli_args=cli_args)

        # Validate if live mode
        if trading_config.is_live:
            try:
                trading_config.validate()
            except ValueError as e:
                console.print(f"[bold red]Config Error: {e}[/bold red]")
                console.print("[yellow]Set required env vars in .env file or create from .env.example[/yellow]")
                return

        # Build paper trade config
        config = UnifiedPaperTradeConfig(
            model_path=trading_config.model.path,
            initial_balance=args.balance,
            min_confidence=trading_config.model.min_confidence,
            min_expected_return=trading_config.model.min_expected_return,
            log_dir=trading_config.logging.dir,
            enable_ml_logging=trading_config.logging.enable_ml_logging,
            trading_mode=trading_config.trading_mode,
            max_daily_loss_pct=trading_config.risk.max_daily_loss_pct,
            max_position_size_usdc=trading_config.risk.max_position_size_usdc,
        )

        trader = UnifiedPaperTrader(config, trading_config)
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
