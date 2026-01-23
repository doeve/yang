#!/usr/bin/env python3
"""
Token-Centric Paper Trading for Polymarket.

This uses the new token-centric architecture:
1. Token features (price momentum, convergence, time decay)
2. Edge detector (P(YES wins) estimation → mispricing signal)
3. SAC policy (entry/exit timing optimization)

Key differences from BTC-centric approach:
- Primary signal: token price dynamics, not BTC prediction
- Edge = Our P(YES) - Market YES price
- Profits from token price movement + convergence
"""

# Suppress warnings
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

with suppress_stderr():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.data.token_features import TokenFeatureBuilder
from src.models.edge_detector import EdgeDetectorModel, load_edge_detector
from src.simulation.token_trading_env import TokenTradingEnv, TokenTradingConfig

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
class TokenTradingState:
    """Trading state for token-centric approach."""
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
    entry_tick: int = 0  # Tick when position was entered (for min hold time)
    highest_price_since_entry: float = 0.0  # For trailing stop
    lowest_price_since_entry: float = 1.0  # For trailing stop

    # Market state
    last_yes_price: float = 0.5
    last_no_price: float = 0.5
    btc_open_price: Optional[float] = None
    btc_current_price: Optional[float] = None

    # Price history
    yes_price_history: List[float] = field(default_factory=list)
    no_price_history: List[float] = field(default_factory=list)
    btc_price_history: List[float] = field(default_factory=list)

    # Edge metrics (with EMA smoothing)
    current_edge: float = 0.0
    ema_edge: float = 0.0  # EMA-smoothed edge for stable signals
    p_yes_estimate: float = 0.5
    confidence: float = 0.0

    # Trade history
    trades: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TokenPaperTradeConfig:
    """Configuration for token paper trading."""
    edge_detector_path: str = "./logs/edge_detector_v1"
    sac_model_path: str = "./logs/token_sac_v1/final_model.zip"
    vec_normalize_path: str = "./logs/token_sac_v1/vecnormalize.pkl"

    initial_balance: float = 1000.0
    base_position_size: float = 0.10  # Base 10% of balance per trade
    max_position_size: float = 0.30  # Max 30% for high-edge opportunistic trades

    # Edge thresholds
    min_edge_to_trade: float = 0.05  # Only trade if |edge| > 5%
    min_confidence: float = 0.6  # Minimum confidence to trade
    opportunistic_edge: float = 0.15  # Edge threshold for larger positions

    # Risk management
    stop_loss_pct: float = 0.20  # Exit if price drops 20% from entry
    take_profit_pct: float = 0.40  # Exit if price rises 40% from entry
    min_hold_ticks: int = 3  # Minimum ticks before allowing exit (prevents churning)

    # EMA smoothing for edge signal
    edge_ema_alpha: float = 0.4  # Smoothing factor (0.4 = moderate smoothing)

    # Polymarket API
    polymarket_api: str = "https://clob.polymarket.com"
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"

    # Refresh interval
    refresh_seconds: int = 5

    # Logging
    log_dir: str = "./logs/paper_trade"
    enable_ml_logging: bool = True


class TokenPaperTrader:
    """Paper trader using token-centric approach."""

    def __init__(self, config: TokenPaperTradeConfig):
        self.config = config
        self.state = TokenTradingState(balance=config.initial_balance)
        self.feature_builder = TokenFeatureBuilder()

        # Models (loaded later)
        self.edge_detector: Optional[EdgeDetectorModel] = None
        self.sac_model: Optional[SAC] = None
        self.vec_normalize = None

        # HTTP clients
        self.polymarket_client: Optional[httpx.AsyncClient] = None
        self.binance_client: Optional[httpx.AsyncClient] = None

        # Candle tracking
        self.current_candle_start: Optional[datetime] = None
        self._running = False

        # ML logging
        self.ml_log_file: Optional[Path] = None
        self.ml_log_handle = None
        self._tick_count = 0

    def _get_model_name(self) -> str:
        """Extract model name from config paths."""
        # Try edge detector path first
        edge_path = Path(self.config.edge_detector_path)
        if edge_path.exists():
            return edge_path.name

        # Fallback to SAC model path
        sac_path = Path(self.config.sac_model_path)
        if sac_path.exists():
            return sac_path.stem  # e.g. "final_model" from "final_model.zip"

        # Default
        return "unknown_model"

    def _setup_ml_logging(self):
        """Initialize ML logging file."""
        if not self.config.enable_ml_logging:
            return

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        model_name = self._get_model_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ml_log_file = log_dir / f"{model_name}_{timestamp}.jsonl"

        self.ml_log_handle = open(self.ml_log_file, 'w')
        console.print(f"[blue]ML logging to: {self.ml_log_file}[/blue]")

        # Write header/metadata
        metadata = {
            "type": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "edge_detector_path": self.config.edge_detector_path,
            "sac_model_path": self.config.sac_model_path,
            "config": {
                "initial_balance": self.config.initial_balance,
                "position_size": self.config.position_size,
                "min_edge_to_trade": self.config.min_edge_to_trade,
                "min_confidence": self.config.min_confidence,
            }
        }
        self.ml_log_handle.write(json.dumps(metadata) + "\n")
        self.ml_log_handle.flush()

    def _log_ml_tick(
        self,
        features: np.ndarray,
        edge_result: Dict[str, float],
        sac_action: Dict[str, float],
    ):
        """Log ML predictions and features for a single tick."""
        if not self.config.enable_ml_logging or self.ml_log_handle is None:
            return

        self._tick_count += 1

        # Build comprehensive log entry
        log_entry = {
            "type": "tick",
            "tick": self._tick_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candle_ts": self.state.current_candle_ts,
            "time_remaining": self.get_time_remaining(),

            # Market State
            "market": {
                "yes_price": self.state.last_yes_price,
                "no_price": self.state.last_no_price,
                "btc_price": self.state.btc_current_price,
                "btc_open": self.state.btc_open_price,
                "yes_history_len": len(self.state.yes_price_history),
                "no_history_len": len(self.state.no_price_history),
            },

            # Edge Detector Output
            "edge_detector": {
                "p_yes": edge_result.get("p_yes", 0.5),
                "edge": edge_result.get("edge", 0.0),
                "ema_edge": edge_result.get("ema_edge", 0.0),
                "confidence": edge_result.get("confidence", 0.0),
            },

            # SAC Policy Output
            "sac_action": {
                "direction": sac_action.get("direction", 0.0),
                "size": sac_action.get("size", 0.0),
                "hold_prob": sac_action.get("hold_prob", 1.0),
                "exit_signal": sac_action.get("exit_signal", 0.0),
            },

            # Position State
            "position": {
                "side": self.state.position_side,
                "size": self.state.position_size,
                "entry_price": self.state.entry_price,
            },

            # Account State
            "account": {
                "balance": self.state.balance,
                "total_pnl": self.state.total_pnl,
                "wins": self.state.wins,
                "losses": self.state.losses,
            },

            # Feature Engineering (49-dim vector)
            "features": {
                "raw": features.tolist(),
                "feature_dim": len(features),
                # Named feature groups (based on TokenFeatureBuilder)
                "price_state": features[:7].tolist() if len(features) >= 7 else [],
                "momentum": features[7:23].tolist() if len(features) >= 23 else [],
                "volatility": features[23:29].tolist() if len(features) >= 29 else [],
                "convergence": features[29:35].tolist() if len(features) >= 35 else [],
                "time_features": features[35:39].tolist() if len(features) >= 39 else [],
                "relative_value": features[39:43].tolist() if len(features) >= 43 else [],
                "btc_guidance": features[43:49].tolist() if len(features) >= 49 else [],
            },
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")

        # Flush periodically
        if self._tick_count % 10 == 0:
            self.ml_log_handle.flush()

    def _log_trade(self, action: str, details: Dict[str, Any]):
        """Log trade events."""
        if not self.config.enable_ml_logging or self.ml_log_handle is None:
            return

        log_entry = {
            "type": "trade",
            "action": action,  # "entry", "exit", "settlement"
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tick": self._tick_count,
            **details
        }

        self.ml_log_handle.write(json.dumps(log_entry) + "\n")
        self.ml_log_handle.flush()

    def _close_ml_logging(self):
        """Close ML logging file."""
        if self.ml_log_handle:
            # Write summary
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

    def load_models(self):
        """Load edge detector and SAC models."""
        console.print("[bold blue]Loading models...[/bold blue]")

        # Load edge detector
        edge_path = Path(self.config.edge_detector_path)
        if edge_path.exists():
            self.edge_detector = load_edge_detector(str(edge_path))
            console.print(f"  ✓ Edge detector loaded from {edge_path}")
        else:
            console.print(f"  [yellow]⚠ Edge detector not found at {edge_path}[/yellow]")

        # Load SAC
        sac_path = Path(self.config.sac_model_path)
        if sac_path.exists():
            with suppress_stderr():
                self.sac_model = SAC.load(str(sac_path))
            console.print(f"  ✓ SAC model loaded from {sac_path}")

            # Load VecNormalize
            vec_path = Path(self.config.vec_normalize_path)
            if vec_path.exists():
                dummy_env = DummyVecEnv([lambda: TokenTradingEnv()])
                self.vec_normalize = VecNormalize.load(str(vec_path), dummy_env)
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                console.print(f"  ✓ VecNormalize loaded from {vec_path}")
        else:
            console.print(f"  [yellow]⚠ SAC model not found at {sac_path}[/yellow]")

    async def _setup_clients(self):
        """Setup HTTP clients."""
        self.binance_client = httpx.AsyncClient(timeout=10)

        if SOCKS_AVAILABLE:
            transport = httpx_socks.AsyncProxyTransport.from_url(SOCKS5_PROXY_URL)
            self.polymarket_client = httpx.AsyncClient(transport=transport, timeout=30, verify=False)
        else:
            console.print("[yellow]⚠ SOCKS5 not available, using direct connection[/yellow]")
            self.polymarket_client = httpx.AsyncClient(timeout=30)

    def get_current_candle_timestamp(self) -> int:
        """Get current 15-min candle timestamp."""
        now = int(datetime.now(timezone.utc).timestamp())
        return (now // 900) * 900

    async def fetch_active_market(self, timestamp: int) -> bool:
        """Fetch current active market IDs from Gamma API."""
        slug = f"btc-updown-15m-{timestamp}"
        try:
            url = f"{self.config.polymarket_gamma_api}/events/slug/{slug}"
            response = await self.polymarket_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                if markets:
                    import json
                    market = markets[0]
                    clob_tokens = market.get("clobTokenIds", "[]")
                    if isinstance(clob_tokens, str):
                        clob_tokens = json.loads(clob_tokens)
                    
                    if len(clob_tokens) >= 2:
                        self.state.active_yes_id = clob_tokens[0]
                        self.state.active_no_id = clob_tokens[1]
                        self.state.current_candle_ts = timestamp
                        console.print(f"[blue]Found new market: {slug}[/blue]")
                        return True
        except Exception as e:
            logger.error(f"Market discovery error: {e}")
        return False

    async def fetch_polymarket_prices(self) -> Optional[Dict[str, float]]:
        """Fetch current YES/NO prices from Polymarket."""
        if not self.state.active_yes_id:
            return None

        try:
            # Try Midpoint first (cheaper/cleaner)
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
            
            # Fallback to Orderbook
            url = f"{self.config.polymarket_api}/book"
            params = {"token_id": self.state.active_yes_id}
            response = await self.polymarket_client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                if bids and asks:
                    best_bid = float(bids[0]["price"])
                    best_ask = float(asks[0]["price"])
                    yes_price = (best_bid + best_ask) / 2
                elif asks:
                    yes_price = float(asks[0]["price"])
                elif bids:
                    yes_price = float(bids[0]["price"])
                else:
                    yes_price = 0.5

                return {
                    "yes_price": yes_price,
                    "no_price": 1.0 - yes_price,
                }
        except Exception as e:
            logger.error(f"Polymarket fetch error: {e}")
        return None

    async def fetch_btc_price(self) -> Optional[float]:
        """Fetch current BTC price from Binance."""
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

    def get_time_remaining(self) -> float:
        """Get fraction of 15-min candle remaining based on wall clock."""
        now = datetime.now(timezone.utc)
        
        minute_block = (now.minute // 15) * 15
        start_of_candle = now.replace(minute=minute_block, second=0, microsecond=0)
        
        duration = timedelta(minutes=15)
        
        elapsed = now - start_of_candle
        remaining_seconds = (duration - elapsed).total_seconds()
        
        return max(0.0, min(1.0, remaining_seconds / duration.total_seconds()))

    def compute_edge(self) -> tuple[Dict[str, float], np.ndarray]:
        """Compute edge using edge detector with EMA smoothing.

        Returns:
            Tuple of (edge_result_dict, features_array)
        """
        # Build features
        time_remaining = self.get_time_remaining()
        features = self.feature_builder.compute_features(
            yes_prices=np.array(self.state.yes_price_history[-300:] or [0.5]),
            no_prices=np.array(self.state.no_price_history[-300:] or [0.5]),
            time_remaining=time_remaining,
            btc_prices=np.array(self.state.btc_price_history[-300:] or [100000]),
            btc_open=self.state.btc_open_price,
        )

        if self.edge_detector is None:
            return {"edge": 0.0, "ema_edge": 0.0, "p_yes": 0.5, "confidence": 0.0}, features

        # Get edge prediction
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        edge_result = self.edge_detector.predict_edge(
            features_tensor,
            current_yes_price=self.state.last_yes_price,
        )

        raw_edge = float(edge_result["edge"][0])

        # Apply EMA smoothing to edge signal to reduce noise/reversals
        alpha = self.config.edge_ema_alpha
        if self.state.ema_edge == 0.0:
            # Initialize EMA with first value
            ema_edge = raw_edge
        else:
            ema_edge = alpha * raw_edge + (1 - alpha) * self.state.ema_edge

        # Update state
        self.state.ema_edge = ema_edge

        return {
            "edge": raw_edge,
            "ema_edge": ema_edge,
            "p_yes": float(edge_result["p_yes"][0]),
            "confidence": float(edge_result["confidence"][0]),
        }, features

    def get_sac_action(self) -> Dict[str, float]:
        """Get action from SAC policy."""
        if self.sac_model is None:
            return {"direction": 0.0, "size": 0.0, "hold_prob": 1.0, "exit_signal": 0.0}

        # Build observation
        time_remaining = self.get_time_remaining()
        token_features = self.feature_builder.compute_features(
            yes_prices=np.array(self.state.yes_price_history[-300:] or [0.5]),
            no_prices=np.array(self.state.no_price_history[-300:] or [0.5]),
            time_remaining=time_remaining,
            btc_prices=np.array(self.state.btc_price_history[-300:] or [100000]),
            btc_open=self.state.btc_open_price,
        )

        # Add position state
        if self.state.position_side:
            has_position = 1.0
            position_side = 1.0 if self.state.position_side == "yes" else -1.0
            current_price = self.state.last_yes_price if self.state.position_side == "yes" else self.state.last_no_price
            position_pnl = (current_price - self.state.entry_price) / self.state.entry_price
            steps_in_position = 0.5  # Approximate
        else:
            has_position = 0.0
            position_side = 0.0
            position_pnl = 0.0
            steps_in_position = 0.0

        obs = np.concatenate([
            token_features,
            [has_position, position_side, position_pnl, steps_in_position]
        ]).astype(np.float32)

        # Normalize
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

        # Get action
        action, _ = self.sac_model.predict(obs.reshape(1, -1), deterministic=True)
        action = action[0]

        return {
            "direction": float(action[0]),
            "size": float(np.clip(action[1], 0, 1)),
            "hold_prob": float(action[2]),
            "exit_signal": float(action[3]),
        }

    async def trading_loop(self):
        """Main trading loop."""
        console.print("[bold green]Starting token-centric paper trading...[/bold green]")

        while self._running:
            try:
                # Fetch prices
                # 1. Market Discovery Check
                current_ts = self.get_current_candle_timestamp()
                if self.state.current_candle_ts != current_ts:
                    if self.state.current_candle_ts is not None:
                        self.settle_position()

                    found = await self.fetch_active_market(current_ts)
                    if found:
                        # Reset price history for new candle
                        self.state.yes_price_history = []
                        self.state.no_price_history = []
                        self.state.btc_price_history = []
                        self.state.btc_open_price = None
                        self.current_candle_start = datetime.now(timezone.utc)
                    else:
                        console.print("[yellow]Waiting for new market...[/yellow]")
                        await asyncio.sleep(5)
                        continue

                # 2. Fetch prices (using the IDs found above)
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

                    # Track candle open
                    if self.state.btc_open_price is None:
                        self.state.btc_open_price = btc_price
                        self.current_candle_start = datetime.now(timezone.utc)

                # Compute edge
                edge_result, features = self.compute_edge()
                self.state.current_edge = edge_result["edge"]
                self.state.p_yes_estimate = edge_result["p_yes"]
                self.state.confidence = edge_result["confidence"]

                # Get SAC action
                sac_action = self.get_sac_action()

                # Log ML data
                self._log_ml_tick(features, edge_result, sac_action)

                # Execute trading logic
                await self.execute_trading_logic(edge_result, sac_action)

                # Keep history bounded
                max_history = 1000
                if len(self.state.yes_price_history) > max_history:
                    self.state.yes_price_history = self.state.yes_price_history[-max_history:]
                    self.state.no_price_history = self.state.no_price_history[-max_history:]
                    self.state.btc_price_history = self.state.btc_price_history[-max_history:]

                await asyncio.sleep(self.config.refresh_seconds)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)

    async def execute_trading_logic(self, edge: Dict, action: Dict):
        """Execute trading based on edge and SAC action with risk management."""
        # Check for position exit
        if self.state.position_side:
            current_price = (
                self.state.last_yes_price if self.state.position_side == "yes"
                else self.state.last_no_price
            )

            # Update price extremes for tracking
            self.state.highest_price_since_entry = max(
                self.state.highest_price_since_entry, current_price
            )
            self.state.lowest_price_since_entry = min(
                self.state.lowest_price_since_entry, current_price
            )

            should_exit = False
            exit_reason = ""

            # Check minimum hold time (prevents churning)
            ticks_held = self._tick_count - self.state.entry_tick
            can_exit = ticks_held >= self.config.min_hold_ticks

            if can_exit:
                # STOP-LOSS: Exit if price dropped too much from entry
                price_change = (current_price - self.state.entry_price) / self.state.entry_price
                if price_change < -self.config.stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"

                # TAKE-PROFIT: Exit if price rose enough from entry
                if price_change > self.config.take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"

                # EDGE REVERSAL: Exit if EMA edge flipped against our position
                # If we're long YES and ema_edge is now significantly negative
                if self.state.position_side == "yes" and edge["ema_edge"] < -0.05:
                    should_exit = True
                    exit_reason = "edge_reversal"
                elif self.state.position_side == "no" and edge["ema_edge"] > 0.05:
                    should_exit = True
                    exit_reason = "edge_reversal"

                # SAC exit signal (use smoothed edge confirmation)
                if action["exit_signal"] > 0.5:
                    should_exit = True
                    exit_reason = "sac_signal"

            # Time-based exit (near settlement) - override min hold time
            if self.get_time_remaining() < 0.05:
                should_exit = True
                exit_reason = "time_expiry"

            if should_exit:
                self.execute_exit(reason=exit_reason)
                return

        # Check for entry
        if self.state.position_side is None:
            # SAC says don't trade (but respect high edge opportunities)
            ema_edge_abs = abs(edge["ema_edge"])
            is_opportunistic = ema_edge_abs >= self.config.opportunistic_edge

            if action["hold_prob"] > 0.5 and not is_opportunistic:
                return

            # Check edge threshold (use EMA-smoothed edge for stability)
            if ema_edge_abs < self.config.min_edge_to_trade:
                return

            # Check confidence
            if edge["confidence"] < self.config.min_confidence:
                return

            # Calculate position size based on edge magnitude and SAC size output
            position_size = self._calculate_position_size(edge, action)

            # Determine direction based on EMA-smoothed edge
            if edge["ema_edge"] > 0:
                # YES is underpriced, buy YES
                self.execute_entry("yes", position_size, action)
            else:
                # NO is underpriced, buy NO
                self.execute_entry("no", position_size, action)

    def _calculate_position_size(self, edge: Dict, action: Dict) -> float:
        """Calculate position size based on edge, confidence, and SAC output.

        Uses Kelly-inspired sizing:
        - Base size scales with edge magnitude
        - SAC size output modulates the final size
        - Opportunistic trades get larger allocations
        """
        ema_edge_abs = abs(edge["ema_edge"])
        confidence = edge["confidence"]

        # Base size from edge magnitude (linear scaling)
        # At min_edge (0.05), use base_position_size
        # At opportunistic_edge (0.15), use max_position_size
        edge_range = self.config.opportunistic_edge - self.config.min_edge_to_trade
        edge_normalized = (ema_edge_abs - self.config.min_edge_to_trade) / edge_range
        edge_normalized = np.clip(edge_normalized, 0, 1)

        base_size = (
            self.config.base_position_size +
            edge_normalized * (self.config.max_position_size - self.config.base_position_size)
        )

        # Modulate by confidence (higher confidence = fuller size)
        confidence_factor = 0.5 + 0.5 * confidence  # Range: 0.5 to 1.0

        # Use SAC size output as additional modulation
        sac_size = np.clip(action["size"], 0.3, 1.0)  # Don't let SAC reduce too much

        # Final position size
        position_size = base_size * confidence_factor * sac_size

        # Clamp to bounds
        position_size = np.clip(
            position_size,
            self.config.base_position_size * 0.5,  # Min: half of base
            self.config.max_position_size
        )

        return float(position_size)

    def execute_entry(self, side: str, position_size: float = None, action: Dict = None):
        """Execute position entry with dynamic position sizing."""
        price = self.state.last_yes_price if side == "yes" else self.state.last_no_price

        # Use provided position size or fall back to base
        size = position_size if position_size is not None else self.config.base_position_size

        self.state.position_side = side
        self.state.position_size = size
        self.state.entry_price = price
        self.state.entry_tick = self._tick_count
        self.state.highest_price_since_entry = price
        self.state.lowest_price_since_entry = price

        # Calculate dollar amount for display
        dollar_size = size * self.state.balance

        console.print(
            f"[green]ENTRY: {side.upper()} @ {price:.3f} | "
            f"Size={size:.1%} (${dollar_size:.0f}) | "
            f"Edge={self.state.ema_edge:+.1%}[/green]"
        )

        # Log trade
        self._log_trade("entry", {
            "side": side,
            "price": price,
            "size": size,
            "dollar_size": dollar_size,
            "edge": self.state.current_edge,
            "ema_edge": self.state.ema_edge,
            "p_yes": self.state.p_yes_estimate,
            "confidence": self.state.confidence,
            "yes_price": self.state.last_yes_price,
            "no_price": self.state.last_no_price,
            "btc_price": self.state.btc_current_price,
            "time_remaining": self.get_time_remaining(),
            "sac_size": action.get("size", 0) if action else 0,
            "sac_direction": action.get("direction", 0) if action else 0,
        })

    def execute_exit(self, reason: str = "manual"):
        """Execute position exit with reason tracking."""
        if self.state.position_side is None:
            return

        side = self.state.position_side
        entry_price = self.state.entry_price
        current_price = self.state.last_yes_price if side == "yes" else self.state.last_no_price

        # Calculate PnL
        invested = self.state.position_size * self.state.balance
        shares = invested / entry_price
        exit_value = shares * current_price
        pnl = exit_value - invested

        # Calculate hold duration
        ticks_held = self._tick_count - self.state.entry_tick

        # Update state
        self.state.balance += pnl
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        # Record trade
        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": side,
            "entry": entry_price,
            "exit": current_price,
            "pnl": pnl,
            "reason": reason,
            "ticks_held": ticks_held,
        })

        # Color-coded output with reason
        pnl_color = 'green' if pnl > 0 else 'red'
        reason_emoji = {
            "stop_loss": "[red]SL[/red]",
            "take_profit": "[green]TP[/green]",
            "edge_reversal": "[yellow]ER[/yellow]",
            "sac_signal": "[blue]SAC[/blue]",
            "time_expiry": "[magenta]TIME[/magenta]",
        }.get(reason, reason)

        console.print(
            f"[{pnl_color}]EXIT: {side.upper()} @ {current_price:.3f} | "
            f"PnL=${pnl:+.2f} | {reason_emoji} | Held {ticks_held} ticks[/]"
        )

        # Log trade
        self._log_trade("exit", {
            "side": side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": pnl,
            "invested": invested,
            "shares": shares,
            "exit_value": exit_value,
            "edge_at_exit": self.state.current_edge,
            "ema_edge_at_exit": self.state.ema_edge,
            "time_remaining": self.get_time_remaining(),
            "balance_after": self.state.balance,
            "reason": reason,
            "ticks_held": ticks_held,
            "highest_price": self.state.highest_price_since_entry,
            "lowest_price": self.state.lowest_price_since_entry,
        })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.entry_price = 0.0
        self.state.entry_tick = 0
        self.state.highest_price_since_entry = 0.0
        self.state.lowest_price_since_entry = 1.0

    def settle_position(self):
        """Settle position at candle end based on BTC price movement."""
        if not self.state.position_side:
            return

        # We need the open price of the candle we just finished
        if not self.state.btc_open_price or not self.state.btc_current_price:
            return

        # Determine if UP won or DOWN won
        btc_move = self.state.btc_current_price - self.state.btc_open_price
        up_won = btc_move > 0

        # Calculate Payout (1.0 for win, 0.0 for loss)
        payout = 0.0
        if self.state.position_side == "yes":
            payout = 1.0 if up_won else 0.0
        else: # "no"
            payout = 1.0 if not up_won else 0.0

        # Calculate PnL
        invested = self.state.position_size * self.state.balance
        # Shares = Invested / Entry Price
        shares = invested / self.state.entry_price

        # Return = Shares * Payout
        returned_capital = shares * payout
        pnl = returned_capital - invested

        # Update Balance
        self.state.balance += pnl
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        console.print(f"[bold purple]SETTLEMENT:[/bold purple] {self.state.position_side.upper()} -> {'WIN' if pnl > 0 else 'LOSS'} | PnL=${pnl:+.2f}")

        # Log settlement
        self._log_trade("settlement", {
            "side": self.state.position_side,
            "entry_price": self.state.entry_price,
            "payout": payout,
            "btc_open": self.state.btc_open_price,
            "btc_close": self.state.btc_current_price,
            "btc_move": btc_move,
            "up_won": up_won,
            "pnl": pnl,
            "invested": invested,
            "shares": shares,
            "returned_capital": returned_capital,
            "balance_after": self.state.balance,
        })

        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.entry_price = 0.0

    def build_display(self) -> Panel:
        """Build display panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Balance", f"${self.state.balance:.2f}")
        pnl_color = "green" if self.state.total_pnl >= 0 else "red"
        table.add_row("Total PnL", f"[{pnl_color}]${self.state.total_pnl:+.2f}[/]")

        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        table.add_row("Win Rate", f"{win_rate:.1%} ({self.state.wins}/{self.state.wins + self.state.losses})")

        table.add_row("", "")
        table.add_row("YES Price", f"{self.state.last_yes_price:.3f}")
        table.add_row("NO Price", f"{self.state.last_no_price:.3f}")
        table.add_row("Time Left", f"{self.get_time_remaining():.1%}")

        table.add_row("", "")
        edge_color = "green" if self.state.ema_edge > 0 else "red" if self.state.ema_edge < 0 else "white"
        table.add_row("Raw Edge", f"{self.state.current_edge:+.1%}")
        table.add_row("EMA Edge", f"[{edge_color}]{self.state.ema_edge:+.1%}[/]")
        table.add_row("P(YES wins)", f"{self.state.p_yes_estimate:.1%}")
        table.add_row("Confidence", f"{self.state.confidence:.1%}")

        if self.state.position_side:
            table.add_row("", "")
            color = "green" if self.state.position_side == "yes" else "red"
            current_price = (
                self.state.last_yes_price if self.state.position_side == "yes"
                else self.state.last_no_price
            )
            unrealized_pnl = (current_price - self.state.entry_price) / self.state.entry_price
            pnl_color = "green" if unrealized_pnl > 0 else "red"

            table.add_row(
                "Position",
                f"[{color}]{self.state.position_side.upper()}[/] @ {self.state.entry_price:.3f}"
            )
            table.add_row("Size", f"{self.state.position_size:.1%}")
            table.add_row("Unrealized", f"[{pnl_color}]{unrealized_pnl:+.1%}[/]")
            table.add_row("Ticks Held", f"{self._tick_count - self.state.entry_tick}")

            # Show stop-loss/take-profit levels
            table.add_row(
                "SL/TP",
                f"SL: {self.state.entry_price * (1 - self.config.stop_loss_pct):.3f} | "
                f"TP: {self.state.entry_price * (1 + self.config.take_profit_pct):.3f}"
            )

        return Panel(table, title="[bold blue]Token-Centric Paper Trading v2[/bold blue]", border_style="blue")

    async def run(self):
        """Run paper trading with live display."""
        self.load_models()
        self._setup_ml_logging()
        await self._setup_clients()

        self._running = True

        # Start trading in background
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
    parser = argparse.ArgumentParser(description="Token-centric paper trading")
    parser.add_argument("--edge-model", type=str, default="./logs/edge_detector_v1")
    parser.add_argument("--sac-model", type=str, default="./logs/token_sac_v1/final_model.zip")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge to trade")
    parser.add_argument("--log-dir", type=str, default="./logs/paper_trade", help="Directory for ML logs")
    parser.add_argument("--no-ml-log", action="store_true", help="Disable ML logging")

    args = parser.parse_args()

    config = TokenPaperTradeConfig(
        edge_detector_path=args.edge_model,
        sac_model_path=args.sac_model,
        initial_balance=args.balance,
        min_edge_to_trade=args.min_edge,
        log_dir=args.log_dir,
        enable_ml_logging=not args.no_ml_log,
    )

    trader = TokenPaperTrader(config)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
