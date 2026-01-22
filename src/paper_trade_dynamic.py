#!/usr/bin/env python3
"""
Paper Trading with SAC Dynamic Model + Polymarket CLOB API.

Uses:
- Real-time BTC data from Binance
- Real Polymarket YES/NO token prices from CLOB API (via SOCKS5 proxy)
- SAC Dynamic model with VecNormalize
"""

import asyncio
import json
import os
import signal
import sys
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
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Try to import SOCKS support
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False
    httpx_socks = None

from src.simulation.deep_lob_dynamic_env import (
    DynamicTradingConfig,
    DeepLOBDynamicEnv,
)
from src.inference.deep_lob_inference import (
    DeepLOBTwoLayerBot,
    DeepLOBInferenceConfig,
)

logger = structlog.get_logger(__name__)
console = Console()

# SOCKS5 proxy configuration (via SSH tunnel)
# To create the tunnel: ssh -D 1080 -N -f root@72.62.114.55
SOCKS5_PROXY_URL = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:1080")


@dataclass
class PaperTradeConfig:
    """Configuration for paper trading with Polymarket CLOB."""
    
    # Model paths
    sac_model: str = "./logs/sac_dynamic_v5/final_model.zip"
    vec_normalize: str = "./logs/sac_dynamic_v5/vecnormalize.pkl"
    deep_lob_model: str = "./logs/deep_lob_balanced"
    
    # Trading settings
    initial_balance: float = 1000.0
    max_position_size: float = 0.25
    candle_minutes: int = 15
    
    # API settings
    binance_url: str = "https://api.binance.com/api/v3"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"
    
    # Logging
    log_file: str = "./logs/paper_trade_dynamic.log"
    update_interval: float = 5.0  # Seconds between updates


@dataclass
class TradingState:
    """Current trading state."""

    balance: float = 1000.0
    total_pnl: float = 0.0

    # Current candle
    candle_timestamp: Optional[int] = None
    candle_market: Optional[Dict] = None

    # Position
    position_side: Optional[str] = None  # "yes" or "no"
    position_size: float = 0.0
    entry_price: float = 0.0

    # Stats
    trades: List[Dict] = field(default_factory=list)
    trades_this_candle: int = 0
    wins: int = 0
    losses: int = 0

    # Last update
    last_yes_price: float = 0.5
    last_no_price: float = 0.5
    last_probs: Dict[str, float] = field(default_factory=dict)

    # Price history for momentum calculation
    yes_price_history: List[float] = field(default_factory=list)
    no_price_history: List[float] = field(default_factory=list)

    # BTC prices for display
    btc_open_price: Optional[float] = None
    btc_current_price: Optional[float] = None


class DynamicPaperTrader:
    """Paper trader with Polymarket CLOB API integration via SOCKS5 proxy."""
    
    def __init__(self, config: Optional[PaperTradeConfig] = None):
        self.config = config or PaperTradeConfig()
        self.state = TradingState(balance=self.config.initial_balance)
        
        # Binance client (direct connection)
        self.binance_client = httpx.AsyncClient(timeout=30)
        
        # Polymarket client (via SOCKS5 proxy if available)
        self.polymarket_client: Optional[httpx.AsyncClient] = None
        
        self.sac_model: Optional[SAC] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self.bot: Optional[DeepLOBTwoLayerBot] = None
        
        self._running = False
        self._setup_logging()
    
    async def _setup_polymarket_client(self):
        """Setup Polymarket client with SOCKS5 proxy if available."""
        if SOCKS_AVAILABLE and httpx_socks:
            console.print(f"[green]Using SOCKS5 proxy: {SOCKS5_PROXY_URL}[/green]")
            self.file_logger.info(f"Using SOCKS5 proxy: {SOCKS5_PROXY_URL}")
            
            transport = httpx_socks.AsyncProxyTransport.from_url(
                SOCKS5_PROXY_URL,
                rdns=True,  # Resolve DNS through proxy
            )
            self.polymarket_client = httpx.AsyncClient(
                transport=transport,
                timeout=httpx.Timeout(60.0),
                follow_redirects=True,
                verify=False,  # Skip SSL verification (proxy may have issues)
            )
        else:
            console.print("[yellow]⚠ SOCKS5 proxy not available, using direct connection for Polymarket[/yellow]")
            self.file_logger.warning("SOCKS5 proxy not available, using direct connection")
            self.polymarket_client = httpx.AsyncClient(timeout=30)
    
    def _setup_logging(self):
        """Setup file logging."""
        import logging
        import os
        
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
        
        self.file_logger = logging.getLogger("paper_trade_dynamic")
        self.file_logger.setLevel(logging.DEBUG)
        self.file_logger.handlers.clear()
        
        fh = logging.FileHandler(self.config.log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        self.file_logger.addHandler(fh)
        
        self.file_logger.info("=" * 60)
        self.file_logger.info("PAPER TRADING SESSION STARTED")
        self.file_logger.info(f"SAC Model: {self.config.sac_model}")
        self.file_logger.info("=" * 60)
    
    def load_models(self):
        """Load SAC, VecNormalize, and DeepLOB models."""
        # Load SAC
        console.print("[blue]Loading SAC Dynamic model...[/blue]")
        sac_path = str(self.config.sac_model).replace('.zip', '')
        self.sac_model = SAC.load(sac_path)
        console.print("[green]✓ SAC loaded[/green]")
        
        # Load VecNormalize
        if Path(self.config.vec_normalize).exists():
            config = DynamicTradingConfig()
            dummy_env = DummyVecEnv([lambda: DeepLOBDynamicEnv(config)])
            self.vec_normalize = VecNormalize.load(self.config.vec_normalize, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            console.print("[green]✓ VecNormalize loaded[/green]")
        
        # Load DeepLOB
        console.print("[blue]Loading DeepLOB...[/blue]")
        inference_config = DeepLOBInferenceConfig()
        self.bot = DeepLOBTwoLayerBot(config=inference_config)
        self.bot.load_models(deep_lob_path=self.config.deep_lob_model, sac_path=None)
        console.print("[green]✓ DeepLOB loaded[/green]")
    
    async def get_current_candle_timestamp(self) -> int:
        """Get current 15-min candle timestamp."""
        now = int(datetime.now(timezone.utc).timestamp())
        return (now // 900) * 900
    
    async def fetch_polymarket_candle(self, timestamp: int) -> Optional[Dict]:
        """Fetch current BTC 15-min market from Polymarket via SOCKS5 proxy."""
        slug = f"btc-updown-15m-{timestamp}"
        
        try:
            response = await self.polymarket_client.get(
                f"{self.config.polymarket_gamma_url}/events/slug/{slug}"
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            markets = data.get("markets", [])
            
            if not markets:
                return None
            
            market = markets[0]
            
            # Parse token IDs
            clob_tokens = market.get("clobTokenIds", "[]")
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            
            # outcomePrices only has valid prices for resolved markets
            # For active markets, we'll need to fetch from CLOB orderbook
            outcome_prices = market.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            
            # Check if market is active (not resolved)
            is_closed = market.get("closed", False)
            
            # For active markets, prices will be fetched separately
            if not is_closed:
                yes_price = 0.5  # Placeholder, will be updated by fetch_live_prices
                no_price = 0.5
            else:
                yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
                no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
            
            return {
                "timestamp": timestamp,
                "slug": slug,
                "title": data.get("title", ""),
                "closed": is_closed,
                "yes_token_id": clob_tokens[0] if clob_tokens else None,
                "no_token_id": clob_tokens[1] if len(clob_tokens) > 1 else None,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(data.get("volume", 0)),
            }
            
        except Exception as e:
            logger.debug(f"Error fetching Polymarket candle: {e}")
            return None
    
    async def fetch_live_prices(self, market: Dict) -> tuple[float, float]:
        """Fetch live YES/NO prices from Polymarket CLOB orderbook."""
        yes_token_id = market.get("yes_token_id")
        no_token_id = market.get("no_token_id")
        
        if not yes_token_id:
            return 0.5, 0.5
        
        try:
            # Fetch YES token prices from CLOB
            # The midprice endpoint gives us the best bid/ask
            response = await self.polymarket_client.get(
                f"{self.config.polymarket_clob_url}/midpoint",
                params={"token_id": yes_token_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                yes_price = float(data.get("mid", 0.5))
                no_price = 1.0 - yes_price  # NO price is complement
                return yes_price, no_price
            else:
                # Fallback: try to get price from book
                response = await self.polymarket_client.get(
                    f"{self.config.polymarket_clob_url}/book",
                    params={"token_id": yes_token_id}
                )
                
                if response.status_code == 200:
                    book = response.json()
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    
                    best_bid = float(bids[0].get("price", 0.5)) if bids else 0.5
                    best_ask = float(asks[0].get("price", 0.5)) if asks else 0.5
                    
                    yes_price = (best_bid + best_ask) / 2
                    no_price = 1.0 - yes_price
                    return yes_price, no_price
                    
        except Exception as e:
            logger.debug(f"Error fetching live prices: {e}")
        
        return 0.5, 0.5
    
    async def fetch_btc_data(self, limit: int = 300) -> pd.DataFrame:
        """Fetch BTC 1s klines from Binance (direct connection)."""
        try:
            response = await self.binance_client.get(
                f"{self.config.binance_url}/klines",
                params={"symbol": "BTCUSDT", "interval": "1s", "limit": limit}
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"
            ])
            
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["price"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["taker_buy_volume"] = df["taker_buy_volume"].astype(float)
            
            return df[["timestamp", "price", "volume", "taker_buy_volume"]]
        except Exception as e:
            logger.error(f"Error fetching BTC data: {e}")
            return pd.DataFrame()
    
    def get_time_remaining(self) -> float:
        """Get fraction of time remaining in current candle."""
        now = datetime.now(timezone.utc)
        minutes = (now.minute // self.config.candle_minutes) * self.config.candle_minutes
        candle_end = now.replace(minute=minutes, second=0, microsecond=0) + timedelta(minutes=self.config.candle_minutes)
        
        total_seconds = self.config.candle_minutes * 60
        remaining = (candle_end - now).total_seconds()
        
        return max(0.0, min(1.0, remaining / total_seconds))
    
    def build_observation(self, probs: np.ndarray, yes_price: float, no_price: float) -> np.ndarray:
        """Build 26-dim observation for SAC with momentum and convergence features."""
        prob_down, prob_hold, prob_up = probs
        predicted_class = np.argmax(probs)
        confidence = max(probs)

        time_remaining = self.get_time_remaining()
        time_in_candle = 1.0 - time_remaining
        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        balance_norm = self.state.balance / self.config.initial_balance - 1.0

        # Edge calculations
        model_implied = 0.5 + (prob_up - prob_down) * 0.5
        edge_up = model_implied - yes_price
        edge_down = (1 - model_implied) - no_price

        # Update price history
        self.state.yes_price_history.append(yes_price)
        self.state.no_price_history.append(no_price)
        # Keep bounded
        if len(self.state.yes_price_history) > 20:
            self.state.yes_price_history = self.state.yes_price_history[-20:]
            self.state.no_price_history = self.state.no_price_history[-20:]

        # === NEW FEATURES ===

        # 1. Token momentum (rate of change over 5 steps)
        momentum_window = 5
        if len(self.state.yes_price_history) >= momentum_window:
            yes_momentum = (self.state.yes_price_history[-1] - self.state.yes_price_history[-momentum_window]) / momentum_window
            no_momentum = (self.state.no_price_history[-1] - self.state.no_price_history[-momentum_window]) / momentum_window
        else:
            yes_momentum = 0.0
            no_momentum = 0.0

        # 2. Convergence velocity
        if yes_price > 0.5:
            convergence_velocity = yes_momentum * 10
        else:
            convergence_velocity = -yes_momentum * 10

        # 3. Price distance to settlement
        price_distance_to_settlement = 0.5 - abs(yes_price - 0.5)

        # 4. YES/NO sum deviation (arbitrage signal)
        yes_no_sum_deviation = (yes_price + no_price) - 1.0

        # 5. Time urgency
        time_urgency_multiplier = 2.0
        time_urgency = 1.0 + (time_urgency_multiplier - 1.0) * (time_in_candle ** 2)

        # Position state
        if self.state.position_side:
            position_sign = 1.0 if self.state.position_side == "yes" else -1.0
            position_size = self.state.position_size * position_sign
        else:
            position_size = 0.0

        obs = np.array([
            # DeepLOB predictions (3)
            prob_down, prob_hold, prob_up,
            # Prediction (2)
            predicted_class / 2.0,
            confidence,
            # Market (3)
            yes_price,
            no_price,
            abs(yes_price + no_price - 1.0),  # Actual spread from sum deviation
            # Time (2)
            time_remaining,
            0.0,  # steps_since_entry (not tracked in paper trading)
            # Position (4)
            position_size,
            0.0,  # unrealized_pnl (computed separately)
            0.0,  # max_pnl
            0.0,  # drawdown
            # History (3)
            self.state.trades_this_candle / 5.0,  # max_trades_per_candle = 5
            win_rate,
            balance_norm,
            # Volatility (1)
            0.5,  # volatility estimate
            # Edge (2)
            edge_up,
            edge_down,
            # === NEW: Momentum (2) ===
            yes_momentum * 100,
            no_momentum * 100,
            # === NEW: Convergence (2) ===
            convergence_velocity,
            price_distance_to_settlement,
            # === NEW: Arbitrage (1) ===
            yes_no_sum_deviation * 10,
            # === NEW: Time urgency (1) ===
            time_urgency / time_urgency_multiplier,
        ], dtype=np.float32)

        return obs
    
    def get_sac_action(self, obs: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
        """Get SAC action with fallback."""
        if self.vec_normalize:
            obs_normalized = self.vec_normalize.normalize_obs(obs.reshape(1, -1))
        else:
            obs_normalized = obs.reshape(1, -1)
        
        action, _ = self.sac_model.predict(obs_normalized, deterministic=True)
        
        direction = float(action[0][0])
        size = float(np.clip(action[0][1], 0.0, 1.0))
        hold_prob = float(action[0][2])
        exit_signal = float(action[0][3]) if len(action[0]) > 3 else 0.0
        
        # Use rule-based fallback if SAC is too conservative
        if hold_prob > 0.7 or size < 0.02:
            result = self._rule_based_action(probs)
            result["exit_signal"] = exit_signal
            return result
        
        if abs(direction) > 0.05:
            if direction > 0:
                return {"action": "buy_yes", "size": max(size * 0.25, 0.10), "exit_signal": exit_signal}
            else:
                return {"action": "buy_no", "size": max(size * 0.25, 0.10), "exit_signal": exit_signal}
        
        result = self._rule_based_action(probs)
        result["exit_signal"] = exit_signal
        return result
    
    def _rule_based_action(self, probs: np.ndarray) -> Dict[str, Any]:
        """Rule-based fallback."""
        prob_down, prob_hold, prob_up = probs
        confidence = max(probs)
        
        if confidence < 0.40:
            return {"action": "hold", "size": 0.0}
        
        if prob_up > prob_down and prob_up > 0.4:
            return {"action": "buy_yes", "size": min(confidence * 0.4, 0.20)}
        elif prob_down > prob_up and prob_down > 0.4:
            return {"action": "buy_no", "size": min(confidence * 0.4, 0.20)}
        
        return {"action": "hold", "size": 0.0, "exit_signal": 0.0}
    
    async def early_exit(self, yes_price: float, no_price: float, reason: str = "signal"):
        """Exit position early at current market prices."""
        if not self.state.position_side:
            return
        
        # Calculate PnL based on current market prices
        if self.state.position_side == "yes":
            exit_price = yes_price
        else:
            exit_price = no_price
        
        pnl = (exit_price - self.state.entry_price) * self.state.position_size
        pnl_dollars = pnl * self.state.balance
        
        self.state.balance += pnl_dollars
        self.state.total_pnl += pnl_dollars
        
        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1
        
        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": self.state.position_side,
            "pnl": pnl_dollars,
            "exit_reason": reason,
        })

        self.state.trades_this_candle += 1
        self.file_logger.info(f"EARLY EXIT ({reason}): {self.state.position_side} @ {exit_price:.3f} | PnL=${pnl_dollars:.2f}")
        console.print(f"\n[{'green' if pnl > 0 else 'red'}]Early Exit ({reason}): {self.state.position_side.upper()} @ {exit_price:.3f} | PnL=${pnl_dollars:+.2f}[/]")
        
        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.entry_price = 0.0
    
    async def execute_trade(self, decision: Dict, yes_price: float, no_price: float):
        """Execute a trade."""
        if self.state.position_side:
            return  # Already have position

        action = decision.get("action", "hold")
        if action == "hold":
            return

        size = decision.get("size", 0.20)

        if action == "buy_yes":
            self.state.position_side = "yes"
            self.state.position_size = size
            self.state.entry_price = yes_price
        else:
            self.state.position_side = "no"
            self.state.position_size = size
            self.state.entry_price = no_price

        self.state.trades_this_candle += 1
        self.file_logger.info(f"TRADE: {action} size={size:.2f} price={self.state.entry_price:.3f}")
        console.print(f"[green]Trade executed: {action.upper()} @ {self.state.entry_price:.3f}[/green]")
    
    async def settle_position(self, market: Dict):
        """Settle position at candle end based on BTC price movement."""
        if not self.state.position_side:
            return
        
        if not self.state.btc_open_price or not self.state.btc_current_price:
            console.print("[yellow]No BTC price data for settlement[/yellow]")
            return
        
        # Determine outcome based on BTC price movement
        btc_move = self.state.btc_current_price - self.state.btc_open_price
        up_won = btc_move > 0
        
        # Calculate PnL
        # If we bought YES and Up won, we win (settle at 1.0)
        # If we bought YES and Down won, we lose (settle at 0.0)
        # If we bought NO and Down won, we win (settle at 1.0)
        # If we bought NO and Up won, we lose (settle at 0.0)
        
        if self.state.position_side == "yes":
            if up_won:
                # YES wins - settle at 1.0
                settlement = 1.0
            else:
                # YES loses - settle at 0.0
                settlement = 0.0
            pnl = (settlement - self.state.entry_price) * self.state.position_size
        else:  # NO position
            if not up_won:
                # NO wins - settle at 1.0
                settlement = 1.0
            else:
                # NO loses - settle at 0.0
                settlement = 0.0
            pnl = (settlement - self.state.entry_price) * self.state.position_size
        
        pnl_dollars = pnl * self.state.balance
        self.state.balance += pnl_dollars
        self.state.total_pnl += pnl_dollars
        
        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1
        
        self.state.trades.append({
            "time": datetime.now(timezone.utc),
            "side": self.state.position_side,
            "pnl": pnl_dollars,
            "btc_move": btc_move,
            "up_won": up_won,
        })
        
        outcome = "UP" if up_won else "DOWN"
        self.file_logger.info(f"SETTLE: {self.state.position_side} -> {outcome} | BTC move: ${btc_move:.2f} | PnL=${pnl_dollars:.2f}")
        console.print(f"\n[{'green' if pnl > 0 else 'red'}]Settlement: {self.state.position_side.upper()} -> {outcome} | PnL=${pnl_dollars:+.2f}[/]")
        
        # Clear position
        self.state.position_side = None
        self.state.position_size = 0.0
        self.state.entry_price = 0.0
    
    def build_display(self) -> Panel:
        """Build display panel."""
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold")
        
        table.add_row("Balance", f"${self.state.balance:.2f}")
        table.add_row("Total PnL", f"${self.state.total_pnl:+.2f}")
        
        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        table.add_row("Win Rate", f"{win_rate:.1%} ({self.state.wins}/{self.state.wins + self.state.losses})")
        
        table.add_row("", "")
        table.add_row("Time Remaining", f"{int(self.get_time_remaining() * 15)}m")
        
        # BTC prices
        if self.state.btc_open_price and self.state.btc_current_price:
            btc_change = (self.state.btc_current_price - self.state.btc_open_price) / self.state.btc_open_price * 100
            btc_color = "green" if btc_change >= 0 else "red"
            table.add_row("BTC Open", f"${self.state.btc_open_price:,.2f}")
            table.add_row("BTC Current", f"${self.state.btc_current_price:,.2f} [{btc_color}]{btc_change:+.3f}%[/]")
        
        table.add_row("", "")
        table.add_row("YES Price", f"{self.state.last_yes_price:.3f}")
        table.add_row("NO Price", f"{self.state.last_no_price:.3f}")
        
        if self.state.position_side:
            table.add_row("", "")
            color = "green" if self.state.position_side == "yes" else "red"
            table.add_row("Position", f"[{color}]{self.state.position_side.upper()}[/] @ {self.state.entry_price:.3f}")
            bet_dollars = self.state.position_size * self.state.balance
            table.add_row("Bet Amount", f"${bet_dollars:.2f} ({self.state.position_size:.0%})")
        
        if self.state.last_probs:
            table.add_row("", "")
            table.add_row("Probs", f"↑{self.state.last_probs.get('up', 0):.0%} ↓{self.state.last_probs.get('down', 0):.0%}")
        
        return Panel(table, title="[bold blue]SAC Dynamic Paper Trading[/bold blue]", border_style="blue")
    
    async def run(self):
        """Main trading loop."""
        self._running = True
        
        # Setup Polymarket client with SOCKS5 proxy
        await self._setup_polymarket_client()
        
        self.load_models()
        
        console.print("\n[bold blue]Starting SAC Dynamic Paper Trading[/bold blue]")
        console.print(f"  SAC Model: {self.config.sac_model}")
        console.print(f"  Balance: ${self.config.initial_balance:.2f}")
        console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
        
        last_candle_ts = None
        
        try:
            with Live(self.build_display(), refresh_per_second=1, console=console) as live:
                while self._running:
                    # Get current candle timestamp
                    current_ts = await self.get_current_candle_timestamp()
                    
                    # Check for new candle
                    if last_candle_ts != current_ts:
                        # Settle previous position
                        if last_candle_ts and self.state.candle_market:
                            await self.settle_position(self.state.candle_market)
                        
                        # Fetch new candle market
                        self.state.candle_market = await self.fetch_polymarket_candle(current_ts)
                        self.state.candle_timestamp = current_ts
                        last_candle_ts = current_ts
                        
                        if self.state.candle_market:
                            console.print(f"\n[blue]New candle: {self.state.candle_market.get('slug')}[/blue]")
                            self.file_logger.info(f"NEW CANDLE: {self.state.candle_market.get('slug')}")

                            # Reset state for new candle
                            self.state.btc_open_price = None
                            self.state.trades_this_candle = 0
                            self.state.yes_price_history = []
                            self.state.no_price_history = []
                    
                    # Fetch BTC data
                    btc_data = await self.fetch_btc_data()
                    
                    # Track BTC prices
                    if len(btc_data) > 0:
                        self.state.btc_current_price = btc_data["price"].iloc[-1]
                        if self.state.btc_open_price is None:
                            self.state.btc_open_price = btc_data["price"].iloc[0]
                    
                    if len(btc_data) > 50 and self.state.candle_market:
                        # Fetch live prices from CLOB on each iteration
                        if not self.state.candle_market.get("closed"):
                            yes_price, no_price = await self.fetch_live_prices(self.state.candle_market)
                        else:
                            yes_price = self.state.candle_market.get("yes_price", 0.5)
                            no_price = self.state.candle_market.get("no_price", 0.5)
                        
                        self.state.last_yes_price = yes_price
                        self.state.last_no_price = no_price
                        
                        try:
                            decision = self.bot.step(
                                btc_data=btc_data,
                                market_yes_price=yes_price,
                                market_spread=abs(yes_price + no_price - 1),
                                time_remaining=self.get_time_remaining(),
                            )
                            
                            probs = np.array([
                                decision.get("prob_down", 0.33),
                                decision.get("prob_hold", 0.34),
                                decision.get("prob_up", 0.33),
                            ])
                            
                            self.state.last_probs = {
                                "up": probs[2],
                                "down": probs[0],
                                "hold": probs[1],
                            }
                            
                            # Get SAC action
                            obs = self.build_observation(probs, yes_price, no_price)
                            sac_decision = self.get_sac_action(obs, probs)
                            
                            exit_signal = sac_decision.get("exit_signal", 0.0)

                            # Check for early exit if we have a position
                            if self.state.position_side:
                                # === DYNAMIC EXIT THRESHOLD ===
                                time_remaining = self.get_time_remaining()

                                # Get current token price for our position
                                if self.state.position_side == "yes":
                                    current_token_price = yes_price
                                else:
                                    current_token_price = no_price

                                # Base threshold
                                dynamic_threshold = 0.5

                                # In convergence zone (price > 0.85), raise exit threshold
                                convergence_zone = 0.85
                                if current_token_price > convergence_zone:
                                    price_factor = (current_token_price - convergence_zone) / (1.0 - convergence_zone)
                                    dynamic_threshold += 0.4 * price_factor
                                elif current_token_price < (1.0 - convergence_zone):
                                    dynamic_threshold -= 0.1

                                # Near settlement with profit, lock exits
                                # (simplified: assume profitable if price moved in our favor)
                                if time_remaining < 0.2:
                                    if self.state.position_side == "yes" and yes_price > self.state.entry_price:
                                        time_lock = (0.2 - time_remaining) / 0.2
                                        dynamic_threshold += 0.3 * time_lock
                                    elif self.state.position_side == "no" and no_price > self.state.entry_price:
                                        time_lock = (0.2 - time_remaining) / 0.2
                                        dynamic_threshold += 0.3 * time_lock

                                if exit_signal > dynamic_threshold:
                                    await self.early_exit(yes_price, no_price, reason="SAC exit_signal")

                            # Execute trade if no position
                            elif not self.state.position_side and sac_decision["action"] != "hold":
                                await self.execute_trade(sac_decision, yes_price, no_price)
                        
                        except Exception as e:
                            logger.debug(f"Prediction error: {e}")
                    
                    live.update(self.build_display())
                    await asyncio.sleep(self.config.update_interval)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
        finally:
            await self.binance_client.aclose()
            if self.polymarket_client:
                await self.polymarket_client.aclose()
            self._print_summary()
    
    def _print_summary(self):
        """Print trading summary."""
        console.print("\n[bold]═══ SESSION SUMMARY ═══[/bold]")
        
        pnl = self.state.balance - self.config.initial_balance
        pnl_pct = pnl / self.config.initial_balance * 100
        
        console.print(f"  Final Balance: ${self.state.balance:.2f}")
        console.print(f"  Total PnL: {'$' + f'{pnl:+.2f}' if pnl >= 0 else '[red]$' + f'{pnl:+.2f}[/red]'} ({pnl_pct:+.1f}%)")
        console.print(f"  Trades: {len(self.state.trades)}")
        console.print(f"  Win Rate: {self.state.wins}/{self.state.wins + self.state.losses}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper trade with SAC Dynamic + Polymarket CLOB")
    parser.add_argument("--sac-model", default="./logs/sac_dynamic_v5/final_model.zip")
    parser.add_argument("--vec-normalize", default="./logs/sac_dynamic_v5/vecnormalize.pkl")
    parser.add_argument("--deep-lob", default="./logs/deep_lob_balanced")
    parser.add_argument("--balance", type=float, default=1000.0)
    
    args = parser.parse_args()
    
    config = PaperTradeConfig(
        sac_model=args.sac_model,
        vec_normalize=args.vec_normalize,
        deep_lob_model=args.deep_lob,
        initial_balance=args.balance,
    )
    
    trader = DynamicPaperTrader(config)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
