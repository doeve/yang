"""
Live Paper Trading for DeepLOB Model.

Simulates Polymarket 15-minute crypto markets using real-time Binance data.
Each 15-minute candle is a separate "market" where we predict Up/Down/Hold.
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from rich.text import Text

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class LiveTradeConfig:
    """Configuration for live paper trading."""
    
    # Trading
    initial_balance: float = 1000.0
    max_position_size: float = 0.25
    fee_percent: float = 0.1
    
    # Candle settings (Polymarket-style)
    candle_minutes: int = 15
    warmup_seconds: int = 120  # Seconds into candle before first trade
    
    deep_lob_model: str = "./logs/deep_lob_balanced"
    sac_model: str = "./logs/sac_cv/fold_1/best/best_model.zip"  # Best CV model (87.7% win rate)
    use_sac: bool = True  # Use SAC model by default now
    aggressive: bool = True
    
    # Data settings
    history_seconds: int = 300  # 5 min history for features
    update_interval: float = 1.0  # Seconds between updates
    
    # Display
    show_live: bool = True
    
    # Logging
    log_file: str = "./logs/paper_trade_live.log"
    verbose_logging: bool = True  # Detailed feature/prediction logs
    
    # === NEW: Continuous Trading Settings ===
    
    # Exit thresholds
    take_profit_pct: float = 0.15   # Exit at +15%
    stop_loss_pct: float = 0.10     # Exit at -10%
    trailing_stop_pct: float = 0.05 # Lock profits with trailing stop
    time_decay_threshold: float = 0.2  # Exit in last 20% if edge weak
    min_edge_to_hold: float = 0.03   # Need 3% edge to maintain position
    
    # Cost model (Almgren-Chriss inspired)
    spread_cost: float = 0.002      # 0.2% bid-ask spread
    slippage_linear: float = 0.001  # Linear slippage
    slippage_quadratic: float = 0.0005  # Quadratic impact
    
    # Trade limits
    max_trades_per_candle: int = 3  # Prevent over-trading
    min_hold_seconds: int = 30      # Minimum time before exit


@dataclass
class Position:
    """Track an open position for continuous trading."""
    
    side: str  # "yes" or "no"
    size: float  # Position size (0-1)
    entry_price: float  # Market price at entry
    entry_time: datetime
    entry_btc_price: float  # BTC price at entry
    
    # Tracking
    max_pnl_pct: float = 0.0  # For trailing stop
    trades_count: int = 1
    
    def unrealized_pnl(self, current_market_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.side == "yes":
            return (current_market_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_market_price) * self.size
    
    def unrealized_pnl_pct(self, current_market_price: float) -> float:
        """Calculate unrealized PnL as percentage of entry."""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl(current_market_price) / (self.entry_price * self.size + 1e-10)
    
    def update_max_pnl(self, current_market_price: float) -> None:
        """Update max PnL for trailing stop."""
        pnl_pct = self.unrealized_pnl_pct(current_market_price)
        if pnl_pct > self.max_pnl_pct:
            self.max_pnl_pct = pnl_pct


@dataclass
class LiveTradeState:
    """Current state of live trading."""
    
    balance: float = 1000.0
    total_pnl: float = 0.0
    
    # Current candle
    candle_start_time: Optional[datetime] = None
    candle_open_price: Optional[float] = None
    
    # Position tracking - now using Position class
    current_position: Optional[Position] = None
    trades_this_candle: int = 0
    
    # Statistics
    trades: List[Dict[str, Any]] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    
    # Exit statistics
    take_profits: int = 0
    stop_losses: int = 0
    signal_reversals: int = 0
    time_decay_exits: int = 0
    
    # Last prediction
    last_prediction: Optional[str] = None
    last_confidence: float = 0.0
    last_probs: Dict[str, float] = field(default_factory=dict)
    last_market_price: float = 0.5



class BinanceLiveDataFetcher:
    """Fetch real-time BTC data from Binance."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10)
        self._last_price: Optional[float] = None
    
    async def get_current_price(self) -> float:
        """Get current BTC/USDT price."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/ticker/price",
                params={"symbol": "BTCUSDT"}
            )
            response.raise_for_status()
            self._last_price = float(response.json()["price"])
            return self._last_price
        except Exception as e:
            logger.warning(f"Error fetching price: {e}")
            return self._last_price or 0.0
    
    async def get_recent_klines(
        self,
        interval: str = "1s",
        limit: int = 300,
    ) -> pd.DataFrame:
        """Get recent klines/candles."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": limit,
                }
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
            df["buy_pressure"] = df["taker_buy_volume"] / (df["volume"] + 1e-10)
            
            return df[["timestamp", "price", "volume", "buy_pressure"]]
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()
    
    async def close(self):
        await self.client.aclose()


class LivePaperTrader:
    """Live paper trading simulation."""
    
    def __init__(self, config: Optional[LiveTradeConfig] = None):
        self.config = config or LiveTradeConfig()
        self.state = LiveTradeState(balance=self.config.initial_balance)
        self.data_fetcher = BinanceLiveDataFetcher()
        self.bot = None
        self.sac_model = None
        self._running = False
        self.file_logger = None
        
        # Setup file logging
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file logging for detailed progress tracking."""
        import logging
        import os
        
        # Create logs directory
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
        
        # Create file logger
        self.file_logger = logging.getLogger("paper_trade_live")
        self.file_logger.setLevel(logging.DEBUG if self.config.verbose_logging else logging.INFO)
        
        # Clear existing handlers
        self.file_logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(self.config.log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        self.file_logger.addHandler(fh)
        
        self.file_logger.info("=" * 80)
        self.file_logger.info("PAPER TRADING SESSION STARTED")
        self.file_logger.info(f"Initial Balance: ${self.config.initial_balance:.2f}")
        self.file_logger.info(f"SAC Model: {self.config.sac_model}")
        self.file_logger.info(f"DeepLOB Model: {self.config.deep_lob_model}")
        self.file_logger.info("=" * 80)
    
    def load_model(self):
        """Load the DeepLOB model and optional SAC model."""
        from src.inference.deep_lob_inference import (
            DeepLOBTwoLayerBot,
            DeepLOBInferenceConfig,
        )
        
        inference_config = DeepLOBInferenceConfig()
        self.bot = DeepLOBTwoLayerBot(config=inference_config)
        self.bot.load_models(
            deep_lob_path=self.config.deep_lob_model,
            sac_path=None,  # We'll use our own SAC loading
        )
        console.print("[green]✓ DeepLOB model loaded[/green]")
        self.file_logger.info(f"DeepLOB model loaded: {self.config.deep_lob_model}")
        
        # Load SAC dynamic model if enabled
        if self.config.use_sac and self.config.sac_model:
            try:
                from stable_baselines3 import SAC
                import os
                if os.path.exists(self.config.sac_model):
                    self.sac_model = SAC.load(self.config.sac_model)
                    console.print(f"[green]✓ SAC model loaded: {self.config.sac_model}[/green]")
                    self.file_logger.info(f"SAC model loaded: {self.config.sac_model}")
                else:
                    console.print(f"[yellow]! SAC model not found: {self.config.sac_model}[/yellow]")
                    self.file_logger.warning(f"SAC model not found: {self.config.sac_model}")
            except Exception as e:
                console.print(f"[yellow]! SAC load error: {e}[/yellow]")
                self.file_logger.error(f"SAC load error: {e}")
    
    def get_candle_boundaries(self) -> tuple[datetime, datetime]:
        """Get current 15-min candle start and end times."""
        now = datetime.utcnow()
        
        # Round down to nearest 15 min
        minutes = (now.minute // self.config.candle_minutes) * self.config.candle_minutes
        candle_start = now.replace(minute=minutes, second=0, microsecond=0)
        candle_end = candle_start + timedelta(minutes=self.config.candle_minutes)
        
        return candle_start, candle_end
    
    def get_time_remaining(self) -> float:
        """Get fraction of time remaining in current candle (0-1)."""
        now = datetime.utcnow()
        _, candle_end = self.get_candle_boundaries()
        
        total_seconds = self.config.candle_minutes * 60
        remaining_seconds = (candle_end - now).total_seconds()
        
        return max(0.0, min(1.0, remaining_seconds / total_seconds))
    
    def get_sac_action(self, deeplob_probs: np.ndarray, market_price: float) -> Dict[str, Any]:
        """Get SAC execution decision from DeepLOB probabilities (15-dim observation for CV model)."""
        prob_down, prob_hold, prob_up = deeplob_probs
        predicted_class = np.argmax(deeplob_probs)
        confidence = max(deeplob_probs)
        
        if self.sac_model is None:
            # Rule-based fallback
            if confidence < 0.40:
                return {"action": "hold", "size": 0.0}
            
            if prob_up > prob_down and prob_up > prob_hold:
                return {"action": "buy_yes", "size": min(confidence * 0.5, 0.25)}
            elif prob_down > prob_up and prob_down > prob_hold:
                return {"action": "buy_no", "size": min(confidence * 0.5, 0.25)}
            else:
                return {"action": "hold", "size": 0.0}
        
        # Build 15-dimension observation for HistoricalTradingEnv (CV model)
        time_remaining = self.get_time_remaining()
        
        # Trade stats
        total_trades = self.state.wins + self.state.losses
        win_rate = self.state.wins / max(1, total_trades)
        
        # Win/loss streaks
        consecutive_wins = min(self.state.wins / 5.0, 1.0) if total_trades > 0 else 0.0
        consecutive_losses = min(self.state.losses / 5.0, 1.0) if total_trades > 0 else 0.0
        
        # Market edge calculation
        model_implied = 0.5 + (prob_up - prob_down) * 0.5
        edge_up = model_implied - market_price
        edge_down = (1 - model_implied) - (1 - market_price)
        
        # Volume normalized (use time as proxy, higher later in candle)
        volume_norm = 1.0 - time_remaining
        
        # Time features
        now = datetime.utcnow()
        hour = now.hour / 24.0
        day = now.weekday() / 7.0
        
        obs = np.array([
            # DeepLOB predictions (3)
            prob_down, prob_hold, prob_up,
            # Prediction features (2)
            predicted_class / 2.0,
            confidence,
            # Market features (3)
            market_price,
            1.0 - market_price,
            volume_norm,
            # Time features (2)
            hour,
            day,
            # Position/history state (3)
            win_rate,
            consecutive_wins,
            consecutive_losses,
            # Edge (2)
            edge_up,
            edge_down,
        ], dtype=np.float32).reshape(1, -1)
        
        # Log detailed observation features
        if self.file_logger and self.config.verbose_logging:
            self.file_logger.debug(f"SAC Observation (15-dim CV model):")
            self.file_logger.debug(f"  DeepLOB: down={prob_down:.3f}, hold={prob_hold:.3f}, up={prob_up:.3f}")
            self.file_logger.debug(f"  Prediction: class={predicted_class}, conf={confidence:.3f}")
            self.file_logger.debug(f"  Market: yes_price={market_price:.3f}")
            self.file_logger.debug(f"  Time: hour={hour:.3f}, day={day:.3f}")
            self.file_logger.debug(f"  History: win_rate={win_rate:.2f}")
            self.file_logger.debug(f"  Edge: up={edge_up:.3f}, down={edge_down:.3f}")
        
        action, _ = self.sac_model.predict(obs, deterministic=True)
        
        # 3-dim action: [direction, size, hold_prob]
        direction = float(action[0][0])
        size = float(np.clip(action[0][1], 0.0, 1.0))
        hold_prob = float(action[0][2])
        
        # Log SAC decision
        if self.file_logger:
            self.file_logger.debug(f"SAC Action: dir={direction:.3f}, size={size:.3f}, hold={hold_prob:.3f}")
        
        # Return SAC decision
        if hold_prob > 0.5 or size < 0.05:
            if self.file_logger:
                self.file_logger.info(f"SAC Decision: HOLD (hold_prob={hold_prob:.3f})")
            return {"action": "hold", "size": 0.0, "exit_signal": 0.0}
        
        if direction > 0.1:
            if self.file_logger:
                self.file_logger.info(f"SAC Decision: BUY_YES size={size * self.config.max_position_size:.3f}")
            return {"action": "buy_yes", "size": size * self.config.max_position_size, "exit_signal": 0.0}
        elif direction < -0.1:
            if self.file_logger:
                self.file_logger.info(f"SAC Decision: BUY_NO size={size * self.config.max_position_size:.3f}")
            return {"action": "buy_no", "size": size * self.config.max_position_size, "exit_signal": 0.0}
        else:
            if self.file_logger:
                self.file_logger.info(f"SAC Decision: HOLD (direction={direction:.3f})")
            return {"action": "hold", "size": 0.0, "exit_signal": 0.0}

    
    def should_exit(
        self,
        position: Position,
        current_market_price: float,
        model_probs: Dict[str, float],
        time_remaining: float,
    ) -> tuple[bool, str]:
        """
        Evaluate if current position should be exited.
        
        Returns (should_exit, reason)
        Reasons: "take_profit", "stop_loss", "trailing_stop", "signal_reversal", "time_decay"
        """
        pnl_pct = position.unrealized_pnl_pct(current_market_price)
        
        # Update max PnL for trailing stop
        position.update_max_pnl(current_market_price)
        
        # 1. Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, "take_profit"
        
        # 2. Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, "stop_loss"
        
        # 3. Trailing stop (lock in gains)
        half_take_profit = self.config.take_profit_pct * 0.5
        if position.max_pnl_pct > half_take_profit:
            drawdown = position.max_pnl_pct - pnl_pct
            if drawdown > self.config.trailing_stop_pct:
                return True, "trailing_stop"
        
        # 4. Signal reversal
        prob_up = model_probs.get("up", 0.33)
        prob_down = model_probs.get("down", 0.33)
        predicted = np.argmax([prob_down, model_probs.get("hold", 0.34), prob_up])
        
        if position.side == "yes" and predicted == 0:  # Was Up, now Down
            return True, "signal_reversal"
        if position.side == "no" and predicted == 2:   # Was Down, now Up
            return True, "signal_reversal"
        
        # 5. Time decay with weak edge
        edge = abs(prob_up - prob_down)
        if time_remaining < self.config.time_decay_threshold:
            if edge < self.config.min_edge_to_hold:
                return True, "time_decay"
        
        return False, ""
    
    async def exit_position(self, reason: str, current_btc_price: float, market_price: float):
        """Exit current position before settlement."""
        if not self.state.current_position:
            return
        
        position = self.state.current_position
        
        # Calculate PnL based on exit at current market price (not settlement)
        pnl = position.unrealized_pnl(market_price) * self.state.balance
        
        # Apply costs
        spread_cost = self.config.spread_cost * position.size
        slippage = self.config.slippage_linear * position.size
        total_costs = (spread_cost + slippage) * self.state.balance
        
        net_pnl = pnl - total_costs
        
        self.state.balance += net_pnl
        self.state.total_pnl += net_pnl
        
        # Track outcome
        if net_pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1
        
        # Track exit type
        if reason == "take_profit":
            self.state.take_profits += 1
        elif reason == "stop_loss":
            self.state.stop_losses += 1
        elif reason == "signal_reversal":
            self.state.signal_reversals += 1
        elif reason == "time_decay":
            self.state.time_decay_exits += 1
        
        trade = {
            "time": datetime.utcnow(),
            "position": position.side,
            "size": position.size,
            "exit_reason": reason,
            "pnl": net_pnl,
            "balance": self.state.balance,
            "prediction": self.state.last_prediction,
            "entry_price": position.entry_price,
            "exit_price": market_price,
        }
        self.state.trades.append(trade)
        
        logger.info(
            "Position exited",
            reason=reason,
            pnl=f"${net_pnl:.2f}",
            balance=f"${self.state.balance:.2f}",
        )
        
        # Detailed file logging
        if self.file_logger:
            self.file_logger.info(f"EXIT: {reason} | Side={position.side} | PnL=${net_pnl:.2f} | Balance=${self.state.balance:.2f}")
            self.file_logger.info(f"  Entry: ${position.entry_price:.3f} -> Exit: ${market_price:.3f}")
            self.file_logger.info(f"  Costs: spread=${spread_cost*self.state.balance:.2f}, slip=${slippage*self.state.balance:.2f}")
            self.file_logger.info(f"  Stats: W={self.state.wins} L={self.state.losses} WinRate={self.state.wins/max(1,self.state.wins+self.state.losses)*100:.1f}%")
        
        # Clear position
        self.state.current_position = None
        self.state.trades_this_candle += 1

    
    async def get_prediction(self, btc_data: pd.DataFrame) -> Dict[str, Any]:
        """Get prediction from model, using SAC for execution if enabled."""
        if self.bot is None or len(btc_data) < 50:
            return {"action": "hold", "size": 0, "predicted_class": "Hold"}
        
        # Get market price (simulate Polymarket YES price based on current move)
        if self.state.candle_open_price:
            current_price = btc_data["price"].iloc[-1]
            move = (current_price - self.state.candle_open_price) / self.state.candle_open_price
            # Convert to implied probability
            market_yes_price = 0.5 + move * 100  # Scale move
            market_yes_price = np.clip(market_yes_price, 0.3, 0.7)
        else:
            market_yes_price = 0.5
        
        # Store the market price
        self.state.last_market_price = market_yes_price
        
        # Get DeepLOB base prediction
        decision = self.bot.step(
            btc_data=btc_data,
            market_yes_price=market_yes_price,
            market_spread=0.02,
            time_remaining=self.get_time_remaining(),
        )
        
        # Use SAC for execution decisions if enabled
        if self.sac_model is not None:
            deeplob_probs = np.array([
                decision.get("prob_down", 0.33),
                decision.get("prob_hold", 0.34),
                decision.get("prob_up", 0.33),
            ])
            sac_decision = self.get_sac_action(deeplob_probs, market_yes_price)
            
            # Merge SAC decision with DeepLOB probs
            decision["action"] = sac_decision["action"]
            decision["size"] = sac_decision["size"]
            decision["exit_signal"] = sac_decision.get("exit_signal", 0.0)
            
            return decision
        
        # Apply aggressive mode (rule-based fallback)
        if self.config.aggressive and decision["action"] == "hold":
            prob_up = decision.get("prob_up", 0.33)
            prob_down = decision.get("prob_down", 0.33)
            diff = prob_up - prob_down
            
            if abs(diff) > 0.03:
                decision["action"] = "buy_yes" if diff > 0 else "buy_no"
                decision["size"] = min(abs(diff) * 2, self.config.max_position_size)
        
        return decision
    
    async def handle_new_candle(self):
        """Handle transition to a new candle."""
        candle_start, _ = self.get_candle_boundaries()
        
        # Settle previous position if exists
        if self.state.current_position and self.state.candle_start_time:
            await self.settle_position()
        
        # Start new candle
        self.state.candle_start_time = candle_start
        self.state.candle_open_price = await self.data_fetcher.get_current_price()
        self.state.current_position = None
        self.state.position_size = 0.0
        
        logger.info(
            "New candle started",
            start=candle_start.isoformat(),
            open_price=self.state.candle_open_price,
        )
        
        # File logging for candle transition
        if self.file_logger:
            self.file_logger.info("-" * 60)
            self.file_logger.info(f"NEW CANDLE: {candle_start.strftime('%H:%M:%S')}")
            self.file_logger.info(f"  BTC Price: ${self.state.candle_open_price:.2f}")
            self.file_logger.info(f"  Balance: ${self.state.balance:.2f} | PnL: ${self.state.total_pnl:.2f}")
            self.file_logger.info("-" * 60)
    
    async def settle_position(self):
        """Settle current position at candle end."""
        if not self.state.current_position or not self.state.candle_open_price:
            return
        
        current_price = await self.data_fetcher.get_current_price()
        move = (current_price - self.state.candle_open_price) / self.state.candle_open_price
        
        # Determine outcome
        if move > 0.0001:
            outcome = "Up"
        elif move < -0.0001:
            outcome = "Down"
        else:
            outcome = "Hold"
        
        # Calculate PnL
        pnl = 0.0
        position_value = self.state.position_size * self.state.balance
        
        if self.state.current_position == "yes":
            # Bet on Up
            if outcome == "Up":
                pnl = position_value * (1 - self.state.entry_price - 0.002)
                self.state.wins += 1
            else:
                pnl = -position_value * self.state.entry_price
                self.state.losses += 1
        else:  # "no"
            # Bet on Down
            if outcome == "Down":
                pnl = position_value * (self.state.entry_price - 0.002)
                self.state.wins += 1
            else:
                pnl = -position_value * (1 - self.state.entry_price)
                self.state.losses += 1
        
        # Apply fees
        pnl -= position_value * self.config.fee_percent * 0.01
        
        self.state.balance += pnl
        self.state.total_pnl += pnl
        
        trade = {
            "time": self.state.candle_start_time,
            "position": self.state.current_position,
            "size": self.state.position_size,
            "outcome": outcome,
            "pnl": pnl,
            "balance": self.state.balance,
            "prediction": self.state.last_prediction,
        }
        self.state.trades.append(trade)
        
        logger.info(
            "Position settled",
            position=self.state.current_position,
            outcome=outcome,
            pnl=f"${pnl:.2f}",
            balance=f"${self.state.balance:.2f}",
        )
    
    async def execute_trade(self, decision: Dict[str, Any], current_btc_price: float = 0.0, market_price: float = 0.5):
        """Execute a trade based on decision using Position class."""
        if self.state.current_position:
            return  # Already have position
        
        action = decision.get("action", "hold")
        if action == "hold":
            return
        
        size = decision.get("size", 0.25)
        side = "yes" if action == "buy_yes" else "no"
        
        # Create Position object
        self.state.current_position = Position(
            side=side,
            size=size,
            entry_price=market_price,
            entry_time=datetime.utcnow(),
            entry_btc_price=current_btc_price,
        )
        
        self.state.last_prediction = decision.get("predicted_class", "Hold")
        self.state.last_confidence = decision.get("confidence", 0.0)
        self.state.last_probs = {
            "up": decision.get("prob_up", 0.33),
            "down": decision.get("prob_down", 0.33),
            "hold": decision.get("prob_hold", 0.34),
        }
        
        logger.info(
            "Trade executed",
            action=action,
            side=side,
            size=size,
            entry_price=f"{market_price:.3f}",
            prediction=self.state.last_prediction,
        )

    
    def build_display(self) -> Panel:
        """Build rich display panel."""
        now = datetime.utcnow()
        candle_start, candle_end = self.get_candle_boundaries()
        time_remaining = self.get_time_remaining()
        
        # Stats table
        stats = Table(show_header=False, box=None)
        stats.add_column("Key", style="dim")
        stats.add_column("Value", style="bold")
        
        stats.add_row("Balance", f"${self.state.balance:.2f}")
        stats.add_row("Total PnL", f"${self.state.total_pnl:+.2f}")
        stats.add_row("Trades", str(len(self.state.trades)))
        
        win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
        stats.add_row("Win Rate", f"{win_rate:.1%} ({self.state.wins}/{self.state.wins + self.state.losses})")
        
        stats.add_row("", "")
        stats.add_row("Candle Start", candle_start.strftime("%H:%M:%S"))
        stats.add_row("Time Remaining", f"{int(time_remaining * 15)}:{int((time_remaining * 15 * 60) % 60):02d}")
        
        if self.state.candle_open_price:
            stats.add_row("Open Price", f"${self.state.candle_open_price:,.2f}")
        
        stats.add_row("", "")
        if self.state.current_position:
            pos = self.state.current_position
            pos_color = "green" if pos.side == "yes" else "red"
            pnl_pct = pos.unrealized_pnl_pct(self.state.last_market_price)
            pnl_color = "green" if pnl_pct > 0 else "red"
            stats.add_row("Position", f"[{pos_color}]{pos.side.upper()}[/] ({pos.size:.0%})")
            stats.add_row("Unrealized", f"[{pnl_color}]{pnl_pct:+.1%}[/]")
        else:
            stats.add_row("Position", "[dim]None[/dim]")
        
        if self.state.last_prediction:
            pred_color = {"Up": "green", "Down": "red", "Hold": "yellow"}.get(self.state.last_prediction, "white")
            stats.add_row("Prediction", f"[{pred_color}]{self.state.last_prediction}[/]")
            stats.add_row("Probs", f"↑{self.state.last_probs.get('up', 0):.0%} ↓{self.state.last_probs.get('down', 0):.0%}")
        
        # Exit stats
        if self.state.take_profits + self.state.stop_losses + self.state.signal_reversals > 0:
            stats.add_row("", "")
            stats.add_row("Exits", f"TP:{self.state.take_profits} SL:{self.state.stop_losses} Rev:{self.state.signal_reversals}")

        
        # Recent trades
        if self.state.trades:
            stats.add_row("", "")
            stats.add_row("[bold]Recent Trades[/bold]", "")
            for trade in self.state.trades[-3:]:
                pnl_color = "green" if trade["pnl"] > 0 else "red"
                # Handle both settlement (outcome) and early exit (exit_reason)
                reason = trade.get("outcome") or trade.get("exit_reason", "exit")
                stats.add_row(
                    trade["time"].strftime("%H:%M"),
                    f"[{pnl_color}]{reason} ${trade['pnl']:+.2f}[/]"
                )

        
        return Panel(stats, title="[bold blue]Live Paper Trading[/bold blue]", border_style="blue")
    
    async def run(self):
        """Main trading loop."""
        self._running = True
        last_candle_start = None
        
        console.print("[bold blue]Starting Live Paper Trading[/bold blue]")
        console.print(f"  Model: {self.config.deep_lob_model}")
        console.print(f"  Initial Balance: ${self.config.initial_balance:.2f}")
        console.print(f"  Candle: {self.config.candle_minutes} minutes")
        console.print()
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        console.print()
        
        try:
            with Live(self.build_display(), refresh_per_second=1, console=console) as live:
                while self._running:
                    candle_start, _ = self.get_candle_boundaries()
                    
                    # Check for new candle
                    if last_candle_start != candle_start:
                        await self.handle_new_candle()
                        last_candle_start = candle_start
                    
                    # Get data and make prediction
                    btc_data = await self.data_fetcher.get_recent_klines(limit=300)
                    
                    if len(btc_data) > 50:
                        time_remaining = self.get_time_remaining()
                        time_in_candle = 1.0 - time_remaining
                        current_btc_price = btc_data["price"].iloc[-1]
                        
                        # Calculate current market price
                        if self.state.candle_open_price:
                            move = (current_btc_price - self.state.candle_open_price) / self.state.candle_open_price
                            market_price = 0.5 + move * 100
                            market_price = np.clip(market_price, 0.1, 0.9)
                        else:
                            market_price = 0.5
                        
                        self.state.last_market_price = market_price
                        
                        # === CONTINUOUS TRADING: Check for exits ===
                        if self.state.current_position:
                            # Get latest prediction for exit decision
                            decision = await self.get_prediction(btc_data)
                            model_probs = {
                                "up": decision.get("prob_up", 0.33),
                                "down": decision.get("prob_down", 0.33),
                                "hold": decision.get("prob_hold", 0.34),
                            }
                            
                            should_exit, exit_reason = self.should_exit(
                                self.state.current_position,
                                market_price,
                                model_probs,
                                time_remaining,
                            )
                            
                            if should_exit:
                                await self.exit_position(exit_reason, current_btc_price, market_price)
                        
                        # === Check for new entries ===
                        warmup_passed = time_in_candle > (self.config.warmup_seconds / (self.config.candle_minutes * 60))
                        
                        if warmup_passed and not self.state.current_position:
                            # Check if we haven't exceeded max trades per candle
                            if self.state.trades_this_candle < self.config.max_trades_per_candle:
                                decision = await self.get_prediction(btc_data)
                                await self.execute_trade(decision, current_btc_price, market_price)
                    
                    # Update display
                    live.update(self.build_display())
                    
                    await asyncio.sleep(self.config.update_interval)

        
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
        finally:
            # Final settlement
            if self.state.current_position:
                await self.settle_position()
            
            await self.data_fetcher.close()
            
            # Print summary
            console.print()
            console.print("[bold]Final Summary[/bold]")
            console.print(f"  Balance: ${self.state.balance:.2f}")
            console.print(f"  Total PnL: ${self.state.total_pnl:+.2f}")
            console.print(f"  Return: {(self.state.balance / self.config.initial_balance - 1) * 100:+.1f}%")
            console.print(f"  Trades: {len(self.state.trades)}")
            if self.state.trades:
                win_rate = self.state.wins / max(1, self.state.wins + self.state.losses)
                console.print(f"  Win Rate: {win_rate:.1%}")
    
    def stop(self):
        """Stop the trading loop."""
        self._running = False


async def run_live_paper_trading(
    deep_lob_model: str = "./logs/deep_lob_balanced",
    initial_balance: float = 1000.0,
    aggressive: bool = True,
):
    """Main entry point for live paper trading."""
    config = LiveTradeConfig(
        deep_lob_model=deep_lob_model,
        initial_balance=initial_balance,
        aggressive=aggressive,
    )
    
    trader = LivePaperTrader(config)
    trader.load_model()
    
    # Handle interrupt
    def signal_handler(sig, frame):
        trader.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    await trader.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live paper trading")
    parser.add_argument("--model", default="./logs/deep_lob_balanced")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--aggressive", action="store_true", default=True)
    
    args = parser.parse_args()
    
    asyncio.run(run_live_paper_trading(
        deep_lob_model=args.model,
        initial_balance=args.balance,
        aggressive=args.aggressive,
    ))
