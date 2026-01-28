"""
Core Trading Engine.
"""
import asyncio
import logging
from typing import Optional, Dict
import numpy as np

from src.app.config import AppConfig
from src.app.data import MarketDataService
from src.app.execution import ExecutionAdapter, PaperAdapter, PolymarketAdapter
from src.app.models import ModelFactory, Predictor

logger = logging.getLogger(__name__)

class TradingEngine:
    """Orchestrates Data -> Model -> Execution."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Modules
        self.data_service = MarketDataService(config)
        self.model: Optional[Predictor] = None
        self.execution: Optional[ExecutionAdapter] = None
        
        # State
        self.running = False
        self.last_prediction = {}
        
    async def start(self):
        """Initialize all components."""
        logger.info("Engine starting...")
        
        # 1. Execution
        if self.config.trading_mode == "paper":
            self.execution = PaperAdapter(self.config)
        else:
            self.execution = PolymarketAdapter(self.config)
            
        # 2. Data
        await self.data_service.start()
        
        # 3. Model
        try:
            self.model = ModelFactory.load_model(self.config.model_path)
            logger.info("Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
        self.running = True
        
    async def stop(self):
        self.running = False
        await self.data_service.stop()
        
    async def tick(self):
        """Single tick of logic."""
        if not self.running:
            return
            
        # 1. Update Data
        await self.data_service.refresh()
        snapshot = self.data_service.get_snapshot()
        
        market_slug = snapshot.get("market_slug")
        if not market_slug:
            return # No active market
            
        # 2. Update Position State in Execution (Paper only mostly)
        # We need to tell the adapter current prices to update PnL
        if isinstance(self.execution, PaperAdapter):
            price = snapshot.get("yes_price")
            # If we hold NO, price is 1-price really, but let's handle side logic in adapter or here
            # Logic: If Position is "yes", price is yes_price. If "no", price is no_price.
            
            # This is sloppy, let's just pass YES price and let logic handle it?
            # Actually simplest: Just update generic price mapping if possible
            # For now, let's assume single market focus
            
            pos = await self.execution.get_position(market_slug)
            if pos:
                current_p = snapshot["yes_price"] if pos.side == "yes" else snapshot["no_price"]
                self.execution.update_position_price(market_slug, current_p)
        
        # 3. Model Inference
        # We need to inject position state into snapshot for the model
        pos = await self.execution.get_position(market_slug)
        if pos:
            snapshot["position"] = {
                "side": pos.side,
                "entry_price": pos.entry_price,
                "ticks_held": pos.ticks_held,
                "max_pnl": pos.max_pnl
            }
        else:
            snapshot["position"] = {}
            
        if self.model:
            prediction = self.model.predict(snapshot)
            self.last_prediction = prediction
            
            # 4. Trading Logic
            await self._handle_signal(market_slug, prediction, snapshot)
            
    def get_todays_pnl(self):
        """Calculate realized PnL for today (simplified)."""
        # In a real app we'd track trade history with timestamps
        # Here we just use execution adapter's 'balance' change since start logic
        # For Paper, simple. For Real, adapter needs to provide this.
        # Let's assume PaperAdapter tracks cumulative PnL or we track it here.
        if isinstance(self.execution, PaperAdapter):
            return self.execution.balance - 1000.0 # Hack: assuming 1000 start
        return 0.0

    
    def calculate_position_size(self, expected_return: float, confidence: float) -> float:
        """Calculate position size based on model outputs."""
        # Base size scales linearly with expected return
        # 0.5x to 2x based on return relative to 10% benchmark
        return_factor = np.clip(expected_return / 0.10, 0.5, 2.0)
        
        # Confidence factor: 0.5x to 1x
        conf_factor = 0.5 + 0.5 * confidence
        
        size = self.config.strategy.base_position_size * return_factor * conf_factor
        
        return float(np.clip(size, self.config.strategy.min_position_size, self.config.strategy.max_position_size))

    async def _handle_signal(self, market_id: str, prediction: Dict, snapshot: Dict):
        """Interpret signal and execute."""
        # 0. Check Max Daily Loss (Real mode only)
        daily_pnl = self.get_todays_pnl()
        
        if self.config.trading_mode == "real":
            balance = await self.execution.get_balance()
            if balance > 0:
                pnl_pct = (daily_pnl / balance) * 100
                if pnl_pct < -self.config.risk.max_daily_loss_pct:
                    logger.warning(f"Max Daily Loss hit: {pnl_pct:.2f}% < -{self.config.risk.max_daily_loss_pct}%")
                    return # STOP TRADING

        action = prediction["action"]
        conf = prediction["confidence"]
        expected_return = prediction.get("expected_return", 0.0)
        time_remaining = snapshot.get("time_remaining", 0.0)
        
        # Get current pos
        pos = await self.execution.get_position(market_id)
        
        # Check Expiry/Settlement
        if time_remaining == 0 and pos:
             logger.info(f"Market Expired: Forcing Close for {market_id}")
             await self.execution.close_position(market_id)
             return

        # 1. EXIT
        if action == "EXIT" and pos:
            # We could add logic here: only exit if expected_return is low?
            # But "EXIT" usually means the model sees negative value.
            await self.execution.close_position(market_id)
            return

        # 2. ENTRY
        if action in ["BUY_YES", "BUY_NO"] and not pos:
            # Safety Filters
            if conf < self.config.strategy.min_confidence:
                return
            if expected_return < self.config.strategy.min_expected_return:
                return
            if time_remaining < self.config.strategy.min_time_remaining:
                return
                
            side = "yes" if action == "BUY_YES" else "no"
            price = snapshot["yes_price"] if side == "yes" else snapshot["no_price"]
            
            # Dynamic Sizing
            pct_size = self.calculate_position_size(expected_return, conf)
            
            balance = await self.execution.get_balance()
            size_usd = balance * pct_size
            
            # Cap at balance
            if size_usd > balance:
                size_usd = balance
                
            size = size_usd / price if price > 0 else 0
                
            if size > 0:
                logger.info(f"ENTRY: {side.upper()} @ {price:.3f} | Size={pct_size:.1%} (${size_usd:.0f}) | E[R]={expected_return:.1%}")
                await self.execution.execute_order(market_id, side, size, price)

        # Log heartbeat
        if action in ["WAIT", "HOLD"]:
            # logger.info(f"Model Signal: {action} | Conf: {conf:.2f}")
            pass


    # Accessors for UI
    def get_state(self):
        """Return full state for UI."""
        return {
            "market": self.data_service.get_snapshot(),
            "prediction": self.last_prediction,
            "running": self.running
        }
