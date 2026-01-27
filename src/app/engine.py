"""
Core Trading Engine.
"""
import asyncio
import logging
from typing import Optional, Dict

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

    async def _handle_signal(self, market_id: str, prediction: Dict, snapshot: Dict):
        """Interpret signal and execute."""
        # 0. Check Max Daily Loss
        daily_pnl = self.get_todays_pnl()
        if daily_pnl < -self.config.risk.max_daily_loss_usd:
            logger.warning(f"Max Daily Loss hit: {daily_pnl:.2f} < -{self.config.risk.max_daily_loss_usd}")
            return # STOP TRADING

        action = prediction["action"]
        conf = prediction["confidence"]
        
        # Get current pos
        pos = await self.execution.get_position(market_id)
        
        # Rules
        # 1. EXIT
        if action == "EXIT" and pos:
            await self.execution.close_position(market_id)
            return

        # 2. ENTRY
        if action in ["BUY_YES", "BUY_NO"] and not pos:
            # Check thresholds (simple safety)
            if conf < 0.3: # Hardcoded safety floor
                return
                
            side = "yes" if action == "BUY_YES" else "no"
            price = snapshot["yes_price"] if side == "yes" else snapshot["no_price"]
            
            # Size calc - User said "bot chooses", so let's try to infer or use default
            # If we removed Max Pos from UI, we need a default or dynamic size
            # Let's use 10% of balance or 100 USD default if config is missing
            balance = await self.execution.get_balance()
            
            # Dynamic sizing: 10% of current balance
            size_usd = balance * 0.10
            
            # Cap size at balance
            if size_usd > balance:
                size_usd = balance
                
            size = size_usd / price if price > 0 else 0
                
            if size > 0:
                await self.execution.execute_order(market_id, side, size, price)

    # Accessors for UI
    def get_state(self):
        """Return full state for UI."""
        return {
            "market": self.data_service.get_snapshot(),
            "prediction": self.last_prediction,
            "running": self.running
        }
