"""
Execution Adapters.
"""
import abc
import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from src.app.config import AppConfig

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str  # e.g. "btc-15m-UUID" or "YES"
    side: str   # "yes" or "no"
    size: float
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    ticks_held: float = 0.0
    max_pnl: float = 0.0

class ExecutionAdapter(abc.ABC):
    """Base interface for executing trades."""
    
    @abc.abstractmethod
    async def get_balance(self) -> float:
        pass
        
    @abc.abstractmethod
    async def get_position(self, market_id: str) -> Optional[Position]:
        pass
        
    @abc.abstractmethod
    async def execute_order(self, market_id: str, side: str, size: float, price_limit: float = None):
        """Execute BUY order (entry)."""
        pass
        
    @abc.abstractmethod
    async def get_trade_history(self) -> list:
        """Return list of past trades."""
        pass

class PaperAdapter(ExecutionAdapter):
    """Paper trading simulation."""
    
    def __init__(self, config: AppConfig):
        self.balance = 1000.0  # Default paper balance
        self.positions: Dict[str, Position] = {} # market_id -> Position
        self.trades = [] # List of closed trades
        self.config = config
        
    async def get_balance(self) -> float:
        return self.balance
        
    async def get_position(self, market_id: str) -> Optional[Position]:
        return self.positions.get(market_id)
        
    async def get_trade_history(self) -> list:
        return self.trades

    async def execute_order(self, market_id: str, side: str, size: float, price_limit: float = None):
        # In paper mode, we assume immediate fill at current price if not provided
        # The engine usually passes the current market price as 'limit' for Paper Adapter consistency
        price = price_limit or 0.5
        
        cost = size * price
        if cost > self.balance:
            logger.warning("Paper: Insufficient funds")
            return
            
        self.balance -= cost
        self.positions[market_id] = Position(
            symbol=market_id,
            side=side,
            size=size,
            entry_price=price,
            current_price=price,
            pnl=0.0
        )
        logger.info(f"Paper BUY {side} {size} @ {price}")
        
    async def close_position(self, market_id: str):
        pos = self.positions.get(market_id)
        if not pos:
            return
            
        # PnL calc depends on current price updates which happens in Engine loop
        # Here we just realize whatever the current_price IS
        proceeds = pos.size * pos.current_price
        pnl = proceeds - (pos.size * pos.entry_price)
        self.balance += proceeds
        
        # Log trade
        from datetime import datetime
        self.trades.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "market": market_id,
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": pos.current_price,
            "pnl": pnl
        })
        
        logger.info(f"Paper SELL {pos.side} {pos.size} @ {pos.current_price} PnL={pnl:.2f}")
        del self.positions[market_id]

    @dataclass
    class Position:
        symbol: str  # e.g. "btc-15m-UUID" or "YES"
        side: str   # "yes" or "no"
        size: float
        entry_price: float
        current_price: float = 0.0
        pnl: float = 0.0
        ticks_held: float = 0.0 # Changed to float to allow fractional accumulation
        max_pnl: float = 0.0

    # ... (in PaperAdapter)

    def update_position_price(self, market_id: str, new_price: float):
        """Helper for Engine to update tracking."""
        if market_id in self.positions:
            pos = self.positions[market_id]
            pos.current_price = new_price
            
            # Increment ticks: Paper trade script counted 1 tick per 5 seconds.
            # App runs at 1Hz (1 tick per 1 second).
            # So we increment by 0.2 to match the RATE of accumulation.
            # The model will see int(ticks_held) or float, but scaling remains consistent.
            pos.ticks_held += 0.2
            
            # Calc PnL
            cost = pos.size * pos.entry_price
            val = pos.size * new_price
            pos.pnl = val - cost
            pos.max_pnl = max(pos.max_pnl, (new_price - pos.entry_price)/pos.entry_price)

class PolymarketAdapter(ExecutionAdapter):
    """Real execution via CLOB API."""
    def __init__(self, config: AppConfig):
        self.config = config
        # Placeholder for real client init
        pass
        
    async def get_balance(self) -> float:
        # TODO: Implement real balance fetch
        return 0.0
        
    async def get_position(self, market_id: str) -> Optional[Position]:
        return None
        
    async def execute_order(self, market_id: str, side: str, size: float, price_limit: float = None):
        logger.warning("Real execution not yet fully implemented")
        
    async def close_position(self, market_id: str):
        pass
