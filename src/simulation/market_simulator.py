"""
Market simulator for order execution.

Simulates realistic trading mechanics:
- Execution latency
- Slippage based on order size
- Transaction costs (fees + spread)
- Position limits
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """A trading order."""
    id: str
    side: OrderSide
    order_type: OrderType
    size: float  # Quantity in shares/contracts
    price: float | None = None  # Limit price (None for market orders)
    
    # Execution details (filled after execution)
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_price: float = 0.0
    filled_at: datetime | None = None
    
    # Cost breakdown
    slippage: float = 0.0
    fees: float = 0.0
    total_cost: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position in a market."""
    market_id: str
    size: float = 0.0  # Positive = long, negative = short
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    peak_value: float = 0.0


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    
    # Performance tracking
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    
    # Risk metrics
    max_drawdown: float = 0.0
    peak_value: float = 0.0
    
    def get_position(self, market_id: str) -> Position:
        """Get or create position for a market."""
        if market_id not in self.positions:
            self.positions[market_id] = Position(market_id=market_id)
        return self.positions[market_id]
    
    def total_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value."""
        value = self.cash
        for market_id, position in self.positions.items():
            if market_id in current_prices:
                value += position.size * current_prices[market_id]
        return value


@dataclass
class SimulatorConfig:
    """Configuration for market simulator."""
    
    # Initial portfolio
    initial_cash: float = 10000.0
    
    # Execution settings
    execution_delay_steps: int = 1  # Steps between order and execution
    
    # Slippage model (as percentage of price)
    slippage_model: str = "linear"  # "linear", "sqrt", "fixed"
    slippage_factor: float = 0.0005  # 0.05% base slippage
    slippage_volume_factor: float = 0.00001  # Additional slippage per unit
    
    # Transaction costs
    fee_rate: float = 0.001  # 0.1% fee
    min_fee: float = 0.01  # Minimum fee per trade
    
    # Position limits
    max_position_pct: float = 0.5  # Max 50% of portfolio in one market
    min_order_size: float = 0.0001  # Minimum order size (for fractional crypto)
    
    # Market constraints (tick as percentage of price for generality)
    price_tick: float = 0.01  # Minimum price increment ($0.01)
    
    # Risk management
    stop_loss_pct: float | None = None  # Optional stop-loss percentage
    take_profit_pct: float | None = None  # Optional take-profit percentage


class MarketSimulator:
    """
    Simulates market execution with realistic constraints.
    
    Features:
    - Order queuing with execution delay
    - Slippage modeling based on order size
    - Transaction fees
    - Position tracking with PnL calculation
    - Risk management (position limits, stop-loss)
    
    Example:
        simulator = MarketSimulator(config)
        simulator.reset()
        
        # Submit order
        order = simulator.submit_order(OrderSide.BUY, 100)
        
        # Process pending orders
        simulator.process_orders(current_price, current_step)
        
        # Get portfolio state
        portfolio = simulator.portfolio
    """
    
    def __init__(self, config: SimulatorConfig | None = None):
        """
        Initialize simulator.
        
        Args:
            config: Simulator configuration
        """
        self.config = config or SimulatorConfig()
        
        # State
        self.portfolio: Portfolio = Portfolio(cash=self.config.initial_cash)
        self._pending_orders: list[tuple[int, Order]] = []  # (execute_at_step, order)
        self._order_history: list[Order] = []
        self._current_step: int = 0
        self._order_counter: int = 0
        
        # Price tracking for PnL
        self._current_prices: dict[str, float] = {}
        
        logger.info("MarketSimulator initialized", config=self.config)
    
    def reset(self) -> Portfolio:
        """Reset simulator to initial state."""
        self.portfolio = Portfolio(cash=self.config.initial_cash)
        self._pending_orders = []
        self._order_history = []
        self._current_step = 0
        self._order_counter = 0
        self._current_prices = {}
        
        return self.portfolio
    
    def submit_order(
        self,
        market_id: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Order | None:
        """
        Submit a trading order.
        
        Args:
            market_id: Market to trade
            side: Buy or sell
            size: Order size (positive)
            order_type: Market or limit
            price: Limit price (required for limit orders)
            metadata: Optional order metadata
            
        Returns:
            Order object, or None if rejected
        """
        # Validate size
        if size < self.config.min_order_size:
            logger.warning("Order rejected: size too small", size=size)
            return None
        
        # Check position limits
        if not self._check_position_limits(market_id, side, size):
            logger.warning("Order rejected: position limit exceeded")
            return None
        
        # Create order
        self._order_counter += 1
        order = Order(
            id=f"order_{self._order_counter}",
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            metadata=metadata or {"market_id": market_id},
        )
        order.metadata["market_id"] = market_id
        
        # Queue for execution with delay
        execute_at = self._current_step + self.config.execution_delay_steps
        self._pending_orders.append((execute_at, order))
        
        logger.debug(
            "Order submitted",
            order_id=order.id,
            side=side.value,
            size=size,
            execute_at=execute_at,
        )
        
        return order
    
    def submit_close_position(self, market_id: str) -> Order | None:
        """Submit order to close entire position in a market."""
        position = self.portfolio.get_position(market_id)
        
        if abs(position.size) < self.config.min_order_size:
            return None  # No position to close
        
        side = OrderSide.SELL if position.size > 0 else OrderSide.BUY
        size = abs(position.size)
        
        return self.submit_order(market_id, side, size)
    
    def process_orders(
        self,
        current_price: float,
        market_id: str,
        current_step: int,
    ) -> list[Order]:
        """
        Process pending orders that are ready for execution.
        
        Args:
            current_price: Current market price
            market_id: Market being processed
            current_step: Current simulation step
            
        Returns:
            List of orders that were executed
        """
        self._current_step = current_step
        self._current_prices[market_id] = current_price
        
        executed = []
        remaining = []
        
        for execute_at, order in self._pending_orders:
            if order.metadata.get("market_id") != market_id:
                remaining.append((execute_at, order))
                continue
            
            if current_step >= execute_at:
                # Execute the order
                self._execute_order(order, current_price, market_id)
                executed.append(order)
                self._order_history.append(order)
            else:
                remaining.append((execute_at, order))
        
        self._pending_orders = remaining
        
        # Update unrealized PnL for all positions
        self._update_unrealized_pnl()
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        return executed
    
    def _execute_order(
        self,
        order: Order,
        current_price: float,
        market_id: str,
    ) -> None:
        """Execute a single order."""
        # Calculate slippage
        slippage = self._calculate_slippage(order.size, current_price)
        
        # Adjust price based on side
        if order.side == OrderSide.BUY:
            execution_price = current_price + slippage
        else:
            execution_price = current_price - slippage
        
        # Round to tick size
        execution_price = round(
            execution_price / self.config.price_tick
        ) * self.config.price_tick
        
        # Ensure price is positive
        execution_price = max(self.config.price_tick, execution_price)
        
        # Calculate costs
        trade_value = order.size * execution_price
        fees = max(self.config.min_fee, trade_value * self.config.fee_rate)
        
        # Check if we have enough cash for buys
        if order.side == OrderSide.BUY:
            total_cost = trade_value + fees
            if total_cost > self.portfolio.cash:
                order.status = OrderStatus.REJECTED
                logger.warning("Order rejected: insufficient cash")
                return
        
        # Update position
        position = self.portfolio.get_position(market_id)
        old_size = position.size
        
        if order.side == OrderSide.BUY:
            # Update average entry price
            if position.size >= 0:
                # Adding to long or opening long
                total_value = position.size * position.avg_entry_price + order.size * execution_price
                position.size += order.size
                if position.size > 0:
                    position.avg_entry_price = total_value / position.size
            else:
                # Reducing short position
                closed_size = min(abs(position.size), order.size)
                pnl = closed_size * (position.avg_entry_price - execution_price)
                position.realized_pnl += pnl
                position.size += order.size
                
                if position.size > 0:
                    # Flipped to long
                    position.avg_entry_price = execution_price
            
            self.portfolio.cash -= trade_value + fees
        else:  # SELL
            if position.size <= 0:
                # Adding to short or opening short
                total_value = abs(position.size) * position.avg_entry_price + order.size * execution_price
                position.size -= order.size
                if position.size < 0:
                    position.avg_entry_price = total_value / abs(position.size)
            else:
                # Reducing long position
                closed_size = min(position.size, order.size)
                pnl = closed_size * (execution_price - position.avg_entry_price)
                position.realized_pnl += pnl
                position.size -= order.size
                
                if position.size < 0:
                    # Flipped to short
                    position.avg_entry_price = execution_price
            
            self.portfolio.cash += trade_value - fees
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.filled_price = execution_price
        order.filled_at = datetime.now(timezone.utc)
        order.slippage = slippage
        order.fees = fees
        order.total_cost = trade_value + fees if order.side == OrderSide.BUY else trade_value - fees
        
        # Update portfolio stats
        self.portfolio.total_fees += fees
        self.portfolio.total_trades += 1
        
        # Check if winning trade (only for closes)
        if old_size != 0 and (
            (old_size > 0 and position.size < old_size) or
            (old_size < 0 and position.size > old_size)
        ):
            if position.realized_pnl > 0:
                self.portfolio.winning_trades += 1
        
        logger.debug(
            "Order executed",
            order_id=order.id,
            side=order.side.value,
            size=order.size,
            price=execution_price,
            slippage=slippage,
            fees=fees,
        )
    
    def _calculate_slippage(self, size: float, price: float) -> float:
        """Calculate slippage based on order size."""
        base_slippage = self.config.slippage_factor * price
        
        if self.config.slippage_model == "linear":
            volume_slippage = self.config.slippage_volume_factor * size * price
        elif self.config.slippage_model == "sqrt":
            volume_slippage = self.config.slippage_volume_factor * np.sqrt(size) * price
        else:  # fixed
            volume_slippage = 0
        
        return base_slippage + volume_slippage
    
    def _check_position_limits(
        self,
        market_id: str,
        side: OrderSide,
        size: float,
    ) -> bool:
        """Check if order would exceed position limits."""
        # Calculate portfolio value
        total_value = self.portfolio.total_value(self._current_prices)
        max_position_value = total_value * self.config.max_position_pct
        
        # Current position
        position = self.portfolio.get_position(market_id)
        current_price = self._current_prices.get(market_id, 0.5)
        
        # Projected position after trade
        if side == OrderSide.BUY:
            new_size = position.size + size
        else:
            new_size = position.size - size
        
        projected_value = abs(new_size) * current_price
        
        return projected_value <= max_position_value
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL for all positions."""
        for market_id, position in self.portfolio.positions.items():
            if market_id in self._current_prices:
                current_price = self._current_prices[market_id]
                if position.size > 0:
                    position.unrealized_pnl = position.size * (current_price - position.avg_entry_price)
                elif position.size < 0:
                    position.unrealized_pnl = abs(position.size) * (position.avg_entry_price - current_price)
                else:
                    position.unrealized_pnl = 0
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics."""
        # Total PnL
        realized = sum(p.realized_pnl for p in self.portfolio.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.portfolio.positions.values())
        self.portfolio.total_pnl = realized + unrealized
        
        # Max drawdown
        current_value = self.portfolio.total_value(self._current_prices)
        if current_value > self.portfolio.peak_value:
            self.portfolio.peak_value = current_value
        
        if self.portfolio.peak_value > 0:
            drawdown = (self.portfolio.peak_value - current_value) / self.portfolio.peak_value
            if drawdown > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = drawdown
    
    def get_state(self, market_id: str) -> dict[str, float]:
        """
        Get normalized state for model input.
        
        Returns dict with:
        - position_normalized: Current position as fraction of max
        - unrealized_pnl_normalized: Unrealized PnL normalized by initial cash
        - cash_ratio: Available cash ratio
        - exposure: Total market exposure
        """
        position = self.portfolio.get_position(market_id)
        current_price = self._current_prices.get(market_id, 0.5)
        total_value = self.portfolio.total_value(self._current_prices)
        max_position_value = total_value * self.config.max_position_pct
        
        position_value = position.size * current_price
        
        return {
            "position_normalized": position_value / max_position_value if max_position_value > 0 else 0,
            "unrealized_pnl_normalized": position.unrealized_pnl / self.config.initial_cash,
            "cash_ratio": self.portfolio.cash / self.config.initial_cash,
            "exposure": abs(position_value) / total_value if total_value > 0 else 0,
        }
    
    def settle_market(
        self,
        market_id: str,
        outcome: bool,
        settlement_price: float,
    ) -> float:
        """
        Settle a market at resolution.
        
        Args:
            market_id: Market that resolved
            outcome: True = Yes wins, False = No wins
            settlement_price: Final settlement price (1.0 for Yes, 0.0 for No)
            
        Returns:
            Settlement PnL
        """
        position = self.portfolio.get_position(market_id)
        
        if abs(position.size) < 0.0001:
            return 0.0
        
        # Calculate settlement value
        settlement_value = position.size * settlement_price
        
        # PnL from settlement
        entry_value = position.size * position.avg_entry_price
        pnl = settlement_value - entry_value
        
        position.realized_pnl += pnl
        position.unrealized_pnl = 0
        position.size = 0
        
        # Add settlement proceeds to cash
        self.portfolio.cash += settlement_value
        
        logger.info(
            "Market settled",
            market_id=market_id,
            outcome=outcome,
            settlement_price=settlement_price,
            pnl=pnl,
        )
        
        return pnl
