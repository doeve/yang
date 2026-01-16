"""
Live trading adapter for deploying trained models.

Provides an abstraction layer between trained models and live trading:
- Abstract interface for different exchanges/markets
- Paper trading implementation for testing
- Polymarket CLOB adapter (placeholder)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
import asyncio

import numpy as np
import structlog

from ..models.agent import TradingAgent
from ..data.preprocessor import FeaturePreprocessor, FeatureConfig

logger = structlog.get_logger(__name__)


@dataclass
class MarketState:
    """Current market state for live trading."""
    market_id: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: datetime
    
    # Additional context
    time_to_resolution: float = 1.0
    metadata: dict[str, Any] | None = None


@dataclass
class Order:
    """Live trading order."""
    id: str
    market_id: str
    side: Literal["buy", "sell"]
    size: float
    order_type: Literal["market", "limit"]
    price: float | None = None  # For limit orders
    
    # Status tracking
    status: Literal["pending", "filled", "cancelled", "rejected"] = "pending"
    filled_size: float = 0.0
    filled_price: float = 0.0
    
    created_at: datetime | None = None
    filled_at: datetime | None = None


@dataclass
class Position:
    """Current position in a market."""
    market_id: str
    size: float
    avg_entry_price: float
    unrealized_pnl: float
    current_price: float


class LiveTradingAdapter(ABC):
    """
    Abstract interface for live trading execution.
    
    Implementations can target:
    - Polymarket CLOB API (production)
    - Paper trading (testing)
    - Other prediction markets
    
    Example:
        adapter = PolymarketAdapter(api_key)
        
        # Get market state
        state = await adapter.get_market_state("btc-100k")
        
        # Place order
        order = await adapter.place_order(
            market_id="btc-100k",
            side="buy",
            size=100,
        )
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to trading venue."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass
    
    @abstractmethod
    async def get_market_state(self, market_id: str) -> MarketState:
        """Get current market state."""
        pass
    
    @abstractmethod
    async def get_position(self, market_id: str) -> Position | None:
        """Get current position in a market."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        market_id: str,
        side: Literal["buy", "sell"],
        size: float,
        order_type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> Order:
        """Place a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get status of an order."""
        pass


class PaperTradingAdapter(LiveTradingAdapter):
    """
    Paper trading adapter for testing strategies.
    
    Simulates order execution with configurable:
    - Execution delay
    - Slippage
    - Fees
    
    Uses real market data but doesn't execute real trades.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        slippage: float = 0.001,
        fee_rate: float = 0.001,
        execution_delay_seconds: float = 1.0,
    ):
        """
        Args:
            initial_balance: Starting paper balance
            slippage: Simulated slippage (0.001 = 0.1%)
            fee_rate: Trading fee rate
            execution_delay_seconds: Simulated execution delay
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.slippage = slippage
        self.fee_rate = fee_rate
        self.execution_delay = execution_delay_seconds
        
        # State
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self._order_counter = 0
        
        # Mock price data
        self._mock_prices: dict[str, float] = {}
        
        self.is_connected = False
    
    async def connect(self) -> None:
        """Simulate connection."""
        self.is_connected = True
        logger.info("Paper trading connected")
    
    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self.is_connected = False
        logger.info("Paper trading disconnected")
    
    def set_mock_price(self, market_id: str, price: float) -> None:
        """Set mock price for a market (for testing)."""
        self._mock_prices[market_id] = price
    
    async def get_market_state(self, market_id: str) -> MarketState:
        """Get simulated market state."""
        price = self._mock_prices.get(market_id, 0.5)
        spread = 0.005  # 0.5% spread
        
        return MarketState(
            market_id=market_id,
            price=price,
            bid=price - spread / 2,
            ask=price + spread / 2,
            volume_24h=100000.0,
            timestamp=datetime.now(timezone.utc),
        )
    
    async def get_position(self, market_id: str) -> Position | None:
        """Get current paper position."""
        return self.positions.get(market_id)
    
    async def get_balance(self) -> float:
        """Get paper balance."""
        return self.balance
    
    async def place_order(
        self,
        market_id: str,
        side: Literal["buy", "sell"],
        size: float,
        order_type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> Order:
        """Simulate order placement and execution."""
        self._order_counter += 1
        order_id = f"paper_{self._order_counter}"
        
        order = Order(
            id=order_id,
            market_id=market_id,
            side=side,
            size=size,
            order_type=order_type,
            price=price,
            created_at=datetime.now(timezone.utc),
        )
        
        self.orders[order_id] = order
        
        # Simulate execution delay
        await asyncio.sleep(self.execution_delay)
        
        # Execute the order
        await self._execute_order(order)
        
        return order
    
    async def _execute_order(self, order: Order) -> None:
        """Simulate order execution."""
        market_price = self._mock_prices.get(order.market_id, 0.5)
        
        # Calculate execution price with slippage
        if order.side == "buy":
            execution_price = market_price * (1 + self.slippage)
        else:
            execution_price = market_price * (1 - self.slippage)
        
        # Calculate trade value and fees
        trade_value = order.size * execution_price
        fees = trade_value * self.fee_rate
        
        # Check balance for buys
        if order.side == "buy":
            total_cost = trade_value + fees
            if total_cost > self.balance:
                order.status = "rejected"
                logger.warning("Order rejected: insufficient balance")
                return
            self.balance -= total_cost
        else:
            self.balance += trade_value - fees
        
        # Update position
        position = self.positions.get(order.market_id)
        
        if position is None:
            if order.side == "buy":
                self.positions[order.market_id] = Position(
                    market_id=order.market_id,
                    size=order.size,
                    avg_entry_price=execution_price,
                    unrealized_pnl=0.0,
                    current_price=market_price,
                )
            else:
                self.positions[order.market_id] = Position(
                    market_id=order.market_id,
                    size=-order.size,
                    avg_entry_price=execution_price,
                    unrealized_pnl=0.0,
                    current_price=market_price,
                )
        else:
            if order.side == "buy":
                new_size = position.size + order.size
            else:
                new_size = position.size - order.size
            
            if abs(new_size) < 0.0001:
                del self.positions[order.market_id]
            else:
                position.size = new_size
                position.current_price = market_price
        
        # Update order status
        order.status = "filled"
        order.filled_size = order.size
        order.filled_price = execution_price
        order.filled_at = datetime.now(timezone.utc)
        
        logger.info(
            "Paper order executed",
            order_id=order.id,
            side=order.side,
            size=order.size,
            price=execution_price,
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == "pending":
                order.status = "cancelled"
                return True
        return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get paper order status."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]


class LiveTradingBot:
    """
    Bot that runs a trained agent in live trading mode.
    
    Handles:
    - Real-time market data processing
    - Feature computation
    - Action generation
    - Order execution
    
    Example:
        bot = LiveTradingBot(agent, adapter)
        await bot.run(market_id="btc-100k", duration_hours=24)
    """
    
    def __init__(
        self,
        agent: TradingAgent,
        adapter: LiveTradingAdapter,
        feature_config: FeatureConfig | None = None,
    ):
        """
        Args:
            agent: Trained trading agent
            adapter: Live trading adapter
            feature_config: Feature preprocessing config
        """
        self.agent = agent
        self.adapter = adapter
        self.preprocessor = FeaturePreprocessor(feature_config or FeatureConfig())
        
        # State
        self._running = False
        self._price_history: list[tuple[datetime, float]] = []
    
    async def run(
        self,
        market_id: str,
        interval_seconds: float = 1.0,
        max_iterations: int | None = None,
    ) -> None:
        """
        Run the trading bot.
        
        Args:
            market_id: Market to trade
            interval_seconds: Polling interval
            max_iterations: Maximum iterations (None = run indefinitely)
        """
        await self.adapter.connect()
        self._running = True
        
        iteration = 0
        
        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Get market state
                state = await self.adapter.get_market_state(market_id)
                
                # Update price history
                self._price_history.append((state.timestamp, state.price))
                if len(self._price_history) > 1000:  # Keep last 1000 prices
                    self._price_history = self._price_history[-1000:]
                
                # Build observation
                obs = self._build_observation(state)
                
                # Get action from agent
                action, _ = self.agent.predict(obs, deterministic=True)
                
                # Execute action
                await self._execute_action(action, market_id, state.price)
                
                iteration += 1
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            logger.error("Bot error", error=str(e))
            raise
        finally:
            await self.adapter.disconnect()
    
    def stop(self) -> None:
        """Stop the bot."""
        self._running = False
    
    def _build_observation(self, state: MarketState) -> np.ndarray:
        """Build observation from current state."""
        import pandas as pd
        
        # Convert price history to DataFrame
        if len(self._price_history) < 60:
            # Pad with current price
            prices = [state.price] * (60 - len(self._price_history))
            prices.extend([p for _, p in self._price_history])
        else:
            prices = [p for _, p in self._price_history[-60:]]
        
        timestamps = pd.date_range(
            end=state.timestamp,
            periods=len(prices),
            freq='1s',
        )
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "volume": [1.0] * len(prices),  # Mock volume
        })
        
        # Preprocess
        features_df = self.preprocessor.transform(df)
        
        # Get sequence
        seq = self.preprocessor.get_sequence(features_df, len(features_df) - 1)
        current_features = features_df.drop(columns=["timestamp"]).iloc[-1].values
        
        # Build full observation
        obs_parts = [
            seq.flatten(),
            current_features,
            np.array([0.0, 0.0, 1.0, 0.0]),  # Portfolio state (placeholder)
            np.array([state.time_to_resolution, 0.5]),  # Time features
        ]
        
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        return obs
    
    async def _execute_action(
        self,
        action: int | np.ndarray,
        market_id: str,
        current_price: float,
    ) -> None:
        """Execute trading action."""
        action_idx = int(action) if isinstance(action, (int, np.integer)) else int(action[0])
        
        position = await self.adapter.get_position(market_id)
        balance = await self.adapter.get_balance()
        
        # Map discrete actions
        if action_idx == 0:  # Hold
            return
        elif action_idx == 1:  # Small buy
            size = (balance * 0.05) / current_price
            await self.adapter.place_order(market_id, "buy", size)
        elif action_idx == 2:  # Large buy
            size = (balance * 0.15) / current_price
            await self.adapter.place_order(market_id, "buy", size)
        elif action_idx == 3:  # Small sell
            size = (balance * 0.05) / current_price
            await self.adapter.place_order(market_id, "sell", size)
        elif action_idx == 4:  # Large sell
            size = (balance * 0.15) / current_price
            await self.adapter.place_order(market_id, "sell", size)
        elif action_idx == 5:  # Close
            if position and abs(position.size) > 0.01:
                side = "sell" if position.size > 0 else "buy"
                await self.adapter.place_order(market_id, side, abs(position.size))
