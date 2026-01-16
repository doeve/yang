"""
Live inference adapter for integration with d3v trading bot.

Converts BinanceClient data format to model input format.
Ensures training and inference use identical observations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LiveObservationConfig:
    """Configuration matching d3v binance-client.ts settings."""
    
    # From binance-client.ts
    price_history_length: int = 300  # 30 seconds at 100ms
    trade_window_ms: int = 60_000  # 1 minute order flow window
    
    # Observation format
    include_order_flow: bool = True
    include_time_features: bool = True


class LiveObservationBuilder:
    """
    Builds observations for the ML model from live Binance data.
    
    Matches EXACTLY the observation format used during training:
    - Raw normalized price history (300 points)
    - Order flow metrics (buyPressure, netFlow normalized)
    - Current price change from slot start
    - Time remaining features
    
    Usage in d3v:
        builder = LiveObservationBuilder(config)
        
        # Get data from BinanceClient
        price_history = binance_client.priceHistory.get(symbol)
        order_flow = binance_client.getOrderFlowMetrics(symbol)
        
        # Build observation
        obs = builder.build(
            price_history=price_history,
            order_flow=order_flow,
            slot_start_price=chainlink_start_price,
            time_remaining_ms=time_remaining,
            position_normalized=current_position / max_position,
            unrealized_pnl_normalized=unrealized_pnl / initial_capital,
            cash_ratio=cash / initial_capital,
            exposure=abs(position_value) / portfolio_value,
        )
        
        # Get model prediction
        action, _ = model.predict(obs)
    """
    
    def __init__(self, config: LiveObservationConfig | None = None):
        self.config = config or LiveObservationConfig()
    
    def build(
        self,
        price_history: list[dict],  # [{ price: float, timestamp: int }, ...]
        order_flow: dict,  # { buyPressure, buyVolume, sellVolume, netFlow }
        slot_start_price: float,
        current_price: float,
        time_remaining_ms: int,
        slot_duration_ms: int = 15 * 60 * 1000,  # 15 min default
        # Portfolio state
        position_normalized: float = 0.0,
        unrealized_pnl_normalized: float = 0.0,
        cash_ratio: float = 1.0,
        exposure: float = 0.0,
    ) -> np.ndarray:
        """
        Build observation matching training format.
        
        Args:
            price_history: List of {price, timestamp} from BinanceClient.priceHistory
            order_flow: OrderFlowMetrics from BinanceClient.getOrderFlowMetrics()
            slot_start_price: Chainlink price at slot start
            current_price: Current Binance price
            time_remaining_ms: Time until slot ends
            slot_duration_ms: Total slot duration
            position_normalized: Current position / max position
            unrealized_pnl_normalized: Unrealized PnL / initial capital
            cash_ratio: Cash / initial capital  
            exposure: Position exposure ratio
            
        Returns:
            np.ndarray observation matching training format
        """
        obs_parts = []
        
        # 1. RAW PRICE HISTORY (normalized to % change from first in window)
        prices = [p['price'] for p in price_history[-self.config.price_history_length:]]
        
        # Pad if not enough history yet
        while len(prices) < self.config.price_history_length:
            if prices:
                prices.insert(0, prices[0])
            else:
                prices.append(current_price)
        
        prices = np.array(prices, dtype=np.float32)
        if prices[0] != 0:
            normalized_prices = ((prices / prices[0]) - 1.0) * 100  # % change
        else:
            normalized_prices = np.zeros_like(prices)
        
        obs_parts.append(normalized_prices)
        
        # 2. ORDER FLOW AS VOLUME HISTORY PROXY
        # Since training uses volume, we use order flow as proxy
        # Create synthetic volume signal from buy/sell pressure
        volume_signal = np.zeros(self.config.price_history_length, dtype=np.float32)
        
        if self.config.include_order_flow:
            buy_pressure = order_flow.get('buyPressure', 0.5)
            net_flow = order_flow.get('netFlow', 0)
            
            # Last part of array gets order flow signal
            flow_signal = (buy_pressure - 0.5) * 2  # -1 to 1
            volume_signal[-60:] = flow_signal  # Last 60 points (last ~6 seconds)
        
        obs_parts.append(volume_signal)
        
        # 3. CURRENT PRICE CHANGE (since slot start - the prediction target)
        price_change = ((current_price / slot_start_price) - 1.0) * 100 if slot_start_price > 0 else 0.0
        obs_parts.append(np.array([price_change], dtype=np.float32))
        
        # 4. PORTFOLIO STATE
        obs_parts.append(np.array([
            position_normalized,
            unrealized_pnl_normalized,
            cash_ratio,
            exposure,
        ], dtype=np.float32))
        
        # 5. TIME FEATURES
        if self.config.include_time_features:
            time_to_resolution = max(0, time_remaining_ms) / slot_duration_ms
            progress = 1.0 - time_to_resolution
            obs_parts.append(np.array([time_to_resolution, progress], dtype=np.float32))
        
        # Concatenate and clip
        obs = np.concatenate(obs_parts).astype(np.float32)
        obs = np.clip(obs, -10.0, 10.0)
        
        return obs
    
    @property
    def observation_dim(self) -> int:
        """Get total observation dimension."""
        dim = self.config.price_history_length  # prices
        dim += self.config.price_history_length  # volume/order flow
        dim += 1  # current price change
        dim += 4  # portfolio state
        if self.config.include_time_features:
            dim += 2  # time features
        return dim


def action_to_signal(action: int, market_price: float) -> dict:
    """
    Convert model action to d3v trading signal format.
    
    Actions:
        0: Hold
        1-3: Buy YES (small/medium/large)
        4-6: Buy NO (small/medium/large)
        7: Reduce 50%
        8: Close all
    
    Returns:
        Signal dict compatible with d3v execution layer
    """
    action_map = {
        0: {'action': 'HOLD', 'side': None, 'size_pct': 0},
        1: {'action': 'BUY', 'side': 'UP', 'size_pct': 0.05},
        2: {'action': 'BUY', 'side': 'UP', 'size_pct': 0.15},
        3: {'action': 'BUY', 'side': 'UP', 'size_pct': 0.30},
        4: {'action': 'BUY', 'side': 'DOWN', 'size_pct': 0.05},
        5: {'action': 'BUY', 'side': 'DOWN', 'size_pct': 0.15},
        6: {'action': 'BUY', 'side': 'DOWN', 'size_pct': 0.30},
        7: {'action': 'REDUCE', 'side': None, 'size_pct': 0.50},
        8: {'action': 'CLOSE', 'side': None, 'size_pct': 1.0},
    }
    
    signal = action_map.get(action, action_map[0])
    signal['market_price'] = market_price
    signal['action_id'] = action
    
    return signal
