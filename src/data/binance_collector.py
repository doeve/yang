"""
Crypto exchange data collector using Binance API.

Alternative to Polymarket for training data - provides freely accessible
crypto price data at 1-second resolution without geo-restrictions.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger(__name__)


class BinanceCollector:
    """
    Collects historical crypto data from Binance API.
    
    Provides 1-second kline (candlestick) data for training.
    No API key required for public endpoints.
    
    Example:
        collector = BinanceCollector()
        df = await collector.fetch_klines("BTCUSDT", days=7)
    """
    
    BASE_URL = "https://api.binance.com"
    
    # Popular crypto pairs for training
    DEFAULT_SYMBOLS = [
        "BTCUSDT",   # Bitcoin
        "ETHUSDT",   # Ethereum
        "BNBUSDT",   # BNB
        "SOLUSDT",   # Solana
        "XRPUSDT",   # XRP
        "ADAUSDT",   # Cardano
        "DOGEUSDT",  # Dogecoin
        "AVAXUSDT",  # Avalanche
        "LINKUSDT",  # Chainlink
        "MATICUSDT", # Polygon
    ]
    
    def __init__(self, rate_limit_per_minute: int = 1200):
        """
        Initialize the collector.
        
        Args:
            rate_limit_per_minute: Binance limit is 1200/min for IP
        """
        self.rate_limit = rate_limit_per_minute
        self._last_request_time = 0.0
        self._min_interval = 60.0 / rate_limit_per_minute
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "BinanceCollector":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={"Accept": "application/json"},
        )
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            raise RuntimeError("Collector must be used as async context manager")
        return self._client
    
    async def _rate_limit(self) -> None:
        """Simple rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1s",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1s, 1m, 5m, 15m, 1h, 1d)
            start_time: Start of time range
            end_time: End of time range
            limit: Max records per request (1000 max)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        await self._rate_limit()
        
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        url = f"{self.BASE_URL}/api/v3/klines"
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Parse Binance kline format:
        # [open_time, open, high, low, close, volume, close_time, ...]
        records = []
        for kline in data:
            records.append({
                "timestamp": datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            })
        
        return pd.DataFrame(records)
    
    async def fetch_aggtrades(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch aggregated trades (aggTrades) - EXACTLY matches d3v binance-client.ts format.
        
        This is tick-by-tick trade data, the same format received via:
        wss://stream.binance.com:9443/stream?streams=btcusdt@aggTrade
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_time: Start of time range
            end_time: End of time range
            limit: Max records per request (1000 max)
            
        Returns:
            DataFrame with columns: timestamp, price, quantity, is_buyer_maker
            - timestamp: Trade time (matches msg.data.T)
            - price: Trade price (matches msg.data.p)
            - quantity: Trade quantity (matches msg.data.q)
            - is_buyer_maker: True=sell aggressor (matches msg.data.m)
        """
        await self._rate_limit()
        
        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": limit,
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        url = f"{self.BASE_URL}/api/v3/aggTrades"
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Parse Binance aggTrade format (EXACTLY matches WebSocket format):
        # { a: aggTradeId, p: price, q: quantity, f: firstTradeId, l: lastTradeId, 
        #   T: timestamp, m: isBuyerMaker, M: isBestMatch }
        records = []
        for trade in data:
            records.append({
                "timestamp": datetime.fromtimestamp(trade["T"] / 1000, tz=timezone.utc),
                "price": float(trade["p"]),
                "quantity": float(trade["q"]),
                "is_buyer_maker": trade["m"],  # True = seller was aggressor
                "trade_id": trade["a"],
            })
        
        return pd.DataFrame(records)
    
    async def fetch_aggtrades_historical(
        self,
        symbol: str,
        hours: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch hours of tick-level aggTrades data.
        
        MATCHES D3V FORMAT EXACTLY:
        - priceHistory: Array<{ price: number; timestamp: number }>
        - historyLength = 300 (30 seconds at ~100ms resolution)
        
        WARNING: This is a lot of data! 1 hour of BTC = ~50-500k trades
        
        Args:
            symbol: Trading pair
            hours: Hours of history
            
        Returns:
            DataFrame with tick-level trade data
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        all_data: list[pd.DataFrame] = []
        current_start = start_time
        
        logger.info("Fetching aggTrades (tick-level)", symbol=symbol, hours=hours)
        
        batch_count = 0
        while current_start < end_time:
            df = await self.fetch_aggtrades(
                symbol=symbol,
                start_time=current_start,
                limit=1000,
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            batch_count += 1
            
            # Move to next batch using last trade time
            last_time = df["timestamp"].max()
            current_start = last_time + timedelta(milliseconds=1)
            
            if batch_count % 50 == 0:
                progress = (current_start - start_time).total_seconds() / (end_time - start_time).total_seconds()
                logger.info("aggTrades progress", symbol=symbol, progress=f"{progress:.1%}", trades=sum(len(d) for d in all_data))
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["trade_id"]).sort_values("timestamp")
        
        logger.info("Fetched aggTrades", symbol=symbol, total_trades=len(result))
        
        return result
    
    async def fetch_historical_data(
        self,
        symbol: str,
        days: int = 7,
        interval: str = "1s",
    ) -> pd.DataFrame:
        """
        Fetch multiple days of historical data.
        
        Handles pagination automatically for large date ranges.
        
        Args:
            symbol: Trading pair
            days: Number of days of history
            interval: Kline interval
            
        Returns:
            DataFrame with full historical data
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        all_data: list[pd.DataFrame] = []
        current_start = start_time
        
        # Interval to milliseconds mapping
        interval_ms = {
            "1s": 1000,
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        
        step_ms = interval_ms.get(interval, 1000) * 1000  # 1000 records per request
        
        logger.info("Fetching historical data", symbol=symbol, days=days, interval=interval)
        
        while current_start < end_time:
            current_end = min(
                current_start + timedelta(milliseconds=step_ms),
                end_time,
            )
            
            df = await self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            current_start = current_end
            
            # Progress logging
            progress = (current_start - start_time).total_seconds() / (end_time - start_time).total_seconds()
            if len(all_data) % 10 == 0:
                logger.info("Fetch progress", symbol=symbol, progress=f"{progress:.1%}")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        
        logger.info("Fetched data", symbol=symbol, records=len(result))
        
        return result
    
    async def collect_training_data(
        self,
        symbols: list[str] | None = None,
        days: int = 7,
        interval: str = "1m",  # Use 1m for reasonable data size
    ) -> dict[str, pd.DataFrame]:
        """
        Collect training data for multiple symbols.
        
        Args:
            symbols: List of trading pairs (default: popular cryptos)
            days: Days of history per symbol
            interval: Kline interval
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        symbols = symbols or self.DEFAULT_SYMBOLS[:5]  # Limit to 5 by default
        
        results: dict[str, pd.DataFrame] = {}
        
        for i, symbol in enumerate(symbols):
            logger.info("Collecting", symbol=symbol, progress=f"{i+1}/{len(symbols)}")
            
            try:
                df = await self.fetch_historical_data(
                    symbol=symbol,
                    days=days,
                    interval=interval,
                )
                
                if not df.empty:
                    # Normalize prices to 0-1 range for compatibility with prediction market format
                    df["price"] = df["close"]
                    
                    # Create market_id from symbol
                    df["market_id"] = symbol
                    
                    results[symbol] = df
                    
            except Exception as e:
                logger.error("Failed to collect", symbol=symbol, error=str(e))
        
        return results
    
    def normalize_to_prediction_market_format(
        self,
        df: pd.DataFrame,
        market_id: str,
    ) -> pd.DataFrame:
        """
        Convert exchange data to prediction market format.
        
        Normalizes prices to 0-1 probability-like range.
        
        Args:
            df: Exchange data DataFrame
            market_id: Market identifier
            
        Returns:
            DataFrame in prediction market format
        """
        result = df.copy()
        
        # Normalize close price to 0-1 range using min-max scaling
        min_price = result["close"].min()
        max_price = result["close"].max()
        price_range = max_price - min_price
        
        if price_range > 0:
            result["price"] = (result["close"] - min_price) / price_range
        else:
            result["price"] = 0.5
        
        # Ensure price is clipped to valid range
        result["price"] = np.clip(result["price"], 0.01, 0.99)
        
        # Add market_id
        result["market_id"] = market_id
        
        return result[["timestamp", "price", "volume", "market_id"]]


async def collect_and_save(
    output_dir: str = "./data",
    symbols: list[str] | None = None,
    days: int = 7,
) -> None:
    """
    Convenience function to collect and save training data.
    
    Args:
        output_dir: Output directory for data files
        symbols: Symbols to collect (default: top cryptos)
        days: Days of history
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    async with BinanceCollector() as collector:
        data = await collector.collect_training_data(
            symbols=symbols,
            days=days,
            interval="1m",
        )
        
        for symbol, df in data.items():
            # Normalize to prediction market format
            normalized = collector.normalize_to_prediction_market_format(df, symbol)
            
            # Save as parquet
            output_file = output_path / f"{symbol.lower()}_prices.parquet"
            normalized.to_parquet(output_file)
            
            logger.info("Saved", file=str(output_file), records=len(normalized))
