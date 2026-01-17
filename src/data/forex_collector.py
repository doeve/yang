"""
Forex Data Collector.

Collects DXY (Dollar Index) and EUR/USD data for correlation analysis with BTC.
Uses free APIs to fetch historical data at 1-minute resolution.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ForexCollector:
    """
    Collects forex data from free APIs.
    
    Data sources:
    - Twelve Data API (free tier: 800 req/day)
    - Alpha Vantage (free tier: 500 req/day)
    - Yahoo Finance (no limit, but less granular)
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # We'll use Yahoo Finance as it's free and reliable
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    async def fetch_from_yahoo(
        self, 
        symbol: str,
        interval: str = "1m",
        range_days: int = 7,
        max_retries: int = 3,
    ) -> pd.DataFrame | None:
        """
        Fetch data from Yahoo Finance with retry logic.
        
        Note: 1m data only available for last 7 days
              1h data available for 730 days
        """
        import aiohttp
        
        # Yahoo symbols: DX-Y.NYB for DXY, EURUSD=X for EUR/USD
        yahoo_symbols = {
            "DXY": "DX-Y.NYB",
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "XAUUSD": "GC=F",  # Gold futures
        }
        
        yahoo_sym = yahoo_symbols.get(symbol, symbol)
        
        # Determine range based on interval
        if interval == "1m":
            range_str = "7d"  # Max for 1m
        elif interval == "5m":
            range_str = "60d"
        else:
            range_str = "2y"
        
        url = f"{self.base_url}/{yahoo_sym}"
        params = {
            "interval": interval,
            "range": range_str,
        }
        
        # Headers to avoid rate limiting
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                        if resp.status == 429:
                            # Rate limited - wait and retry
                            wait_time = (attempt + 1) * 5
                            logger.warning(f"Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if resp.status != 200:
                            logger.error(f"Yahoo API error: {resp.status}")
                            return None
                        
                        data = await resp.json()
                        
                result = data.get("chart", {}).get("result", [])
                if not result:
                    logger.error(f"No data for {symbol}")
                    return None
                
                chart = result[0]
                timestamps = chart.get("timestamp", [])
                quote = chart.get("indicators", {}).get("quote", [{}])[0]
                
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                    "volume": quote.get("volume"),
                })
                
                # Use close as price, handle NaN
                df["price"] = df["close"]
                df = df.dropna(subset=["price"])
                
                logger.info(f"Fetched {len(df)} rows for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        return None
    
    async def collect_all(self, interval: str = "1h") -> dict[str, pd.DataFrame]:
        """Collect all forex pairs."""
        symbols = ["DXY", "EURUSD"]
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")
            df = await self.fetch_from_yahoo(symbol, interval)
            if df is not None:
                results[symbol] = df
                
                # Save to parquet
                path = self.data_dir / f"{symbol.lower()}_{interval}.parquet"
                df.to_parquet(path)
                logger.info(f"Saved {symbol} to {path}")
            
            # Rate limiting
            await asyncio.sleep(1)
        
        return results
    
    def load_forex_data(self, symbol: str, interval: str = "1h") -> pd.DataFrame | None:
        """Load previously collected forex data."""
        path = self.data_dir / f"{symbol.lower()}_{interval}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def align_with_btc(
        self, 
        btc_df: pd.DataFrame, 
        forex_df: pd.DataFrame,
        resample_interval: str = "1min",
    ) -> pd.DataFrame:
        """
        Align forex data with BTC data by resampling and forward-filling.
        
        Since forex is hourly and BTC is per-second, we forward-fill
        forex values to match BTC timestamps.
        """
        # Set timestamp as index for both
        btc = btc_df.set_index("timestamp").copy()
        forex = forex_df.set_index("timestamp").copy()
        
        # Resample BTC to 1-minute for manageable size
        btc_resampled = btc.resample(resample_interval).last()
        
        # Forward-fill forex to match BTC timestamps
        # This makes hourly forex data available at every BTC timestamp
        combined_index = btc_resampled.index.union(forex.index).sort_values()
        forex_aligned = forex.reindex(combined_index).ffill()
        
        # Now reindex to BTC timestamps only
        forex_final = forex_aligned.reindex(btc_resampled.index)
        
        return forex_final.reset_index()


async def main():
    """Test the collector."""
    collector = ForexCollector("./data")
    
    print("Collecting forex data...")
    results = await collector.collect_all(interval="1h")
    
    for symbol, df in results.items():
        print(f"\n{symbol}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Latest price: {df['price'].iloc[-1]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
