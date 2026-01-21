"""
Balanced Data Generator for DeepLOB Training.

Fetches random intervals from Binance ensuring balanced Up/Down representation.
This avoids training bias from trending markets.
"""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import httpx
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.progress import Progress

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class BalancedDataConfig:
    """Configuration for balanced data generation."""
    
    # Target samples
    n_up_samples: int = 1000
    n_down_samples: int = 1000
    n_hold_samples: int = 500
    
    # Interval parameters
    interval_seconds: int = 200  # Length of each interval
    candle_minutes: int = 15  # To match prediction horizon
    
    # Thresholds for Up/Down/Hold classification
    up_threshold: float = 0.0001  # 0.01%
    down_threshold: float = -0.0001
    
    # Data range
    days_back: int = 30  # How far back to sample from
    
    # Binance settings
    symbol: str = "BTCUSDT"
    kline_interval: str = "1s"


class BinanceHistoricalFetcher:
    """Fetch historical data from Binance for balanced sampling."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
    
    async def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1s",
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch kline/candlestick data."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        response = await self.client.get(f"{self.BASE_URL}/klines", params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return pd.DataFrame()
        
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
    
    async def fetch_random_interval(
        self,
        days_back: int = 30,
        interval_seconds: int = 200,
        symbol: str = "BTCUSDT",
    ) -> Tuple[pd.DataFrame, float]:
        """
        Fetch a random interval and return data + return.
        
        Returns:
            (dataframe, return_pct)
        """
        now = datetime.utcnow()
        
        # Random time in past N days
        random_offset_minutes = random.randint(60, days_back * 24 * 60)
        target_time = now - timedelta(minutes=random_offset_minutes)
        
        end_time = int(target_time.timestamp() * 1000)
        start_time = int((target_time - timedelta(seconds=interval_seconds + 100)).timestamp() * 1000)
        
        df = await self.fetch_klines(
            symbol=symbol,
            interval="1s",
            start_time=start_time,
            end_time=end_time,
            limit=interval_seconds + 200,
        )
        
        if len(df) < 50:
            return df, 0.0
        
        # Calculate return over the interval
        start_price = df["price"].iloc[0]
        end_price = df["price"].iloc[-1]
        ret = (end_price - start_price) / start_price
        
        return df, ret
    
    async def close(self):
        await self.client.aclose()


class BalancedDataGenerator:
    """Generate balanced training data with equal Up/Down samples."""
    
    def __init__(self, config: Optional[BalancedDataConfig] = None):
        self.config = config or BalancedDataConfig()
        self.fetcher = BinanceHistoricalFetcher()
        
        # Collected intervals
        self.up_intervals: List[pd.DataFrame] = []
        self.down_intervals: List[pd.DataFrame] = []
        self.hold_intervals: List[pd.DataFrame] = []
    
    async def collect_balanced_intervals(self) -> Tuple[List[pd.DataFrame], List[int]]:
        """
        Collect balanced Up/Down/Hold intervals from random historical samples.
        
        Returns:
            (list of dataframes, list of labels)
        """
        console.print("[bold blue]Collecting balanced training intervals...[/bold blue]")
        console.print(f"  Target: {self.config.n_up_samples} Up, {self.config.n_down_samples} Down, {self.config.n_hold_samples} Hold")
        
        max_attempts = (self.config.n_up_samples + self.config.n_down_samples + self.config.n_hold_samples) * 5
        attempts = 0
        
        with Progress() as progress:
            task = progress.add_task(
                "Fetching intervals", 
                total=self.config.n_up_samples + self.config.n_down_samples + self.config.n_hold_samples
            )
            
            while attempts < max_attempts:
                # Check if we have enough samples
                if (len(self.up_intervals) >= self.config.n_up_samples and
                    len(self.down_intervals) >= self.config.n_down_samples and
                    len(self.hold_intervals) >= self.config.n_hold_samples):
                    break
                
                try:
                    df, ret = await self.fetcher.fetch_random_interval(
                        days_back=self.config.days_back,
                        interval_seconds=self.config.interval_seconds,
                        symbol=self.config.symbol,
                    )
                    
                    if len(df) < 100:
                        attempts += 1
                        continue
                    
                    # Classify and collect
                    if ret > self.config.up_threshold and len(self.up_intervals) < self.config.n_up_samples:
                        self.up_intervals.append(df)
                        progress.update(task, advance=1)
                        logger.info(f"Up: {len(self.up_intervals)}/{self.config.n_up_samples}, ret={ret:.4%}")
                    elif ret < self.config.down_threshold and len(self.down_intervals) < self.config.n_down_samples:
                        self.down_intervals.append(df)
                        progress.update(task, advance=1)
                        logger.info(f"Down: {len(self.down_intervals)}/{self.config.n_down_samples}, ret={ret:.4%}")
                    elif len(self.hold_intervals) < self.config.n_hold_samples:
                        self.hold_intervals.append(df)
                        progress.update(task, advance=1)
                        logger.info(f"Hold: {len(self.hold_intervals)}/{self.config.n_hold_samples}, ret={ret:.4%}")
                    
                    attempts += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching interval: {e}")
                    attempts += 1
                    await asyncio.sleep(1)
        
        # Combine all intervals
        all_intervals = []
        all_labels = []
        
        for df in self.up_intervals[:self.config.n_up_samples]:
            all_intervals.append(df)
            all_labels.append(2)  # Up
        
        for df in self.down_intervals[:self.config.n_down_samples]:
            all_intervals.append(df)
            all_labels.append(0)  # Down
        
        for df in self.hold_intervals[:self.config.n_hold_samples]:
            all_intervals.append(df)
            all_labels.append(1)  # Hold
        
        console.print(f"[green]✓ Collected {len(all_intervals)} balanced intervals[/green]")
        console.print(f"  Up: {len(self.up_intervals)}, Down: {len(self.down_intervals)}, Hold: {len(self.hold_intervals)}")
        
        return all_intervals, all_labels
    
    async def generate_and_save(self, output_path: str) -> str:
        """Generate balanced dataset and save to parquet."""
        intervals, labels = await self.collect_balanced_intervals()
        
        if not intervals:
            raise ValueError("No intervals collected")
        
        # Combine into single dataframe with interval_id
        combined_dfs = []
        for i, (df, label) in enumerate(zip(intervals, labels)):
            df = df.copy()
            df["interval_id"] = i
            df["label"] = label
            combined_dfs.append(df)
        
        combined = pd.concat(combined_dfs, ignore_index=True)
        
        # Save
        output_file = Path(output_path) / "balanced_intervals.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_file)
        
        # Save metadata
        metadata = {
            "n_intervals": len(intervals),
            "n_up": len(self.up_intervals),
            "n_down": len(self.down_intervals),
            "n_hold": len(self.hold_intervals),
            "up_threshold": self.config.up_threshold,
            "down_threshold": self.config.down_threshold,
            "interval_seconds": self.config.interval_seconds,
            "days_back": self.config.days_back,
            "generated_at": datetime.utcnow().isoformat(),
        }
        
        import json
        with open(Path(output_path) / "balanced_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"[bold green]Saved to {output_file}[/bold green]")
        return str(output_file)
    
    async def close(self):
        await self.fetcher.close()


async def fetch_balanced_data(
    n_up: int = 500,
    n_down: int = 500,
    n_hold: int = 250,
    days_back: int = 30,
    output_dir: str = "./data/balanced",
) -> str:
    """Main entry point for fetching balanced data."""
    config = BalancedDataConfig(
        n_up_samples=n_up,
        n_down_samples=n_down,
        n_hold_samples=n_hold,
        days_back=days_back,
    )
    
    generator = BalancedDataGenerator(config)
    
    try:
        output_file = await generator.generate_and_save(output_dir)
        return output_file
    finally:
        await generator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate balanced training data")
    parser.add_argument("--n-up", type=int, default=500)
    parser.add_argument("--n-down", type=int, default=500)
    parser.add_argument("--n-hold", type=int, default=250)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", default="./data/balanced")
    
    args = parser.parse_args()
    
    asyncio.run(fetch_balanced_data(
        n_up=args.n_up,
        n_down=args.n_down,
        n_hold=args.n_hold,
        days_back=args.days,
        output_dir=args.output,
    ))
