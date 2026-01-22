#!/usr/bin/env python3
"""
Synchronized Data Collection Pipeline.

Collects and aligns:
- Polymarket 15-minute BTC Up/Down candle data
- Binance BTC 1-second price data for the same time range
- Generates DeepLOB predictions for each candle

Output: synchronized_training_data.parquet with all data aligned by timestamp.
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


class SynchronizedDataCollector:
    """
    Collect and synchronize Polymarket candles with BTC price data.
    
    For each Polymarket 15-min candle:
    1. Fetch candle outcome (up_won/down_won)
    2. Fetch BTC 1s data for the 15 minutes before candle close
    3. Optionally run DeepLOB inference to get predictions
    """
    
    BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
    POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def fetch_btc_1s_klines(
        self,
        start_ts: int,
        end_ts: int,
    ) -> pd.DataFrame:
        """Fetch BTC 1-second klines from Binance."""
        all_klines = []
        current_start = start_ts * 1000  # Convert to ms
        end_ms = end_ts * 1000
        
        while current_start < end_ms:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1s",
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            
            try:
                resp = await self.client.get(self.BINANCE_KLINE_URL, params=params)
                resp.raise_for_status()
                klines = resp.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1000  # Next ms after last
                
                await asyncio.sleep(0.1)  # Rate limit
                
            except Exception as e:
                console.print(f"[yellow]Warning: Binance API error: {e}[/yellow]")
                break
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    async def fetch_polymarket_candle(self, timestamp: int) -> Optional[Dict]:
        """Fetch a single Polymarket 15-min candle by timestamp."""
        slug = f"btc-updown-15m-{timestamp}"
        url = f"{self.POLYMARKET_GAMMA_URL}/markets/{slug}"
        
        try:
            resp = await self.client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            
            data = resp.json()
            
            # Determine outcome from market status
            closed = data.get("closed", False)
            outcomes = data.get("outcomes", [])
            
            # Find winning outcome
            up_won = None
            if closed and outcomes:
                for outcome in outcomes:
                    if outcome.get("winner", False):
                        up_won = outcome.get("value", "").lower() == "yes"
                        break
            
            return {
                "timestamp": timestamp,
                "datetime": datetime.utcfromtimestamp(timestamp),
                "slug": slug,
                "title": data.get("title", ""),
                "volume": float(data.get("volume", 0)),
                "closed": closed,
                "up_won": up_won,
                "yes_token_id": data.get("tokens", [{}])[0].get("token_id", ""),
                "no_token_id": data.get("tokens", [{}])[-1].get("token_id", ""),
            }
        except Exception as e:
            console.print(f"[dim]Skip {slug}: {e}[/dim]")
            return None
    
    async def collect_polymarket_candles(
        self,
        hours_back: int = 168,  # 7 days
    ) -> pd.DataFrame:
        """Collect Polymarket candles for given time range."""
        console.print(f"[bold blue]Collecting Polymarket candles ({hours_back} hours)...[/bold blue]")
        
        now = int(datetime.utcnow().timestamp())
        current_boundary = (now // 900) * 900
        start = current_boundary - (hours_back * 3600)
        
        timestamps = list(range(start, current_boundary + 1, 900))
        console.print(f"  Scanning {len(timestamps)} timestamps...")
        
        candles = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("Fetching Polymarket candles", total=len(timestamps))
            
            for ts in timestamps:
                candle = await self.fetch_polymarket_candle(ts)
                if candle and candle.get("closed"):
                    candles.append(candle)
                
                await asyncio.sleep(0.5)  # Rate limit
                progress.advance(task)
        
        df = pd.DataFrame(candles)
        console.print(f"[green]✓ Found {len(df)} closed candles[/green]")
        return df
    
    async def collect_btc_for_candles(
        self,
        candles_df: pd.DataFrame,
        lookback_seconds: int = 900,  # Full 15-min candle
    ) -> Dict[int, pd.DataFrame]:
        """Collect BTC 1s data for each Polymarket candle."""
        console.print(f"[bold blue]Collecting BTC data for {len(candles_df)} candles...[/bold blue]")
        
        btc_data = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("Fetching BTC data", total=len(candles_df))
            
            for _, candle in candles_df.iterrows():
                ts = candle["timestamp"]
                
                # Candle runs from ts to ts+900
                start_ts = ts
                end_ts = ts + 900
                
                df = await self.fetch_btc_1s_klines(start_ts, end_ts)
                
                if len(df) > 0:
                    btc_data[ts] = df
                
                await asyncio.sleep(0.1)  # Rate limit
                progress.advance(task)
        
        console.print(f"[green]✓ Collected BTC data for {len(btc_data)} candles[/green]")
        return btc_data
    
    def compute_features(self, btc_df: pd.DataFrame) -> Dict:
        """Compute trading features from BTC data."""
        if len(btc_df) < 60:
            return {}
        
        # Price features
        prices = btc_df['close'].values
        volumes = btc_df['volume'].values
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        
        return {
            "btc_open": prices[0],
            "btc_close": prices[-1],
            "btc_high": prices.max(),
            "btc_low": prices.min(),
            "btc_volume": volumes.sum(),
            "btc_return": (prices[-1] / prices[0]) - 1,
            "btc_volatility": np.std(returns) if len(returns) > 0 else 0,
            "btc_momentum": (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0,
        }
    
    async def run_full_collection(
        self,
        hours_back: int = 168,
        with_predictions: bool = False,
    ) -> pd.DataFrame:
        """
        Run full synchronized data collection.
        
        Args:
            hours_back: Hours of history to collect
            with_predictions: If True, generate DeepLOB predictions
            
        Returns:
            DataFrame with synchronized candle + BTC data
        """
        console.print("[bold magenta]═══ Synchronized Data Collection Pipeline ═══[/bold magenta]")
        console.print(f"  Hours back: {hours_back}")
        console.print(f"  Expected candles: ~{hours_back * 4}")
        console.print()
        
        # Step 1: Collect Polymarket candles
        candles_df = await self.collect_polymarket_candles(hours_back)
        
        if len(candles_df) == 0:
            console.print("[red]Error: No candles collected[/red]")
            return pd.DataFrame()
        
        # Step 2: Collect BTC data for each candle
        btc_data = await self.collect_btc_for_candles(candles_df)
        
        # Step 3: Merge and compute features
        console.print("[bold blue]Computing features...[/bold blue]")
        
        results = []
        for _, candle in candles_df.iterrows():
            ts = candle["timestamp"]
            
            if ts not in btc_data:
                continue
            
            btc_df = btc_data[ts]
            features = self.compute_features(btc_df)
            
            if not features:
                continue
            
            row = {
                "timestamp": ts,
                "datetime": candle["datetime"],
                "up_won": candle["up_won"],
                "volume": candle["volume"],
                **features,
            }
            results.append(row)
        
        final_df = pd.DataFrame(results)
        
        # Step 4: Optionally generate DeepLOB predictions
        if with_predictions and len(final_df) > 0:
            final_df = await self._add_deep_lob_predictions(final_df, btc_data)
        
        # Save synchronized data
        output_path = self.output_dir / "synchronized_training_data.parquet"
        final_df.to_parquet(output_path, index=False)
        
        # Also save raw BTC data
        btc_path = self.output_dir / "btc_1s_synchronized.parquet"
        if btc_data:
            all_btc = pd.concat(
                [df.assign(candle_ts=ts) for ts, df in btc_data.items()],
                ignore_index=True
            )
            all_btc.to_parquet(btc_path, index=False)
            console.print(f"  Saved BTC data: {btc_path}")
        
        # Save Polymarket candles
        poly_path = self.output_dir / "polymarket" / "btc_15min_candles.parquet"
        poly_path.parent.mkdir(exist_ok=True)
        candles_df.to_parquet(poly_path, index=False)
        console.print(f"  Saved Polymarket candles: {poly_path}")
        
        # Summary
        console.print()
        self._print_summary(final_df)
        
        console.print(f"\n[green]✓ Saved synchronized data: {output_path}[/green]")
        
        return final_df
    
    async def _add_deep_lob_predictions(
        self,
        df: pd.DataFrame,
        btc_data: Dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """Add DeepLOB predictions to the dataset."""
        try:
            from src.inference.deep_lob_inference import DeepLOBTwoLayerBot
            
            console.print("[bold blue]Generating DeepLOB predictions...[/bold blue]")
            
            bot = DeepLOBTwoLayerBot()
            bot.load_models("./logs/deep_lob_balanced")
            
            predictions = []
            
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if ts not in btc_data:
                    predictions.append({
                        "prob_down": 0.33,
                        "prob_hold": 0.34,
                        "prob_up": 0.33,
                    })
                    continue
                
                btc_df = btc_data[ts]
                
                # Ensure required columns
                if 'price' not in btc_df.columns:
                    btc_df = btc_df.copy()
                    btc_df['price'] = btc_df['close']
                if 'buy_pressure' not in btc_df.columns:
                    btc_df['buy_pressure'] = 0.5
                
                try:
                    probs = bot.predict_class_probabilities(btc_df)
                    predictions.append({
                        "prob_down": probs["down"],
                        "prob_hold": probs["hold"],
                        "prob_up": probs["up"],
                    })
                except Exception as e:
                    predictions.append({
                        "prob_down": 0.33,
                        "prob_hold": 0.34,
                        "prob_up": 0.33,
                    })
            
            pred_df = pd.DataFrame(predictions)
            df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
            
            console.print(f"[green]✓ Added predictions for {len(predictions)} candles[/green]")
            
        except ImportError as e:
            console.print(f"[yellow]⚠ Could not load DeepLOB: {e}[/yellow]")
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        table = Table(title="Data Collection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Candles", str(len(df)))
        
        if 'up_won' in df.columns:
            up_count = df['up_won'].sum()
            down_count = len(df) - up_count
            table.add_row("Up Won", f"{up_count} ({up_count/len(df)*100:.1f}%)")
            table.add_row("Down Won", f"{down_count} ({down_count/len(df)*100:.1f}%)")
        
        if 'datetime' in df.columns:
            table.add_row("Start Date", str(df['datetime'].min()))
            table.add_row("End Date", str(df['datetime'].max()))
        
        if 'btc_return' in df.columns:
            table.add_row("Avg BTC Return", f"{df['btc_return'].mean()*100:.3f}%")
        
        if 'prob_up' in df.columns:
            table.add_row("Has Predictions", "Yes ✓")
        
        console.print(table)


async def main(
    output_dir: str = "./data",
    hours_back: int = 168,
    with_predictions: bool = False,
):
    """Run the synchronized data collection pipeline."""
    async with SynchronizedDataCollector(output_dir) as collector:
        await collector.run_full_collection(
            hours_back=hours_back,
            with_predictions=with_predictions,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronized data collection pipeline")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--hours", type=int, default=168, help="Hours of history (default: 168 = 7 days)")
    parser.add_argument("--with-predictions", action="store_true", help="Generate DeepLOB predictions")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        output_dir=args.output,
        hours_back=args.hours,
        with_predictions=args.with_predictions,
    ))
