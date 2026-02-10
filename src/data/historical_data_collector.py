"""
Historical Data Collector for Training.

Fetches 30 days of synchronized:
1. BTC price data from Binance (1-second resolution)
2. Polymarket 15-minute market outcomes and prices

This provides REAL data for training instead of simulated data.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import httpx
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

logger = structlog.get_logger(__name__)
console = Console()

# SOCKS5 proxy support
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False
    httpx_socks = None

SOCKS5_PROXY_URL = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:1080")


class HistoricalDataCollector:
    """
    Collect historical data for training.

    Data sources:
    1. Binance: BTC/USDT klines at 1s resolution
    2. Polymarket: 15-minute BTC up/down markets
    """

    def __init__(
        self,
        output_dir: str = "./data/historical",
        use_proxy: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_proxy = use_proxy and SOCKS_AVAILABLE

        # API endpoints
        self.binance_base = "https://api.binance.com"
        self.polymarket_clob = "https://clob.polymarket.com"
        self.polymarket_gamma = "https://gamma-api.polymarket.com"

        # Clients
        self.binance_client: Optional[httpx.AsyncClient] = None
        self.polymarket_client: Optional[httpx.AsyncClient] = None

    async def setup(self):
        """Initialize HTTP clients."""
        self.binance_client = httpx.AsyncClient(timeout=30)

        if self.use_proxy:
            transport = httpx_socks.AsyncProxyTransport.from_url(SOCKS5_PROXY_URL)
            self.polymarket_client = httpx.AsyncClient(
                transport=transport,
                timeout=30,
                verify=False
            )
            console.print("[blue]Using SOCKS5 proxy for Polymarket[/blue]")
        else:
            self.polymarket_client = httpx.AsyncClient(timeout=30)

    async def close(self):
        """Close HTTP clients."""
        if self.binance_client:
            await self.binance_client.aclose()
        if self.polymarket_client:
            await self.polymarket_client.aclose()

    async def fetch_binance_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1s",
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch klines from Binance.

        Args:
            symbol: Trading pair
            interval: Candle interval (1s, 1m, 15m, etc.)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            limit: Max records per request

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.binance_base}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = await self.binance_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
            return pd.DataFrame()

    async def fetch_btc_data_range(
        self,
        days_back: int = 30,
        interval: str = "1s",
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch BTC data for a date range.

        For 1s data, we need to batch requests due to API limits.
        """
        if start_date is not None:
            start_time = start_date
            end_time = start_date + timedelta(days=days_back)
        else:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)

        console.print(f"[bold blue]Fetching BTC {interval} data: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}[/bold blue]")

        all_data = []
        current_start = start_time

        # Binance returns max 1000 records per request
        # For 1s data: 1000s = ~16.7 minutes per request
        batch_duration = timedelta(seconds=1000)

        total_batches = int((end_time - start_time) / batch_duration) + 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Fetching BTC data", total=total_batches)

            while current_start < end_time:
                current_end = min(current_start + batch_duration, end_time)

                df = await self.fetch_binance_klines(
                    symbol="BTCUSDT",
                    interval=interval,
                    start_time=int(current_start.timestamp() * 1000),
                    end_time=int(current_end.timestamp() * 1000),
                    limit=1000,
                )

                if not df.empty:
                    all_data.append(df)

                current_start = current_end
                progress.update(task, advance=1)

                # Rate limiting
                await asyncio.sleep(0.1)

        if not all_data:
            console.print("[red]No BTC data fetched![/red]")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        console.print(f"[green]Fetched {len(combined)} BTC data points[/green]")

        return combined

    async def fetch_polymarket_15min_candle(self, timestamp: int) -> Optional[Dict]:
        """
        Fetch a specific 15-minute BTC Up/Down market.

        Slug pattern: btc-updown-15m-{unix_timestamp}
        """
        slug = f"btc-updown-15m-{timestamp}"

        try:
            response = await self.polymarket_client.get(
                f"{self.polymarket_gamma}/events/slug/{slug}"
            )

            if response.status_code != 200:
                return None

            data = response.json()
            markets = data.get("markets", [])

            if not markets:
                return None

            market = markets[0]

            # Parse outcome
            outcome_prices = market.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            closed = market.get("closed", False)

            if closed and outcome_prices and len(outcome_prices) >= 2:
                up_won = float(outcome_prices[0]) > 0.5
                outcome = 1 if up_won else 0
            else:
                outcome = None  # Not resolved

            # Parse token IDs
            clob_tokens = market.get("clobTokenIds", "[]")
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)

            return {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                "slug": slug,
                "closed": closed,
                "outcome": outcome,
                "yes_token_id": clob_tokens[0] if clob_tokens else None,
                "no_token_id": clob_tokens[1] if len(clob_tokens) > 1 else None,
                "volume": float(data.get("volume", 0)),
            }

        except Exception as e:
            logger.debug(f"Error fetching {slug}: {e}")
            return None

    async def fetch_token_price_history(
        self,
        token_id: str,
        fidelity: int = 1,  # 1-minute resolution
    ) -> pd.DataFrame:
        """
        Fetch price history for a token.
        """
        try:
            response = await self.polymarket_client.get(
                f"{self.polymarket_clob}/prices-history",
                params={
                    "market": token_id,
                    "fidelity": fidelity,
                    "interval": "max",
                }
            )

            if response.status_code != 200:
                return pd.DataFrame()

            data = response.json()
            history = data.get("history", [])

            if not history:
                return pd.DataFrame()

            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df["price"] = df["p"].astype(float)

            return df[["timestamp", "price"]].sort_values("timestamp")

        except Exception as e:
            logger.error(f"Error fetching price history for {token_id}: {e}")
            return pd.DataFrame()

    async def fetch_polymarket_data_range(
        self,
        days_back: int = 30,
        start_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch Polymarket 15-minute candles and their price histories.

        Returns:
            (candles_df, prices_df)
        """
        if start_date is not None:
            start_ts = (int(start_date.timestamp()) // 900) * 900
            end_ts = start_ts + (days_back * 24 * 3600)
            console.print(f"[bold blue]Fetching Polymarket data: {start_date.strftime('%Y-%m-%d')} + {days_back} days[/bold blue]")
        else:
            now = int(datetime.now(timezone.utc).timestamp())
            end_ts = (now // 900) * 900
            start_ts = end_ts - (days_back * 24 * 3600)
            console.print(f"[bold blue]Fetching {days_back} days of Polymarket data...[/bold blue]")

        # Generate all 15-min timestamps
        timestamps = list(range(start_ts, end_ts + 1, 900))

        console.print(f"Scanning {len(timestamps)} candle timestamps...")

        candles = []
        all_prices = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Fetching candles", total=len(timestamps))

            for ts in timestamps:
                candle = await self.fetch_polymarket_15min_candle(ts)

                if candle and candle.get("closed") and candle.get("outcome") is not None:
                    candles.append(candle)

                    # Fetch price history for this candle
                    if candle.get("yes_token_id"):
                        prices = await self.fetch_token_price_history(
                            candle["yes_token_id"],
                            fidelity=1,  # 1-minute resolution
                        )

                        if not prices.empty:
                            prices["candle_timestamp"] = ts
                            prices["outcome"] = candle["outcome"]
                            all_prices.append(prices)

                        # Rate limiting
                        await asyncio.sleep(0.5)

                progress.update(task, advance=1)

                # Rate limiting
                await asyncio.sleep(0.2)

        # Build DataFrames
        candles_df = pd.DataFrame(candles) if candles else pd.DataFrame()

        if all_prices:
            prices_df = pd.concat(all_prices, ignore_index=True)
        else:
            prices_df = pd.DataFrame()

        console.print(f"[green]Fetched {len(candles_df)} resolved candles[/green]")
        console.print(f"[green]Fetched {len(prices_df)} price points[/green]")

        return candles_df, prices_df

    async def collect_all_data(
        self,
        days_back: int = 30,
        btc_interval: str = "1s",
        start_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all historical data.

        Returns dictionary with:
        - btc_data: BTC price data at specified interval
        - candles: Polymarket candle metadata
        - prices: Polymarket YES token prices
        """
        await self.setup()

        try:
            # Fetch BTC data
            btc_df = await self.fetch_btc_data_range(
                days_back=days_back,
                interval=btc_interval,
                start_date=start_date,
            )

            # Fetch Polymarket data
            candles_df, prices_df = await self.fetch_polymarket_data_range(
                days_back=days_back,
                start_date=start_date,
            )

            # Save to disk
            if not btc_df.empty:
                btc_path = self.output_dir / f"btc_{btc_interval}_{days_back}d.parquet"
                btc_df.to_parquet(btc_path)
                console.print(f"[green]Saved BTC data to {btc_path}[/green]")

            if not candles_df.empty:
                candles_path = self.output_dir / f"polymarket_candles_{days_back}d.parquet"
                candles_df.to_parquet(candles_path)
                console.print(f"[green]Saved candles to {candles_path}[/green]")

            if not prices_df.empty:
                prices_path = self.output_dir / f"polymarket_prices_{days_back}d.parquet"
                prices_df.to_parquet(prices_path)
                console.print(f"[green]Saved prices to {prices_path}[/green]")

            # Summary
            console.print("\n[bold]Data Collection Summary:[/bold]")
            console.print(f"  BTC data points: {len(btc_df)}")
            console.print(f"  Polymarket candles: {len(candles_df)}")
            console.print(f"  Polymarket price points: {len(prices_df)}")

            if not candles_df.empty:
                outcomes = candles_df['outcome'].value_counts()
                console.print(f"  Outcome distribution: YES={outcomes.get(1, 0)}, NO={outcomes.get(0, 0)}")

            return {
                'btc_data': btc_df,
                'candles': candles_df,
                'prices': prices_df,
            }

        finally:
            await self.close()


class TrainingDataBuilder:
    """
    Build training dataset from collected historical data.

    Creates labeled examples for the MarketPredictor model.
    """

    def __init__(self, data_dir: str = "./data/historical"):
        self.data_dir = Path(data_dir)

    def load_data(
        self,
        days_back: int = 30,
        btc_interval: str = "1s",
    ) -> Dict[str, pd.DataFrame]:
        """Load previously collected data."""
        data = {}

        btc_path = self.data_dir / f"btc_{btc_interval}_{days_back}d.parquet"
        if btc_path.exists():
            data['btc'] = pd.read_parquet(btc_path)
            console.print(f"[blue]Loaded {len(data['btc'])} BTC records[/blue]")

        candles_path = self.data_dir / f"polymarket_candles_{days_back}d.parquet"
        if candles_path.exists():
            data['candles'] = pd.read_parquet(candles_path)
            console.print(f"[blue]Loaded {len(data['candles'])} candles[/blue]")

        prices_path = self.data_dir / f"polymarket_prices_{days_back}d.parquet"
        if prices_path.exists():
            data['prices'] = pd.read_parquet(prices_path)
            console.print(f"[blue]Loaded {len(data['prices'])} price points[/blue]")

        return data

    def build_training_examples(
        self,
        data: Dict[str, pd.DataFrame],
        samples_per_candle: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build training examples from historical data.

        For each sample point in each candle, compute:
        1. Features (using EnhancedFeatureBuilder)
        2. Position state
        3. Optimal action (using hindsight)
        4. Expected return

        Returns:
            (features, position_states, actions, returns)
        """
        from src.data.enhanced_features import EnhancedFeatureBuilder
        from src.models.market_predictor import (
            OptimalActionLabeler,
            EnhancedPositionState,
            Action,
        )

        feature_builder = EnhancedFeatureBuilder()
        labeler = OptimalActionLabeler(transaction_cost=0.05, min_profit_threshold=0.06)

        features_list = []
        position_states_list = []
        actions_list = []
        returns_list = []

        prices_df = data.get('prices', pd.DataFrame())
        candles_df = data.get('candles', pd.DataFrame())
        btc_df = data.get('btc', pd.DataFrame())

        if prices_df.empty or candles_df.empty:
            console.print("[red]No price or candle data available![/red]")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Iterate candles in temporal order (important for time-based train/test split)
        candle_timestamps_sorted = sorted(prices_df['candle_timestamp'].unique())

        console.print(f"Building training examples from {len(candle_timestamps_sorted)} candles...")

        for candle_ts in candle_timestamps_sorted:
            candle_prices = prices_df[prices_df['candle_timestamp'] == candle_ts]
            # Need at least 20 prices to have meaningful samples (start from idx 10)
            if len(candle_prices) < 20:
                continue

            outcome = candle_prices['outcome'].iloc[0]
            yes_prices = candle_prices['price'].values
            no_prices = 1.0 - yes_prices

            # Get BTC data for this candle
            candle_start = pd.Timestamp(candle_ts, unit='s', tz='UTC')
            candle_end = candle_start + pd.Timedelta(minutes=15)

            if not btc_df.empty and 'timestamp' in btc_df.columns:
                btc_mask = (btc_df['timestamp'] >= candle_start) & (btc_df['timestamp'] < candle_end)
                btc_candle = btc_df.loc[btc_mask]
                if not btc_candle.empty:
                    btc_prices = btc_candle['close'].values
                    btc_open = btc_prices[0]
                    btc_volumes = btc_candle['volume'].values if 'volume' in btc_candle.columns else None
                else:
                    btc_prices = None
                    btc_open = None
                    btc_volumes = None
            else:
                btc_prices = None
                btc_open = None
                btc_volumes = None

            # Sample points within candle (start from index 10, end at last valid index)
            max_idx = len(yes_prices) - 1
            start_idx = min(10, max_idx - 1)  # Ensure start is valid
            sample_indices = np.linspace(
                start_idx, max_idx, samples_per_candle, dtype=int
            )
            sample_indices = np.unique(sample_indices)  # Remove duplicates

            for idx in sample_indices:
                # Ensure idx is within bounds
                if idx >= len(yes_prices):
                    continue
                time_remaining = max(0.0, 1.0 - idx / len(yes_prices))

                # Build realistic position scenarios per sample point
                import random
                scenarios = [(False, None, 0.0, 0, 0.0)]  # Always include no-position

                if idx > 5:
                    # Random YES entries from earlier indices
                    num_yes = random.randint(1, 2)
                    for _ in range(num_yes):
                        entry_idx = random.randint(0, idx - 1)
                        entry_p = yes_prices[entry_idx]
                        ticks = idx - entry_idx
                        # Compute actual max unrealized PnL between entry and current
                        max_pnl = float(np.max(yes_prices[entry_idx:idx+1]) - entry_p) / (entry_p + 1e-8)
                        scenarios.append((True, "yes", entry_p, ticks, max_pnl))

                    # Random NO entries from earlier indices
                    num_no = random.randint(1, 2)
                    for _ in range(num_no):
                        entry_idx = random.randint(0, idx - 1)
                        entry_p = no_prices[entry_idx]
                        ticks = idx - entry_idx
                        max_pnl = float(np.max(no_prices[entry_idx:idx+1]) - entry_p) / (entry_p + 1e-8)
                        scenarios.append((True, "no", entry_p, ticks, max_pnl))

                # Compute features
                btc_slice = None
                btc_vol_slice = None
                if btc_prices is not None and len(btc_prices) > 0:
                    btc_idx = min(idx + 1, len(btc_prices))
                    btc_slice = btc_prices[:btc_idx]
                    if btc_volumes is not None and len(btc_volumes) > 0:
                        btc_vol_slice = btc_volumes[:btc_idx]

                features = feature_builder.compute_features(
                    yes_prices=yes_prices[:idx+1],
                    no_prices=no_prices[:idx+1],
                    time_remaining=time_remaining,
                    btc_prices=btc_slice,
                    btc_open=btc_open,
                    btc_volumes=btc_vol_slice,
                )

                for has_pos, pos_side, entry_price, ticks_held, max_pnl in scenarios:
                    # Compute position state
                    current_price = yes_prices[idx] if pos_side == "yes" else no_prices[idx]

                    position_state = EnhancedPositionState.compute(
                        has_position=has_pos,
                        position_side=pos_side,
                        entry_price=entry_price,
                        current_price=current_price,
                        time_remaining=time_remaining,
                        ticks_held=ticks_held,
                        max_pnl_seen=max_pnl,
                    )

                    # Compute optimal action using hindsight
                    action, expected_return = labeler.compute_optimal_action(
                        yes_prices=yes_prices,
                        current_idx=idx,
                        has_position=has_pos,
                        position_side=pos_side,
                        entry_price=entry_price,
                        outcome=outcome,
                    )

                    features_list.append(features)
                    position_states_list.append(position_state)
                    actions_list.append(action)
                    returns_list.append(expected_return)

        console.print(f"[green]Built {len(features_list)} training examples[/green]")

        return (
            np.array(features_list, dtype=np.float32),
            np.array(position_states_list, dtype=np.float32),
            np.array(actions_list, dtype=np.int64),
            np.array(returns_list, dtype=np.float32),
        )


async def collect_training_data(
    output_dir: str = "./data/historical",
    days_back: int = 30,
    btc_interval: str = "1s",
) -> Dict[str, pd.DataFrame]:
    """
    Main entry point for data collection.
    """
    collector = HistoricalDataCollector(output_dir=output_dir)
    return await collector.collect_all_data(
        days_back=days_back,
        btc_interval=btc_interval,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect historical training data")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--output", type=str, default="./data/historical")
    parser.add_argument("--btc-interval", type=str, default="1s")

    args = parser.parse_args()

    asyncio.run(collect_training_data(
        output_dir=args.output,
        days_back=args.days,
        btc_interval=args.btc_interval,
    ))
