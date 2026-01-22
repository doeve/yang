"""
Polymarket CLOB API Data Fetcher.

Fetches historical YES/NO token prices at 15-minute fidelity.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
import pandas as pd
import structlog
from rich.console import Console
from rich.progress import Progress

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class PolymarketConfig:
    """Configuration for Polymarket data fetching."""
    
    # API endpoints
    clob_base_url: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    
    # Data settings
    fidelity: int = 15  # 15-minute intervals
    
    # Rate limiting
    requests_per_second: float = 2.0


class PolymarketDataFetcher:
    """Fetch historical data from Polymarket CLOB API."""
    
    def __init__(self, config: Optional[PolymarketConfig] = None):
        self.config = config or PolymarketConfig()
        self.client = httpx.AsyncClient(timeout=30)
    
    async def search_markets(self, query: str = "crypto", limit: int = 50) -> List[Dict]:
        """Search for markets by query."""
        try:
            response = await self.client.get(
                f"{self.config.gamma_base_url}/markets",
                params={"closed": "false", "limit": limit}
            )
            response.raise_for_status()
            markets = response.json()
            
            # Filter by query
            filtered = [
                m for m in markets 
                if query.lower() in m.get("question", "").lower() 
                or query.lower() in m.get("slug", "").lower()
            ]
            
            return filtered
        except Exception as e:
            logger.error(f"Error searching markets: {e}")
            return []
    
    async def get_market_details(self, condition_id: str) -> Optional[Dict]:
        """Get details for a specific market."""
        try:
            response = await self.client.get(
                f"{self.config.gamma_base_url}/markets/{condition_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting market {condition_id}: {e}")
            return None
    
    async def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: Optional[str] = None,
        fidelity: int = 15,
    ) -> pd.DataFrame:
        """
        Fetch historical prices for a token.
        
        Args:
            token_id: CLOB token ID (YES or NO token)
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            interval: Predefined interval ('1h', '1d', '1w', 'max')
            fidelity: Resolution in minutes (default 15)
            
        Returns:
            DataFrame with timestamp and price columns
        """
        params = {
            "market": token_id,
            "fidelity": fidelity,
        }
        
        if start_ts and end_ts:
            params["startTs"] = start_ts
            params["endTs"] = end_ts
        elif interval:
            params["interval"] = interval
        else:
            params["interval"] = "max"
        
        try:
            response = await self.client.get(
                f"{self.config.clob_base_url}/prices-history",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            history = data.get("history", [])
            if not history:
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df["price"] = df["p"]
            df = df[["timestamp", "price"]].sort_values("timestamp")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching price history for {token_id}: {e}")
            return pd.DataFrame()
    
    async def fetch_btc_15min_candle(self, timestamp: int) -> Optional[Dict]:
        """
        Fetch a specific 15-minute BTC Up/Down market by timestamp.
        
        Slug pattern: btc-updown-15m-{unix_timestamp}
        The timestamp should be aligned to 15-minute boundaries (divisible by 900).
        """
        slug = f"btc-updown-15m-{timestamp}"
        
        try:
            response = await self.client.get(
                f"{self.config.gamma_base_url}/events/slug/{slug}"
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            markets = data.get("markets", [])
            
            if not markets:
                return None
            
            market = markets[0]
            
            # Parse outcome - outcomePrices is ["1", "0"] if Up won, ["0", "1"] if Down won
            outcome_prices = market.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            
            if outcome_prices and len(outcome_prices) >= 2:
                up_won = float(outcome_prices[0]) > 0.5
            else:
                up_won = None  # Not yet resolved
            
            # Parse clobTokenIds
            clob_tokens = market.get("clobTokenIds", "[]")
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            
            return {
                "timestamp": timestamp,
                "datetime": datetime.utcfromtimestamp(timestamp),
                "slug": slug,
                "title": data.get("title", ""),
                "volume": float(data.get("volume", 0)),
                "closed": market.get("closed", False),
                "up_won": up_won,
                "yes_token_id": clob_tokens[0] if clob_tokens else None,
                "no_token_id": clob_tokens[1] if len(clob_tokens) > 1 else None,
            }
            
        except Exception as e:
            logger.debug(f"Error fetching {slug}: {e}")
            return None
    
    async def fetch_btc_15min_candles(
        self,
        hours_back: int = 24,
    ) -> List[Dict]:
        """
        Fetch historical 15-minute BTC Up/Down markets.
        
        Iterates through timestamps at 15-minute intervals.
        
        Args:
            hours_back: How many hours of history to fetch
        
        Returns:
            List of market data dictionaries
        """
        console.print(f"[bold blue]Fetching {hours_back} hours of 15-min BTC candles...[/bold blue]")
        
        # Calculate timestamp range
        now = int(datetime.utcnow().timestamp())
        # Round down to current 15-min boundary
        current_boundary = (now // 900) * 900
        
        # Start from hours_back ago
        start = current_boundary - (hours_back * 3600)
        
        # Generate all 15-min timestamps
        timestamps = list(range(start, current_boundary + 1, 900))
        
        console.print(f"Scanning {len(timestamps)} timestamps...")
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Fetching candles", total=len(timestamps))
            
            for ts in timestamps:
                candle = await self.fetch_btc_15min_candle(ts)
                
                if candle:
                    results.append(candle)
                
                # Rate limiting
                await asyncio.sleep(1 / self.config.requests_per_second)
                
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Found {len(results)} candles with data[/green]")
        
        return results
    
    async def fetch_crypto_15min_markets(
        self,
        days_back: int = 30,
    ) -> List[Dict]:
        """
        Find and fetch data for crypto 15-minute prediction markets.
        
        Returns list of markets with their price histories.
        """
        console.print("[bold blue]Searching for crypto 15-minute markets...[/bold blue]")

        
        # Search for crypto-related markets
        all_markets = await self.search_markets("BTC", limit=100)
        all_markets.extend(await self.search_markets("Bitcoin", limit=100))
        all_markets.extend(await self.search_markets("crypto", limit=100))
        
        # Deduplicate by condition_id
        seen = set()
        unique_markets = []
        for m in all_markets:
            cid = m.get("conditionId", m.get("id", ""))
            if cid not in seen:
                seen.add(cid)
                unique_markets.append(m)
        
        console.print(f"Found {len(unique_markets)} unique crypto markets")
        
        # Filter for 15-minute style markets (look for "15" in question/slug)
        fifteen_min_markets = [
            m for m in unique_markets
            if "15" in m.get("question", "") or "15" in m.get("slug", "")
        ]
        
        if not fifteen_min_markets:
            console.print("[yellow]No 15-minute markets found, using all crypto markets[/yellow]")
            fifteen_min_markets = unique_markets[:20]
        
        console.print(f"Processing {len(fifteen_min_markets)} markets")
        
        # Fetch price histories
        results = []
        end_ts = int(datetime.utcnow().timestamp())
        start_ts = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
        
        with Progress() as progress:
            task = progress.add_task("Fetching prices", total=len(fifteen_min_markets))
            
            for market in fifteen_min_markets:
                # clobTokenIds is a JSON string, not array - parse it
                clob_tokens_raw = market.get("clobTokenIds", "[]")
                try:
                    if isinstance(clob_tokens_raw, str):
                        clob_tokens = json.loads(clob_tokens_raw)
                    else:
                        clob_tokens = clob_tokens_raw
                except json.JSONDecodeError:
                    clob_tokens = []
                
                if len(clob_tokens) < 2:
                    progress.update(task, advance=1)
                    continue
                
                yes_token_id = clob_tokens[0]
                no_token_id = clob_tokens[1]

                
                # Fetch YES prices
                yes_prices = await self.get_price_history(
                    yes_token_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    fidelity=self.config.fidelity,
                )
                
                await asyncio.sleep(1 / self.config.requests_per_second)
                
                if not yes_prices.empty:
                    results.append({
                        "market": market,
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                        "yes_prices": yes_prices,
                        "question": market.get("question", ""),
                        "slug": market.get("slug", ""),
                    })
                    logger.info(
                        "Fetched market",
                        slug=market.get("slug", ""),
                        rows=len(yes_prices),
                    )
                
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Fetched {len(results)} markets with price data[/green]")
        return results
    
    async def close(self):
        await self.client.aclose()


async def fetch_polymarket_data(
    output_dir: str = "./data/polymarket",
    days_back: int = 30,
) -> str:
    """Main entry point for fetching Polymarket data."""
    config = PolymarketConfig()
    fetcher = PolymarketDataFetcher(config)
    
    try:
        markets = await fetcher.fetch_crypto_15min_markets(days_back=days_back)
        
        if not markets:
            console.print("[red]No markets found[/red]")
            return ""
        
        # Save data
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Combine all price data
        all_prices = []
        for m in markets:
            df = m["yes_prices"].copy()
            df["market_slug"] = m["slug"]
            df["question"] = m["question"]
            df["yes_token_id"] = m["yes_token_id"]
            all_prices.append(df)
        
        combined = pd.concat(all_prices, ignore_index=True)
        output_file = output_path / "crypto_15min_prices.parquet"
        combined.to_parquet(output_file)
        
        # Save market metadata
        metadata = {
            "n_markets": len(markets),
            "total_rows": len(combined),
            "markets": [
                {
                    "slug": m["slug"],
                    "question": m["question"],
                    "rows": len(m["yes_prices"]),
                }
                for m in markets
            ],
            "fetched_at": datetime.utcnow().isoformat(),
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"[bold green]Saved to {output_file}[/bold green]")
        console.print(f"  Markets: {len(markets)}")
        console.print(f"  Total rows: {len(combined)}")
        
        return str(output_file)
    
    finally:
        await fetcher.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Polymarket data")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", default="./data/polymarket")
    
    args = parser.parse_args()
    
    asyncio.run(fetch_polymarket_data(
        output_dir=args.output,
        days_back=args.days,
    ))
