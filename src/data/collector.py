"""
Polymarket API data collector.

Fetches historical price data from Polymarket's CLOB and Gamma APIs
for crypto prediction markets with rate limiting and incremental updates.

Uses SOCKS5 proxy via SSH tunnel for regions where Polymarket is restricted.
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import httpx
import structlog
from pydantic import BaseModel

# Try to import SOCKS support
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False
    httpx_socks = None

logger = structlog.get_logger(__name__)

# SOCKS5 proxy configuration (via SSH tunnel)
# To create the tunnel: ssh -D 1080 -N -f root@72.62.114.55
SOCKS5_PROXY_URL = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:1080")


class MarketStatus(str, Enum):
    """Market status enum."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    UPCOMING = "upcoming"


class MarketType(str, Enum):
    """Type of prediction market."""
    BINARY = "binary"
    SCALAR = "scalar"


@dataclass
class PricePoint:
    """A single price observation."""
    timestamp: datetime
    price: float  # 0.0 to 1.0 for binary markets
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class Market:
    """Polymarket market metadata."""
    id: str
    condition_id: str
    question: str
    description: str
    market_type: MarketType
    status: MarketStatus
    created_at: datetime
    resolution_at: datetime | None
    resolved_at: datetime | None
    outcome: bool | None  # For resolved binary markets
    tokens: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class PriceHistory(BaseModel):
    """Price history for a market."""
    market_id: str
    token_id: str
    start_time: datetime
    end_time: datetime
    resolution_seconds: int
    prices: list[float]
    volumes: list[float]
    timestamps: list[datetime]


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request: float = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait until we can make another request."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request = asyncio.get_event_loop().time()


class PolymarketCollector:
    """
    Collects historical data from Polymarket APIs.
    
    Supports:
    - Gamma API for market discovery and metadata
    - CLOB API for price history and orderbook data
    - Incremental updates with checkpoint/resume
    
    Example:
        collector = PolymarketCollector()
        markets = await collector.fetch_crypto_markets()
        history = await collector.fetch_price_history(market_id, days=30)
    """
    
    # API endpoints
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    CLOB_API_URL = "https://clob.polymarket.com"
    
    # Crypto-related tags for filtering
    CRYPTO_TAGS = ["crypto", "bitcoin", "btc", "ethereum", "eth", "defi", "blockchain"]
    
    def __init__(
        self,
        api_key: str | None = None,
        requests_per_minute: int = 100,
    ):
        """
        Initialize the collector.
        
        Args:
            api_key: Optional API key for authenticated endpoints
            requests_per_minute: Rate limit for API requests
        """
        self.api_key = api_key
        self.rate_limiter = RateLimiter(requests_per_minute)
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "PolymarketCollector":
        """Async context manager entry."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use SOCKS5 proxy if available
        if SOCKS_AVAILABLE and httpx_socks:
            logger.info("Using SOCKS5 proxy", proxy=SOCKS5_PROXY_URL)
            # Use rdns=True to resolve DNS through the proxy
            transport = httpx_socks.AsyncProxyTransport.from_url(
                SOCKS5_PROXY_URL,
                rdns=True,  # Resolve DNS through proxy
            )
            self._client = httpx.AsyncClient(
                transport=transport,
                headers=headers,
                timeout=httpx.Timeout(60.0),
                follow_redirects=True,
                verify=False,  # Skip SSL verification (proxy may have issues)
            )
        else:
            logger.warning("SOCKS5 proxy not available, using direct connection")
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Collector must be used as async context manager")
        return self._client
    
    async def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a rate-limited GET request."""
        await self.rate_limiter.acquire()
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("API request failed", url=url, status=e.response.status_code)
            raise
        except Exception as e:
            logger.error("API request error", url=url, error=str(e))
            raise
    
    async def fetch_markets(
        self,
        status: MarketStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        tags: list[str] | None = None,
    ) -> list[Market]:
        """
        Fetch markets from Gamma API.
        
        Args:
            status: Filter by market status
            limit: Maximum number of markets to return
            offset: Pagination offset
            tags: Filter by tags
            
        Returns:
            List of Market objects
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        
        if status:
            params["status"] = status.value
        
        if tags:
            params["tag"] = ",".join(tags)
        
        url = f"{self.GAMMA_API_URL}/markets"
        data = await self._get(url, params)
        
        markets = []
        for item in data.get("data", data) if isinstance(data, dict) else data:
            try:
                market = self._parse_market(item)
                markets.append(market)
            except Exception as e:
                logger.warning("Failed to parse market", market_id=item.get("id"), error=str(e))
        
        return markets
    
    async def fetch_crypto_markets(
        self,
        status: MarketStatus | None = MarketStatus.ACTIVE,
        include_resolved: bool = True,
    ) -> list[Market]:
        """
        Fetch all crypto-related markets.
        
        Args:
            status: Primary status filter
            include_resolved: Also fetch resolved markets for training data
            
        Returns:
            List of crypto-related Market objects
        """
        all_markets: list[Market] = []
        
        # Fetch markets for each crypto tag
        for tag in self.CRYPTO_TAGS:
            logger.info("Fetching markets", tag=tag)
            
            offset = 0
            while True:
                markets = await self.fetch_markets(
                    status=status,
                    limit=100,
                    offset=offset,
                    tags=[tag],
                )
                
                if not markets:
                    break
                
                all_markets.extend(markets)
                offset += len(markets)
                
                if len(markets) < 100:
                    break
        
        # Also fetch resolved markets if requested
        if include_resolved and status != MarketStatus.RESOLVED:
            for tag in self.CRYPTO_TAGS:
                offset = 0
                while True:
                    markets = await self.fetch_markets(
                        status=MarketStatus.RESOLVED,
                        limit=100,
                        offset=offset,
                        tags=[tag],
                    )
                    
                    if not markets:
                        break
                    
                    all_markets.extend(markets)
                    offset += len(markets)
                    
                    if len(markets) < 100:
                        break
        
        # Deduplicate by ID
        seen_ids: set[str] = set()
        unique_markets: list[Market] = []
        for market in all_markets:
            if market.id not in seen_ids:
                seen_ids.add(market.id)
                unique_markets.append(market)
        
        logger.info("Fetched crypto markets", count=len(unique_markets))
        return unique_markets
    
    async def fetch_price_history(
        self,
        token_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        resolution: str = "1m",  # 1m, 5m, 15m, 1h, 1d
    ) -> PriceHistory:
        """
        Fetch historical price data for a token.
        
        Args:
            token_id: The token/outcome ID
            start_time: Start of the time range
            end_time: End of the time range (defaults to now)
            resolution: Price resolution (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            PriceHistory object with price data
        """
        end_time = end_time or datetime.now(timezone.utc)
        start_time = start_time or (end_time - timedelta(days=30))
        
        # Convert resolution to seconds
        resolution_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
        }
        resolution_seconds = resolution_map.get(resolution, 60)
        
        params = {
            "market": token_id,
            "startTs": int(start_time.timestamp()),
            "endTs": int(end_time.timestamp()),
            "fidelity": resolution_seconds,
        }
        
        url = f"{self.CLOB_API_URL}/prices-history"
        data = await self._get(url, params)
        
        prices: list[float] = []
        volumes: list[float] = []
        timestamps: list[datetime] = []
        
        history_data = data.get("history", [])
        for point in history_data:
            ts = datetime.fromtimestamp(point.get("t", 0), tz=timezone.utc)
            price = float(point.get("p", 0))
            volume = float(point.get("v", 0))
            
            timestamps.append(ts)
            prices.append(price)
            volumes.append(volume)
        
        return PriceHistory(
            market_id=token_id,
            token_id=token_id,
            start_time=start_time,
            end_time=end_time,
            resolution_seconds=resolution_seconds,
            prices=prices,
            volumes=volumes,
            timestamps=timestamps,
        )
    
    async def fetch_orderbook(
        self,
        token_id: str,
    ) -> dict[str, Any]:
        """
        Fetch current orderbook for a token.
        
        Args:
            token_id: The token/outcome ID
            
        Returns:
            Orderbook data with bids and asks
        """
        url = f"{self.CLOB_API_URL}/book"
        params = {"token_id": token_id}
        return await self._get(url, params)
    
    def _parse_market(self, data: dict[str, Any]) -> Market:
        """Parse raw API response into Market object."""
        # Parse dates
        created_at = datetime.fromisoformat(
            data.get("created_at", data.get("createdAt", "")).replace("Z", "+00:00")
        )
        
        resolution_at = None
        if res_date := data.get("end_date", data.get("endDate")):
            resolution_at = datetime.fromisoformat(res_date.replace("Z", "+00:00"))
        
        resolved_at = None
        if res_date := data.get("resolved_at", data.get("resolvedAt")):
            resolved_at = datetime.fromisoformat(res_date.replace("Z", "+00:00"))
        
        # Parse status
        status_str = data.get("status", data.get("active", "active"))
        if isinstance(status_str, bool):
            status = MarketStatus.ACTIVE if status_str else MarketStatus.RESOLVED
        else:
            status = MarketStatus(status_str) if status_str in [s.value for s in MarketStatus] else MarketStatus.ACTIVE
        
        # Parse outcome
        outcome = None
        if data.get("resolved"):
            outcome_str = data.get("outcome", data.get("resolution"))
            if outcome_str is not None:
                outcome = outcome_str in ["Yes", "yes", True, 1, "1"]
        
        return Market(
            id=data.get("id", data.get("condition_id", "")),
            condition_id=data.get("condition_id", data.get("conditionId", "")),
            question=data.get("question", ""),
            description=data.get("description", ""),
            market_type=MarketType.BINARY,  # Default to binary
            status=status,
            created_at=created_at,
            resolution_at=resolution_at,
            resolved_at=resolved_at,
            outcome=outcome,
            tokens=data.get("tokens", []),
            tags=data.get("tags", []),
        )
    
    async def interpolate_to_seconds(
        self,
        history: PriceHistory,
    ) -> PriceHistory:
        """
        Interpolate price history to 1-second resolution.
        
        Uses linear interpolation between data points.
        This is used for realistic simulation replay.
        
        Args:
            history: Original price history
            
        Returns:
            Interpolated PriceHistory at 1-second resolution
        """
        if not history.timestamps:
            return history
        
        import numpy as np
        
        # Create target timestamps at 1-second intervals
        start_ts = history.timestamps[0].timestamp()
        end_ts = history.timestamps[-1].timestamp()
        target_timestamps = np.arange(start_ts, end_ts + 1, 1.0)
        
        # Convert original timestamps to numpy array
        original_ts = np.array([t.timestamp() for t in history.timestamps])
        original_prices = np.array(history.prices)
        original_volumes = np.array(history.volumes)
        
        # Interpolate prices (linear)
        interpolated_prices = np.interp(target_timestamps, original_ts, original_prices)
        
        # For volumes, distribute evenly across interpolated points
        # This is a simplification - real volume distribution is unknown
        interpolated_volumes = np.interp(target_timestamps, original_ts, original_volumes)
        interpolated_volumes = interpolated_volumes / history.resolution_seconds
        
        # Convert back to lists
        new_timestamps = [
            datetime.fromtimestamp(ts, tz=timezone.utc)
            for ts in target_timestamps
        ]
        
        return PriceHistory(
            market_id=history.market_id,
            token_id=history.token_id,
            start_time=history.start_time,
            end_time=history.end_time,
            resolution_seconds=1,
            prices=interpolated_prices.tolist(),
            volumes=interpolated_volumes.tolist(),
            timestamps=new_timestamps,
        )
