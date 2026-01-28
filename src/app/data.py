"""
Data Service for fetching market info.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import httpx
try:
    import httpx_socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False

from src.app.config import AppConfig

logger = logging.getLogger(__name__)

class MarketDataSource(ABC):
    """Abstract base for data sources."""
    
    @abstractmethod
    async def start(self):
        pass
        
    @abstractmethod
    async def stop(self):
        pass
        
    @abstractmethod
    async def refresh(self, state: Dict[str, Any]):
        """Update shared state dictionary."""
        pass

class PolymarketAPISource(MarketDataSource):
    """Fetch from official APIs."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        
    async def start(self):
        if self.config.proxy_url and SOCKS_AVAILABLE:
            transport = httpx_socks.AsyncProxyTransport.from_url(self.config.proxy_url)
            self.client = httpx.AsyncClient(transport=transport, verify=False, timeout=10.0)
        else:
            self.client = httpx.AsyncClient(timeout=10.0)
            
    async def stop(self):
        if self.client:
            await self.client.aclose()
            
    async def refresh(self, state: Dict[str, Any]):
        """Fetch active market and prices."""
        # 1. Fetch Active 15m Market
        now_ts = int(datetime.now(timezone.utc).timestamp())
        target_ts = (now_ts // 900) * 900
        
        # If we have a market for this timestamp, we might not need to scan gamma often
        # But for now, let's keep it simple
        
        if state["market_ts"] != target_ts:
            # New candle? Clear history
             state["yes_history"] = []
             state["no_history"] = []
             state["model_yes_history"] = []
             state["model_no_history"] = []
             state["market_slug"] = None
             
        if not state["market_slug"]:
            slug = f"btc-updown-15m-{target_ts}"
            try:
                resp = await self.client.get(f"https://gamma-api.polymarket.com/events/slug/{slug}")
                if resp.status_code == 200:
                    data = resp.json()
                    markets = data.get("markets", [])
                    if markets and isinstance(markets, list):
                        m = markets[0]
                        clob_tokens = m.get("clobTokenIds", [])
                        if isinstance(clob_tokens, str):
                            import json
                            clob_tokens = json.loads(clob_tokens)
                        if len(clob_tokens) >= 2:
                            state["market_slug"] = slug
                            state["market_ts"] = target_ts
                            state["yes_id"] = clob_tokens[0]
                            state["no_id"] = clob_tokens[1]
                            logger.info(f"Targeting market: {slug}")
            except Exception as e:
                logger.error(f"Discovery error: {e}")
        
        # 2. Fetch Prices
        if state["yes_id"]:
            start_ts = now_ts - (3600 * 2) 
            try:
                # History
                url = "https://clob.polymarket.com/prices-history"
                params = {"market": state["yes_id"], "fidelity": 1, "startTs": start_ts, "endTs": now_ts}
                resp = await self.client.get(url, params=params)
                
                if resp.status_code == 200 and "history" in resp.json():
                    hist = resp.json()["history"]
                    if hist:
                        # Full replace
                        state["yes_history"] = [float(p['p']) for p in hist]
                        state["no_history"] = [1.0 - x for x in state["yes_history"]]
                        state["yes_price"] = state["yes_history"][-1]
                        state["no_price"] = state["no_history"][-1]
                    else:
                        # Fallback to book if history empty (rare but possible at scan start)
                        await self._fetch_book(state)
                else:
                    await self._fetch_book(state)
            except Exception as e:
                logger.error(f"Price fetch error: {e}")

    async def _fetch_book(self, state):
        try:
             resp = await self.client.get("https://clob.polymarket.com/book", params={"token_id": state["yes_id"]})
             if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if bids and asks:
                    mid = (float(bids[0]['price']) + float(asks[0]['price'])) / 2
                    state["yes_price"] = mid
                    state["no_price"] = 1.0 - mid
                    state["yes_history"].append(mid)
                    state["no_history"].append(1.0 - mid)
        except Exception as e:
            pass


class OnChainDataSource(MarketDataSource):
    """
    Fetch from RPC/Web3.
    Uses OrderFilled events from Polymarket CTF Exchange.
    """
    CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    NEGRISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"


    def __init__(self, config: AppConfig):
        self.config = config
        self.w3 = None
        # Reuse the API helper just for market discovery (easiest way to get Token IDs)
        self.api_helper = PolymarketAPISource(config)
        self.last_scanned_block = None
        
    async def start(self):
        from web3 import Web3
        logger.info(f"Connecting to OnChain Source: {self.config.rpc_url}")
        self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        
        await self.api_helper.start()
        
        if not self.w3.is_connected():
            logger.error("Failed to connect to RPC!")
        else:
            cid = self.w3.eth.chain_id
            logger.info(f"Connected to Chain ID: {cid}")
        
    async def stop(self):
        await self.api_helper.stop()
        

    async def refresh(self, state: Dict[str, Any]):
        # 1. Discovery (Use API helper for robust ID lookup)
        # We need to know WHICH tokens to look for on-chain
        # This is async (httpx), so we can await it directly
        await self.api_helper.refresh(state)
        
        # 2. On-Chain Price Fetching (Blocking Web3 calls)
        if not self.w3:
            return

        # Offload blocking IO to thread
        await asyncio.to_thread(self._refresh_sync, state)

    def _refresh_sync(self, state: Dict[str, Any]):
        """Synchronous part of refresh running in a thread."""
        try:
            if not self.w3.is_connected():
                return

            yes_id = state.get("yes_id")
            if not yes_id:
                return

            current_block = self.w3.eth.block_number
            
            # Determine block range
            if self.last_scanned_block is None:
                # First run: Scan back 1000 blocks
                from_block = current_block - 1000
                to_block = current_block
                logger.info(f"OnChain: Initial scan from {from_block} to {to_block}")
            else:
                # Incremental: Scan from last + 1
                from_block = self.last_scanned_block + 1
                to_block = current_block
                
                if from_block > to_block:
                    # No new blocks
                    return

            filter_params = {
                "fromBlock": from_block,
                "toBlock": to_block,
                "topics": [self.ORDER_FILLED_TOPIC]
            }
            
            # Check both exchanges (Standard CTF and NegRisk)
            addresses = [
                self.w3.to_checksum_address(self.NEGRISK_CTF_EXCHANGE),
                self.w3.to_checksum_address(self.CTF_EXCHANGE)
            ]
            
            logs = self.w3.eth.get_logs({
                **filter_params,
                "address": addresses
            })
            
            # Decode NEW prices
            new_prices = []
            from eth_abi import decode
            
            for log in logs:
                try:
                    data_hex = log["data"].hex() if hasattr(log["data"], "hex") else log["data"]
                    if data_hex.startswith("0x"): data_hex = data_hex[2:]
                    data_bytes = bytes.fromhex(data_hex)
                    decoded = decode(["uint256", "uint256", "uint256", "uint256", "uint256"], data_bytes)
                    
                    price = None
                    target_int = int(yes_id)
                    
                    # Simplification: Just look for matches (Maker or Taker)
                    if decoded[0] == 0 and decoded[1] == target_int:
                        # Maker=USDC, Taker=YES. Maker BUYING YES? 
                        if decoded[3] > 0:
                            price = (decoded[2] / 1e6) / (decoded[3] / 1e6)
                    elif decoded[1] == 0 and decoded[0] == target_int:
                        # Maker=YES, Taker=USDC. Maker SELLING YES.
                        if decoded[2] > 0:
                            price = (decoded[3] / 1e6) / (decoded[2] / 1e6)
                            
                    if price is not None:
                        new_prices.append(price)
                        
                except Exception:
                    continue
            
            # Update State
            if new_prices:
                logger.info(f"OnChain: Found {len(new_prices)} new trades in blocks {from_block}-{to_block}")
                
                # Append to existing history if incremental
                if self.last_scanned_block is not None:
                    # Keep existing history, append new
                    state["yes_history"].extend(new_prices)
                    state["no_history"].extend([1.0 - p for p in new_prices])
                    
                    # Limit history length to avoid unbounded growth?
                    # Let's keep last 5000 points
                    if len(state["yes_history"]) > 5000:
                         state["yes_history"] = state["yes_history"][-5000:]
                         state["no_history"] = state["no_history"][-5000:]
                else:
                    # Initial scan: Overwrite
                    state["yes_history"] = new_prices
                    state["no_history"] = [1.0 - p for p in new_prices]
                
                # Update latest prices
                if state["yes_history"]:
                    state["yes_price"] = state["yes_history"][-1]
                    state["no_price"] = state["no_history"][-1]
            elif self.last_scanned_block is None:
                 logger.info("OnChain: No trades found in lookback")

            # Update pointer
            self.last_scanned_block = to_block

        except Exception as e:
            logger.error(f"OnChain Fetch Error: {e}")




class MarketDataService:
    """Service to fetch consolidated market data."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.binance_client: Optional[httpx.AsyncClient] = None
        
        # Strategy pattern for source
        self.source: MarketDataSource = PolymarketAPISource(config)
        # We also need Binance for BTC usually (unless onchain has Chainlink)
        
        # Shared State
        self.market_state: Dict[str, Any] = {
            "btc_price": 0.0,
            "btc_open": 0.0,
            "market_slug": None,
            "market_ts": 0,
            "yes_id": None,
            "no_id": None,
            "yes_price": 0.5,
            "no_price": 0.5,
            "yes_history": [],
            "no_history": [],
            "btc_history": []
        }
        
    async def start(self):
        """Initialize clients."""
        self.binance_client = httpx.AsyncClient(timeout=5.0)
        
        # Switch on config
        if self.config.data_source == "onchain":
            self.source = OnChainDataSource(self.config)
        else:
            self.source = PolymarketAPISource(self.config)
            
        await self.source.start()
            
    async def stop(self):
        await self.source.stop()
        if self.binance_client:
            await self.binance_client.aclose()
            
    async def refresh(self):
        """Refresh all data."""
        await asyncio.gather(
            self._fetch_btc_price(),
            self.source.refresh(self.market_state)
        )
        self._update_model_sampling()

    def _update_model_sampling(self):
        """Sample prices at fixed intervals for the model."""
        import time
        now = time.time()
        
        # Init sampling state if needed
        if "last_sample_ts" not in self.market_state:
            self.market_state["last_sample_ts"] = 0
            self.market_state["model_yes_history"] = []
            self.market_state["model_no_history"] = []
            
        # Sample every 5 seconds (matching typical training data freq)
        if now - self.market_state["last_sample_ts"] >= 5.0:
            self.market_state["last_sample_ts"] = now
            
            # Use current price
            y_price = self.market_state.get("yes_price", 0.5)
            n_price = self.market_state.get("no_price", 0.5)
            
            self.market_state["model_yes_history"].append(y_price)
            self.market_state["model_no_history"].append(n_price)
            
            # Keep history bounded (e.g. 500 points = ~40 mins)
            if len(self.market_state["model_yes_history"]) > 500:
                self.market_state["model_yes_history"] = self.market_state["model_yes_history"][-500:]
                self.market_state["model_no_history"] = self.market_state["model_no_history"][-500:]

    async def _fetch_btc_price(self):
        """Fetch BTC price from Binance."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            resp = await self.binance_client.get(url)
            if resp.status_code == 200:
                price = float(resp.json()["price"])
                self.market_state["btc_price"] = price
                self.market_state["btc_history"].append(price)
                if len(self.market_state["btc_history"]) > 500:
                    self.market_state["btc_history"] = self.market_state["btc_history"][-500:]
                    
                if self.market_state["btc_open"] == 0.0:
                    self.market_state["btc_open"] = price
        except Exception as e:
            logger.error(f"Error fetching BTC: {e}")

    def get_snapshot(self) -> Dict[str, Any]:
        """Return standardized data snapshot for Model/UI."""
        now_ts = datetime.now(timezone.utc).timestamp()
        
        # Calculate time remaining
        time_rem = 0.0
        if self.market_state["market_ts"]:
            end_ts = self.market_state["market_ts"] + 900
            rem_sec = max(0, end_ts - now_ts)
            time_rem = min(1.0, rem_sec / 900.0)

        # Use SAMPLED history for the model if available, otherwise fallback (graceful start)
        model_yes = self.market_state.get("model_yes_history", [])
        model_no = self.market_state.get("model_no_history", [])
        
        # If sampling hasn't built up yet, use raw history but warn/limit? 
        # Actually better to send what we have.
        if not model_yes:
            model_yes = self.market_state["yes_history"]
            model_no = self.market_state["no_history"]

        return {
            "yes_price_history": model_yes,
            "no_price_history": model_no,
            "btc_price_history": self.market_state["btc_history"],
            "btc_open_price": self.market_state["btc_open"],
            "btc_current_price": self.market_state["btc_price"],
            "time_remaining": time_rem,
            "market_slug": self.market_state["market_slug"],
            "yes_price": self.market_state["yes_price"],
            "no_price": self.market_state["no_price"],
            "last_updated": now_ts
        }
