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
        await self.api_helper.refresh(state)
        
        # 2. On-Chain Price Fetching
        if not self.w3 or not self.w3.is_connected():
            return
            
        yes_id = state.get("yes_id")
        if not yes_id:
            return
            
        # If we already have fresh history from API helper (fallback), we could use it, 
        # but user specifically asked for on-chain fetching.
        # So let's overwrite history with on-chain data if possible.
        
        try:
            # Look back ~1000 blocks (approx 45 mins on Polygon)
            current_block = self.w3.eth.block_number
            from_block = current_block - 1000
            
            # Normalize ID for topic filter
            # Token ID in topics is hex padded
            token_hex = hex(int(yes_id))
            token_topic = "0x" + token_hex[2:].zfill(64)
            
            # We want logs where EITHER makerAssetId OR takerAssetId is our token
            # But get_logs with OR topics is tricky in one go if positions differ
            # NegRisk OrderFilled: orderHash, maker, taker (topics 1,2,3)
            # IDs are in Data, not Topics for OrderFilled? 
            # Wait, fetch_onchain_prices.py says:
            # "Indexed params... maker, taker. Non-indexed: makerAssetId..."
            # So we cannot filter by TokenID in logs! We must fetch all OrderFilled and filter in client.
            # This is heavy for a full node, but fine for local Bor if volume isn't insane.
            # actually fetch_onchain_prices.py does `get_logs` with just address and topic0
            
            filter_params = {
                "fromBlock": from_block,
                "toBlock": "latest",
                "topics": [self.ORDER_FILLED_TOPIC]
            }
            
            # Check both exchanges? Usually NegRisk for these markets
            logs = self.w3.eth.get_logs({
                **filter_params,
                "address": self.w3.to_checksum_address(self.NEGRISK_CTF_EXCHANGE)
            })
            
            prices = []
            
            from eth_abi import decode
            
            for log in logs:
                try:
                    # Decode Data
                    data_hex = log["data"].hex() if hasattr(log["data"], "hex") else log["data"]
                    if data_hex.startswith("0x"): data_hex = data_hex[2:]
                    data_bytes = bytes.fromhex(data_hex)
                    
                    # uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmount, uint256 takerAmount, uint256 fee
                    decoded = decode(["uint256", "uint256", "uint256", "uint256", "uint256"], data_bytes)
                    
                    maker_id = str(decoded[0])
                    taker_id = str(decoded[1])
                    maker_amt = decoded[2]
                    taker_amt = decoded[3]
                    
                    price = None
                    
                    # Check if our token involved
                    # USDC is usually ID 0 or specific addr, but in CTF it's often 0 implies collateral
                    # Wait, fetch_onchain_prices says: "if maker_asset_id_int == 0 ... price = usdc/tokens"
                    
                    target_int = int(yes_id)
                    
                    if decoded[1] == target_int and decoded[0] == 0:
                        # Taker bought YES (gave USDC)
                        if taker_amt > 0:
                            price = maker_amt / taker_amt # maker gave USDC? No.
                            # Maker Asset 0 (USDC), Taker Asset YES
                            # Taker GAVE YES? No.
                            # Let's trust fetch_onchain_prices logic:
                            # if maker_asset_id_int == 0: Maker gave USDC, got tokens. Price = USDC/Tokens. Token is TakerAsset.
                            pass
                    
                    # Simplification: Just look for matches
                    if decoded[0] == 0 and decoded[1] == target_int:
                        # Maker=USDC, Taker=YES. Maker BUYING YES? 
                        # Maker gave USDC, Taker gave YES. Maker BOUGHT YES.
                        if decoded[3] > 0:
                            price = (decoded[2] / 1e6) / (decoded[3] / 1e6)
                    elif decoded[1] == 0 and decoded[0] == target_int:
                        # Maker=YES, Taker=USDC. Maker SELLING YES.
                        if decoded[2] > 0:
                            price = (decoded[3] / 1e6) / (decoded[2] / 1e6)
                            
                    if price is not None:
                        prices.append(price)
                        
                except Exception:
                    continue
            
            if prices:
                # Update state
                state["yes_history"] = prices
                state["no_history"] = [1.0 - p for p in prices]
                state["yes_price"] = prices[-1]
                state["no_price"] = state["no_history"][-1]
                logger.info(f"OnChain: Found {len(prices)} trades. Last: {prices[-1]:.3f}")
            else:
                logger.info("OnChain: No trades found in lookback")

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

        return {
            "yes_price_history": self.market_state["yes_history"],
            "no_price_history": self.market_state["no_history"],
            "btc_price_history": self.market_state["btc_history"],
            "btc_open_price": self.market_state["btc_open"],
            "btc_current_price": self.market_state["btc_price"],
            "time_remaining": time_rem,
            "market_slug": self.market_state["market_slug"],
            "yes_price": self.market_state["yes_price"],
            "last_updated": now_ts
        }
