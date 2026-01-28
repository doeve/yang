"""
Live Executor for Polymarket trading.

Handles real on-chain order execution via CLOB API and direct contract calls.
Ported from d3v's TypeScript implementation.
"""

import asyncio
import hashlib
import hmac
import time
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import httpx
import structlog
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

logger = structlog.get_logger(__name__)


# Contract addresses (Polygon Mainnet)
CONTRACTS = {
    "CTF_EXCHANGE": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NEG_RISK_CTF_EXCHANGE": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "CONDITIONAL_TOKENS": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "USDC_E": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
}

# ERC20 ABI for balance queries
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    tx_hash: Optional[str] = None


@dataclass
class Position:
    """Current position in a market."""
    token_id: str
    side: str  # "yes" or "no"
    size: float
    avg_price: float
    condition_id: Optional[str] = None


class LiveExecutor:
    """
    Executor for live trading on Polymarket.
    
    Uses CLOB API for order placement and web3 for on-chain queries.
    """
    
    def __init__(
        self,
        polygon_rpc_url: str,
        private_key: str,
        api_key: str,
        api_secret: str,
        passphrase: str,
        socks5_proxy: Optional[str] = None,
    ):
        self.polygon_rpc_url = polygon_rpc_url
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.socks5_proxy = socks5_proxy
        
        # Web3 provider
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.address: Optional[str] = None
        
        # HTTP client for CLOB API
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # CLOB endpoints
        self.clob_url = "https://clob.polymarket.com"
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.data_api_url = "https://data-api.polymarket.com"
        
        # State
        self._connected = False
        self._positions: Dict[str, Position] = {}
    
    async def connect(self) -> bool:
        """Initialize web3 provider and wallet."""
        try:
            # Setup Web3
            self.w3 = Web3(Web3.HTTPProvider(self.polygon_rpc_url))
            if not self.w3.is_connected():
                logger.error("Failed to connect to Polygon RPC")
                return False
            
            # Setup account from private key
            if self.private_key.startswith("0x"):
                self.private_key = self.private_key[2:]
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
            
            logger.info(f"Connected to Polygon", address=self.address)
            
            # Setup HTTP client with optional SOCKS5 proxy
            transport = None
            if self.socks5_proxy:
                try:
                    import httpx_socks
                    transport = httpx_socks.AsyncProxyTransport.from_url(self.socks5_proxy)
                except ImportError:
                    logger.warning("httpx_socks not installed, using direct connection")
            
            if transport:
                self.http_client = httpx.AsyncClient(
                    transport=transport,
                    timeout=30,
                    verify=False,
                )
            else:
                self.http_client = httpx.AsyncClient(timeout=30)
            
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def get_usdc_balance(self) -> float:
        """Query wallet USDC balance."""
        if not self.w3 or not self.address:
            return 0.0
        
        try:
            usdc = self.w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                abi=ERC20_ABI,
            )
            balance_wei = usdc.functions.balanceOf(self.address).call()
            # USDC has 6 decimals
            return balance_wei / 1e6
        except Exception as e:
            logger.error(f"Failed to get USDC balance: {e}")
            return 0.0
    
    def _generate_auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate HMAC authentication headers for CLOB API."""
        timestamp = str(int(time.time()))
        
        # Create signature
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode(),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.b64encode(signature).decode()
        
        return {
            "POLY-API-KEY": self.api_key,
            "POLY-TIMESTAMP": timestamp,
            "POLY-SIGNATURE": signature_b64,
            "POLY-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
    
    async def place_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        size: float,  # in USDC
        price: float,  # 0.0 to 1.0
    ) -> OrderResult:
        """
        Place an order via CLOB API.
        
        Args:
            token_id: The token ID to trade
            side: "BUY" or "SELL"
            size: Position size in USDC
            price: Price (0.0 to 1.0)
        
        Returns:
            OrderResult with success status and order ID
        """
        if not self._connected or not self.http_client:
            return OrderResult(success=False, error="Not connected")
        
        try:
            # Build order payload
            order_payload = {
                "tokenID": token_id,
                "price": str(price),
                "size": str(size),
                "side": side.upper(),
                "orderType": "GTC",  # Good Till Cancelled
            }
            
            path = "/order"
            body = str(order_payload)  # Will be JSON serialized
            headers = self._generate_auth_headers("POST", path, body)
            
            # Note: The actual CLOB API requires signed orders
            # This is a simplified implementation - full implementation needs EIP-712 signing
            # For now, we'll use the simplified endpoint
            
            response = await self.http_client.post(
                f"{self.clob_url}{path}",
                json=order_payload,
                headers=headers,
            )
            
            if response.status_code == 200:
                data = response.json()
                order_id = data.get("orderID") or data.get("id")
                logger.info(f"Order placed", order_id=order_id, side=side, size=size, price=price)
                return OrderResult(success=True, order_id=order_id)
            else:
                error_msg = response.text[:200]
                logger.error(f"Order failed: {error_msg}")
                return OrderResult(success=False, error=error_msg)
                
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._connected or not self.http_client:
            return False
        
        try:
            path = f"/order/{order_id}"
            headers = self._generate_auth_headers("DELETE", path)
            
            response = await self.http_client.delete(
                f"{self.clob_url}{path}",
                headers=headers,
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        """Get current positions from Polymarket API."""
        if not self._connected or not self.http_client or not self.address:
            return []
        
        try:
            # Query positions API
            url = f"{self.data_api_url}/positions"
            params = {"user": self.address.lower()}
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                positions = []
                for pos in data:
                    positions.append(Position(
                        token_id=pos.get("asset", ""),
                        side="yes" if pos.get("outcome", "").lower() == "yes" else "no",
                        size=float(pos.get("size", 0)),
                        avg_price=float(pos.get("avgPrice", 0)),
                        condition_id=pos.get("conditionId"),
                    ))
                return positions
            return []
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    async def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trade history for confirmation."""
        if not self._connected or not self.http_client or not self.address:
            return []
        
        try:
            url = f"{self.data_api_url}/activity"
            params = {"user": self.address.lower(), "limit": limit}
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Get trades error: {e}")
            return []
    
    async def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: int = 30,
        poll_interval: int = 2,
    ) -> bool:
        """
        Poll for order fill confirmation.
        
        Returns True if order was filled, False if timed out or cancelled.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Check order status
                path = f"/order/{order_id}"
                headers = self._generate_auth_headers("GET", path)
                
                response = await self.http_client.get(
                    f"{self.clob_url}{path}",
                    headers=headers,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").upper()
                    
                    if status == "FILLED":
                        logger.info(f"Order filled", order_id=order_id)
                        return True
                    elif status in ["CANCELLED", "EXPIRED", "FAILED"]:
                        logger.warning(f"Order {status}", order_id=order_id)
                        return False
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Poll error: {e}")
                await asyncio.sleep(poll_interval)
        
        logger.warning(f"Order fill timeout", order_id=order_id)
        return False
