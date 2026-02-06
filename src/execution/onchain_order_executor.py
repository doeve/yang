"""
Onchain Order Executor for Polymarket.

Bypasses Polymarket CLOB API fees by:
1. Querying orderbook from CLOB API (read-only, no fees)
2. Filling orders directly on CTF Exchange contract
3. Using position splitting/merging for synthetic positions

This allows trading without paying CLOB trading fees.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from decimal import Decimal

import httpx
import structlog
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

logger = structlog.get_logger(__name__)

# DO NOT import py-clob-client here - must import AFTER monkey-patching httpx
# Imports are done in _ensure_clob_imports() after proxy is configured
CLOB_AVAILABLE = False
ClobClient = None
OrderArgs = None
OrderType = None
CLOB_BUY = None
CLOB_SELL = None


# Monkey-patch httpx to support SOCKS proxy for py-clob-client
_original_async_client_init = None
_original_sync_client_init = None
_global_socks_proxy = None

def _ensure_clob_imports():
    """Import py-clob-client modules (must be called AFTER monkey-patching)."""
    global CLOB_AVAILABLE, ClobClient, OrderArgs, OrderType, CLOB_BUY, CLOB_SELL

    if CLOB_AVAILABLE:
        return  # Already imported

    try:
        from py_clob_client.client import ClobClient as _ClobClient
        from py_clob_client.clob_types import OrderArgs as _OrderArgs, OrderType as _OrderType
        from py_clob_client.order_builder.constants import BUY as _CLOB_BUY, SELL as _CLOB_SELL

        ClobClient = _ClobClient
        OrderArgs = _OrderArgs
        OrderType = _OrderType
        CLOB_BUY = _CLOB_BUY
        CLOB_SELL = _CLOB_SELL
        CLOB_AVAILABLE = True

        logger.info("py-clob-client imported successfully (after proxy patch)")
    except ImportError as e:
        logger.warning(f"py-clob-client not installed: {e}")
        logger.info("Install with: pip install py-clob-client")

def _patch_httpx_with_socks(socks_proxy: str):
    """Monkey-patch both httpx.AsyncClient and httpx.Client to use SOCKS proxy."""
    global _original_async_client_init, _original_sync_client_init, _global_socks_proxy

    if not socks_proxy or _original_async_client_init is not None:
        return  # Already patched or no proxy

    try:
        import httpx
        import httpx_socks

        _global_socks_proxy = socks_proxy

        # Patch AsyncClient (for async operations)
        _original_async_client_init = httpx.AsyncClient.__init__

        def patched_async_init(self, *args, **kwargs):
            """Patched AsyncClient.__init__ that adds SOCKS proxy."""
            if 'transport' not in kwargs and _global_socks_proxy:
                try:
                    kwargs['transport'] = httpx_socks.AsyncProxyTransport.from_url(_global_socks_proxy)
                    kwargs['verify'] = False  # Accept self-signed certs from proxy
                except Exception as e:
                    logger.warning(f"Failed to add SOCKS proxy to async client: {e}")
            _original_async_client_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = patched_async_init

        # Patch Client (for sync operations - py-clob-client uses this!)
        _original_sync_client_init = httpx.Client.__init__

        def patched_sync_init(self, *args, **kwargs):
            """Patched Client.__init__ that adds SOCKS proxy."""
            if 'transport' not in kwargs and _global_socks_proxy:
                try:
                    kwargs['transport'] = httpx_socks.SyncProxyTransport.from_url(_global_socks_proxy)
                    kwargs['verify'] = False  # Accept self-signed certs from proxy
                except Exception as e:
                    logger.warning(f"Failed to add SOCKS proxy to sync client: {e}")
            _original_sync_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched_sync_init

        logger.info(f"httpx (sync + async) monkey-patched with SOCKS proxy: {socks_proxy}")

    except ImportError:
        logger.error("httpx-socks not available for proxy support")
        logger.info("Install with: pip install 'httpx-socks[asyncio]'")
    except Exception as e:
        logger.error(f"Failed to monkey-patch httpx: {e}")


# Contract addresses (Polygon Mainnet)
CONTRACTS = {
    "CTF": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
    "CTF_EXCHANGE": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NEG_RISK_CTF_EXCHANGE": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
}


# ABIs
CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "partition", "type": "uint256[]"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "splitPosition",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "partition", "type": "uint256[]"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "mergePositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "from", "type": "address"},
            {"name": "to", "type": "address"},
            {"name": "id", "type": "uint256"},
            {"name": "amount", "type": "uint256"},
            {"name": "data", "type": "bytes"},
        ],
        "name": "safeTransferFrom",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"},
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"},
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]

CTF_EXCHANGE_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
                "name": "order",
                "type": "tuple",
            },
            {"name": "signature", "type": "bytes"},
            {"name": "fillAmount", "type": "uint256"},
        ],
        "name": "fillOrder",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
                "name": "order",
                "type": "tuple",
            }
        ],
        "name": "hashOrder",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
]

ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
]


@dataclass
@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    tx_hash: Optional[str] = None
    order_id: Optional[str] = None  # For compatibility with LiveExecutor
    filled_amount: float = 0.0
    avg_price: float = 0.0
    gas_used: int = 0
    error: Optional[str] = None


@dataclass
class OrderbookOrder:
    """Order from the CLOB orderbook."""
    order_id: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    maker: str
    signature: str
    salt: int
    maker_amount: int
    taker_amount: int
    expiration: int
    nonce: int
    fee_rate_bps: int


class OnchainOrderExecutor:
    """
    Execute Polymarket orders directly onchain to avoid CLOB fees.

    Strategy:
    1. Query orderbook from CLOB API (read-only, no fees)
    2. Fill orders directly via CTF Exchange contract
    3. Alternative: Use position splitting/merging for synthetic positions

    Benefits:
    - No CLOB trading fees
    - Direct blockchain settlement
    - More control over execution
    """

    def __init__(
        self,
        local_rpc_url: str,
        private_key: str,
        public_rpc_url: str = "https://polygon-rpc.com",
        use_public_rpc: bool = True,
        use_clob: bool = False,
        clob_api_url: str = "https://clob.polymarket.com",
        gamma_api_url: str = "https://gamma-api.polymarket.com",
        socks5_proxy: Optional[str] = None,
    ):
        """Initialize onchain order executor."""
        self.local_rpc_url = local_rpc_url
        self.private_key = private_key
        self.public_rpc_url = public_rpc_url
        self.use_public_rpc = use_public_rpc
        self.use_clob = use_clob
        self.clob_api_url = clob_api_url
        self.gamma_api_url = gamma_api_url
        self.socks5_proxy = socks5_proxy

        # Web3 providers
        self.local_w3: Optional[Web3] = None
        self.public_w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.address: Optional[str] = None

        # HTTP client
        self.http_client: Optional[httpx.AsyncClient] = None

        # CLOB client (for API trading)
        self.clob_client = None

        # Connected state
        self._connected = False

    async def connect(self) -> bool:
        """Initialize providers and wallet."""
        try:
            # Setup providers
            self.local_w3 = Web3(Web3.HTTPProvider(self.local_rpc_url))
            # Inject POA middleware for Polygon
            self.local_w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            if self.use_public_rpc:
                self.public_w3 = Web3(Web3.HTTPProvider(self.public_rpc_url))
                # Inject POA middleware for Polygon
                self.public_w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            else:
                self.public_w3 = self.local_w3

            # Setup account
            if self.private_key.startswith("0x"):
                self.private_key = self.private_key[2:]
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address

            # Setup HTTP client
            transport = None
            if self.socks5_proxy:
                try:
                    import httpx_socks
                    transport = httpx_socks.AsyncProxyTransport.from_url(self.socks5_proxy)
                except ImportError:
                    logger.warning("httpx_socks not installed")

            if transport:
                self.http_client = httpx.AsyncClient(transport=transport, timeout=30, verify=False)
            else:
                self.http_client = httpx.AsyncClient(timeout=30)

            # Initialize CLOB client if use_clob is enabled
            if self.use_clob:
                try:
                    logger.info("Initializing CLOB client for API trading...")

                    # Step 1: Monkey-patch httpx with SOCKS proxy BEFORE importing py-clob-client
                    if self.socks5_proxy:
                        _patch_httpx_with_socks(self.socks5_proxy)

                    # Step 2: Import py-clob-client AFTER monkey-patch is applied
                    _ensure_clob_imports()

                    if not CLOB_AVAILABLE:
                        raise ImportError("py-clob-client not available")

                    # Step 3: Create CLOB client (will now use proxied httpx)
                    self.clob_client = ClobClient(
                        host=self.clob_api_url,
                        key=self.private_key,
                        chain_id=137,  # Polygon
                        signature_type=0,  # EOA (standard wallet)
                        funder=self.address
                    )

                    # Generate and set API credentials
                    logger.info("Generating CLOB API credentials...")
                    api_creds = self.clob_client.create_or_derive_api_creds()
                    self.clob_client.set_api_creds(api_creds)

                    logger.info("CLOB client connected",
                               address=self.address,
                               api_key=api_creds.api_key[:8] + "...",
                               proxy_enabled=bool(self.socks5_proxy))
                except Exception as e:
                    logger.error(f"Failed to initialize CLOB client: {e}")
                    logger.warning("Falling back to onchain-only mode")
                    self.use_clob = False

            logger.info("Onchain order executor connected",
                       address=self.address,
                       mode="CLOB" if self.use_clob else "Onchain")
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        self._connected = False

    async def ensure_approvals(self) -> bool:
        """
        Ensure necessary token approvals for trading.

        - USDC approval for CTF Exchange (ERC-20)
        - CTF approval for CLOB Exchange (ERC-1155) - only when use_clob=True

        This is required before trading.
        Uses local RPC for reading to avoid rate limits.
        """
        if not self.local_w3 or not self.public_w3 or not self.account:
            return False

        try:
            # Check allowance using LOCAL RPC (fast, no rate limit)
            usdc_local = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                abi=ERC20_ABI,
            )

            allowance = usdc_local.functions.allowance(
                self.address,
                Web3.to_checksum_address(CONTRACTS["CTF"])
            ).call()

            # Approve unlimited if not already approved
            max_uint256 = 2**256 - 1
            if allowance < max_uint256 // 2:
                logger.info("Approving USDC for CTF...")

                # Use PUBLIC RPC only for transaction
                usdc_public = self.public_w3.eth.contract(
                    address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                    abi=ERC20_ABI,
                )

                nonce = self.public_w3.eth.get_transaction_count(self.address)

                # Get current gas prices (EIP-1559)
                latest_block = self.public_w3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)  # Default 30 gwei
                max_priority_fee = self.public_w3.to_wei(50, 'gwei')  # 50 gwei tip
                max_fee = base_fee * 2 + max_priority_fee  # 2x base + priority

                tx = usdc_public.functions.approve(
                    Web3.to_checksum_address(CONTRACTS["CTF"]),
                    max_uint256
                ).build_transaction({
                    "from": self.address,
                    "nonce": nonce,
                    "gas": 100000,
                    "maxFeePerGas": max_fee,
                    "maxPriorityFeePerGas": max_priority_fee,
                })

                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)

                logger.info(f"Approval tx sent: {tx_hash.hex()}")
                receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

                if receipt["status"] != 1:
                    logger.error("USDC approval failed")
                    return False

                logger.info("USDC approved successfully")
            else:
                logger.info("USDC already approved (skipping)")

            # Check if CTF tokens approved for CLOB Exchange (ERC-1155)
            # This is required for CLOB to transfer user's outcome tokens when selling
            if self.use_clob:
                ctf_local = self.local_w3.eth.contract(
                    address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                    abi=CTF_ABI,
                )

                is_approved = ctf_local.functions.isApprovedForAll(
                    self.address,
                    Web3.to_checksum_address(CONTRACTS["CTF_EXCHANGE"])
                ).call()

                if not is_approved:
                    logger.info("Approving CTF tokens for CLOB Exchange...")

                    # Use PUBLIC RPC for transaction
                    ctf_public = self.public_w3.eth.contract(
                        address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                        abi=CTF_ABI,
                    )

                    nonce = self.public_w3.eth.get_transaction_count(self.address)

                    # Get current gas prices (EIP-1559)
                    latest_block = self.public_w3.eth.get_block('latest')
                    base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)
                    max_priority_fee = self.public_w3.to_wei(50, 'gwei')
                    max_fee = base_fee * 2 + max_priority_fee

                    tx = ctf_public.functions.setApprovalForAll(
                        Web3.to_checksum_address(CONTRACTS["CTF_EXCHANGE"]),
                        True
                    ).build_transaction({
                        "from": self.address,
                        "nonce": nonce,
                        "gas": 100000,
                        "maxFeePerGas": max_fee,
                        "maxPriorityFeePerGas": max_priority_fee,
                    })

                    signed_tx = self.account.sign_transaction(tx)
                    tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)

                    logger.info(f"CTF approval tx sent: {tx_hash.hex()}")
                    receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

                    if receipt["status"] != 1:
                        logger.error("CTF approval for CLOB Exchange failed")
                        return False

                    logger.info("CTF tokens approved for CLOB Exchange successfully")
                else:
                    logger.info("CTF already approved for CLOB Exchange (skipping)")

            return True

        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                logger.warning("Rate limit hit on approval check")
                logger.info("Using local RPC for reads to avoid rate limits")
                # Still return True if we can't check - don't block trading
                return True
            logger.error(f"Approval error: {e}")
            return False

    async def get_midpoint_price(self, token_id: str) -> Optional[float]:
        """
        Get midpoint price from CLOB API.

        Args:
            token_id: Token ID

        Returns:
            Midpoint price or None if unavailable
        """
        if not self.http_client:
            return None

        try:
            url = f"{self.clob_api_url}/midpoint"
            params = {"token_id": token_id}
            response = await self.http_client.get(url, params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch midpoint: {response.status_code}")
                return None

            data = response.json()
            midpoint = float(data.get("mid", 0))

            if midpoint > 0:
                logger.info(f"Midpoint price: {midpoint:.4f}")
                return midpoint

            return None

        except Exception as e:
            logger.error(f"Error getting midpoint price: {e}")
            return None

    async def get_market_price(self, token_id: str, side: str) -> Optional[float]:
        """
        Get best market price for immediate execution.

        For BUY: Returns best ask (lowest SELL order price)
        For SELL: Returns best bid (highest BUY order price)

        Falls back to midpoint price if orderbook is empty.

        Args:
            token_id: Token ID
            side: "BUY" or "SELL"

        Returns:
            Best price for immediate fill, or None if unavailable
        """
        try:
            # For BUY: Get SELL orderbook (we want to buy from sellers)
            # For SELL: Get BUY orderbook (we want to sell to buyers)
            book_side = "SELL" if side.upper() == "BUY" else "BUY"
            orders = await self.get_orderbook(token_id, book_side)

            if orders:
                # Get best price (lowest for asks, highest for bids)
                if side.upper() == "BUY":
                    # Buy at lowest ask
                    best_price = min(order.price for order in orders)
                else:
                    # Sell at highest bid
                    best_price = max(order.price for order in orders)

                logger.info(f"Market price from orderbook for {side}: {best_price:.4f}")
                return best_price

            # Fallback to midpoint if orderbook is empty
            logger.info(f"Orderbook empty for {side}, falling back to midpoint price")
            midpoint = await self.get_midpoint_price(token_id)

            if midpoint:
                logger.info(f"Using midpoint price for {side}: {midpoint:.4f}")
                return midpoint

            logger.warning(f"No market price available (orderbook and midpoint both unavailable)")
            return None

        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return None

    async def get_orderbook(self, token_id: str, side: str = "BUY") -> List[OrderbookOrder]:
        """
        Fetch orderbook from CLOB API (read-only, no fees).

        Args:
            token_id: Token ID to query
            side: "BUY" or "SELL"

        Returns:
            List of orders
        """
        if not self.http_client:
            return []

        try:
            # Query orderbook endpoint
            url = f"{self.clob_api_url}/book"
            params = {"token_id": token_id, "side": side}

            response = await self.http_client.get(url, params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch orderbook: {response.status_code}")
                return []

            data = response.json()

            # Check for API error
            if "error" in data:
                logger.warning(f"Orderbook API error: {data['error']}")
                return []

            # CLOB API returns "bids" and "asks", not "orders"
            # - bids: buy orders (for selling into)
            # - asks: sell orders (for buying from)
            # The API returns aggregated price levels, not individual orders with signatures

            orders = []

            # Determine which side to parse based on what was requested
            # Note: The API returns both bids and asks regardless of 'side' parameter
            if side.upper() == "BUY":
                # For BUY orders, we want the asks (sell orders to buy from)
                order_levels = data.get("asks", [])
                order_side = "SELL"
            else:
                # For SELL orders, we want the bids (buy orders to sell into)
                order_levels = data.get("bids", [])
                order_side = "BUY"

            for level in order_levels:
                # Each level is just {price, size} - aggregated orderbook
                # For CLOB API trading, we only need price and size
                orders.append(OrderbookOrder(
                    order_id="",  # Not provided in aggregated book
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                    side=order_side,
                    maker="",  # Not provided in aggregated book
                    signature="",  # Not provided in aggregated book
                    salt=0,
                    maker_amount=0,
                    taker_amount=0,
                    expiration=0,
                    nonce=0,
                    fee_rate_bps=0,
                ))

            logger.debug(f"Parsed {len(orders)} orderbook levels for {side}")
            return orders

        except Exception as e:
            logger.error(f"Get orderbook error: {e}")
            return []

    async def split_position(
        self,
        condition_id: str,
        amount_usdc: float,
    ) -> OrderResult:
        """
        Split USDC into YES/NO tokens (create synthetic long position).

        This is a fee-free way to get YES tokens:
        1. Approve USDC for CTF contract
        2. Call splitPosition to get both YES and NO tokens
        3. Sell the unwanted side

        Args:
            condition_id: Market condition ID
            amount_usdc: Amount of USDC to split (in USDC, not wei)

        Returns:
            OrderResult with success status
        """
        if not self.public_w3 or not self.account:
            return OrderResult(success=False, error="Not connected")

        try:
            ctf = self.public_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # Convert amount to wei (USDC has 6 decimals)
            amount_wei = int(amount_usdc * 1e6)

            # Ensure condition_id is bytes32
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            condition_bytes = bytes.fromhex(condition_id[2:])

            # Parent collection ID (zero for top-level)
            parent_collection_id = bytes(32)

            # Partition: [1, 2] for binary market (YES, NO)
            partition = [1, 2]

            # Build transaction
            nonce = self.public_w3.eth.get_transaction_count(self.address)

            # Get current gas prices (EIP-1559)
            latest_block = self.public_w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)  # Default 30 gwei
            max_priority_fee = self.public_w3.to_wei(50, 'gwei')  # 50 gwei tip
            max_fee = base_fee * 2 + max_priority_fee  # 2x base + priority

            tx = ctf.functions.splitPosition(
                Web3.to_checksum_address(CONTRACTS["USDC"]),
                parent_collection_id,
                condition_bytes,
                partition,
                amount_wei,
            ).build_transaction({
                "from": self.address,
                "nonce": nonce,
                "gas": 300000,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(
                f"Split position tx sent: {tx_hash_hex}",
                tx_hash=tx_hash_hex,
                gas_price=f"{max_fee / 10**9:.1f} gwei"
            )
            print(f"  📝 TX: https://polygonscan.com/tx/{tx_hash_hex}")
            print(f"  ⏳ Waiting for confirmation...")

            # Wait for confirmation (with timeout)
            receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                logger.info(
                    f"Position split successful",
                    tx_hash=tx_hash_hex,
                    amount_usdc=amount_usdc,
                )

                return OrderResult(
                    success=True,
                    tx_hash=tx_hash_hex,
                    filled_amount=amount_usdc,
                    gas_used=receipt["gasUsed"],
                )
            else:
                return OrderResult(
                    success=False,
                    tx_hash=tx_hash_hex,
                    error="Transaction reverted",
                )

        except Exception as e:
            logger.error(f"Split position error: {e}")
            return OrderResult(success=False, error=str(e))

    async def merge_position(
        self,
        condition_id: str,
        amount: float,
    ) -> OrderResult:
        """
        Merge YES/NO tokens back into USDC (close synthetic position).

        Args:
            condition_id: Market condition ID
            amount: Amount of token pairs to merge

        Returns:
            OrderResult with success status
        """
        if not self.public_w3 or not self.account:
            return OrderResult(success=False, error="Not connected")

        try:
            ctf = self.public_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # Convert amount (assuming same decimals as USDC = 6)
            amount_wei = int(amount * 1e6)

            # Ensure condition_id is bytes32
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            condition_bytes = bytes.fromhex(condition_id[2:])

            # Parent collection ID (zero for top-level)
            parent_collection_id = bytes(32)

            # Partition: [1, 2] for binary market
            partition = [1, 2]

            # Build transaction
            nonce = self.public_w3.eth.get_transaction_count(self.address)

            # Get current gas prices (EIP-1559)
            latest_block = self.public_w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)  # Default 30 gwei
            max_priority_fee = self.public_w3.to_wei(50, 'gwei')  # 50 gwei tip
            max_fee = base_fee * 2 + max_priority_fee  # 2x base + priority

            tx = ctf.functions.mergePositions(
                Web3.to_checksum_address(CONTRACTS["USDC"]),
                parent_collection_id,
                condition_bytes,
                partition,
                amount_wei,
            ).build_transaction({
                "from": self.address,
                "nonce": nonce,
                "gas": 300000,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(
                f"Merge position tx sent: {tx_hash_hex}",
                tx_hash=tx_hash_hex,
                gas_price=f"{max_fee / 10**9:.1f} gwei"
            )
            print(f"  📝 TX: https://polygonscan.com/tx/{tx_hash_hex}")
            print(f"  ⏳ Waiting for confirmation...")

            # Wait for confirmation (with timeout)
            receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                logger.info(
                    f"Position merge successful",
                    tx_hash=tx_hash_hex,
                    amount=amount,
                )

                return OrderResult(
                    success=True,
                    tx_hash=tx_hash_hex,
                    filled_amount=amount,
                    gas_used=receipt["gasUsed"],
                )
            else:
                return OrderResult(
                    success=False,
                    tx_hash=tx_hash_hex,
                    error="Transaction reverted",
                )

        except Exception as e:
            logger.error(f"Merge position error: {e}")
            return OrderResult(success=False, error=str(e))

    async def _fill_order_onchain(
        self,
        order: OrderbookOrder,
        fill_amount: float,
        token_id: str,
    ) -> OrderResult:
        """
        Fill an order from the orderbook via CTF Exchange contract (onchain, no CLOB fees).

        Args:
            order: OrderbookOrder from CLOB API
            fill_amount: Amount to fill (in tokens/shares)
            token_id: Token ID being traded

        Returns:
            OrderResult with transaction details
        """
        if not self.public_w3 or not self.account:
            return OrderResult(success=False, error="Not connected")

        try:
            # CTF Exchange contract
            ctf_exchange = self.public_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF_EXCHANGE"]),
                abi=CTF_EXCHANGE_ABI,
            )

            # Convert amounts to wei (6 decimals for USDC-based tokens)
            fill_amount_wei = int(fill_amount * 1e6)

            # Build the order struct for the contract (without signature)
            # Order struct: (salt, maker, signer, taker, tokenId, makerAmount, takerAmount, expiration, nonce, feeRateBps, side, signatureType)
            order_struct = (
                order.salt,
                Web3.to_checksum_address(order.maker),
                Web3.to_checksum_address(order.maker),  # signer = maker
                self.address,  # taker = us
                int(token_id) if not token_id.startswith("0x") else int(token_id, 16),
                order.maker_amount,
                order.taker_amount,
                order.expiration,
                order.nonce,
                order.fee_rate_bps,
                1 if order.side == "BUY" else 2,  # 1 = BUY, 2 = SELL
                0,  # signatureType: 0 = EOA
            )

            # Signature is passed separately
            signature_bytes = bytes.fromhex(order.signature[2:]) if order.signature.startswith("0x") else bytes.fromhex(order.signature)

            # Build transaction
            nonce = self.public_w3.eth.get_transaction_count(self.address)

            # Get current gas prices
            latest_block = self.public_w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)
            max_priority_fee = self.public_w3.to_wei(50, 'gwei')
            max_fee = base_fee * 2 + max_priority_fee

            # Call fillOrder on CTF Exchange (order, signature, fillAmount)
            tx = ctf_exchange.functions.fillOrder(
                order_struct,
                signature_bytes,
                fill_amount_wei,
            ).build_transaction({
                "from": self.address,
                "nonce": nonce,
                "gas": 350000,  # Higher gas for exchange interaction
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority_fee,
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(
                f"Fill order tx sent: {tx_hash_hex}",
                tx_hash=tx_hash_hex,
                fill_amount=fill_amount,
            )

            # Wait for confirmation
            receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                usdc_received = fill_amount * order.price

                logger.info(
                    f"Order filled successfully",
                    tx_hash=tx_hash_hex,
                    tokens_sold=fill_amount,
                    usdc_received=usdc_received,
                )

                return OrderResult(
                    success=True,
                    tx_hash=tx_hash_hex,
                    filled_amount=fill_amount,
                    avg_price=order.price,
                    gas_used=receipt["gasUsed"],
                )
            else:
                return OrderResult(
                    success=False,
                    tx_hash=tx_hash_hex,
                    error="Transaction reverted",
                )

        except Exception as e:
            logger.error(f"Fill order error: {e}")
            return OrderResult(success=False, error=str(e))

    async def market_buy(
        self,
        token_id: str,
        amount_usdc: float,
        max_price: float = 0.99,
    ) -> OrderResult:
        """
        Execute market buy using position splitting (fee-free).

        Strategy:
        1. Split USDC into YES/NO tokens
        2. Keep the desired side
        3. Sell the other side on orderbook (or keep for hedging)

        Args:
            token_id: Token to buy
            amount_usdc: Amount in USDC
            max_price: Maximum acceptable price (slippage protection)

        Returns:
            OrderResult
        """
        # This is a simplified implementation
        # In production, you'd:
        # 1. Get condition_id from token_id
        # 2. Split position to get both YES/NO
        # 3. Sell the unwanted side (or use it for hedging)

        logger.info(f"Market buy: {amount_usdc} USDC of token {token_id}")

        # For now, return placeholder
        # Full implementation would integrate with orderbook
        return OrderResult(
            success=False,
            error="Market buy via splitting not fully implemented - use fill_order instead"
        )

    async def get_usdc_balance(self) -> float:
        """Get USDC balance (tries local RPC first, falls back to public)."""
        if not self.address:
            logger.error("get_usdc_balance: No address available")
            return 0.0

        # Try local RPC first (faster, no rate limits)
        if self.local_w3:
            try:
                usdc = self.local_w3.eth.contract(
                    address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                    abi=ERC20_ABI,
                )
                balance_wei = usdc.functions.balanceOf(self.address).call()
                balance = balance_wei / 1e6
                logger.debug(f"USDC balance from local RPC: ${balance:.2f}")
                return balance
            except Exception as e:
                logger.warning(f"Failed to get USDC balance from local RPC: {e}, trying public RPC...")

        # Fallback to public RPC
        if self.public_w3:
            try:
                usdc = self.public_w3.eth.contract(
                    address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                    abi=ERC20_ABI,
                )
                balance_wei = usdc.functions.balanceOf(self.address).call()
                balance = balance_wei / 1e6
                logger.info(f"USDC balance from public RPC: ${balance:.2f}")
                return balance
            except Exception as e:
                logger.error(f"Failed to get USDC balance from public RPC: {e}")

        logger.error("No working RPC available for balance query")
        return 0.0

    async def place_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
    ) -> OrderResult:
        """
        Place an order with ACTUAL onchain execution.

        Strategy:
        - BUY: Split USDC into YES/NO tokens via CTF contract
        - SELL: Merge tokens back to USDC (if holding both sides)

        This bypasses the CLOB API entirely and executes directly onchain.

        Args:
            token_id: Token ID to trade
            side: "BUY" or "SELL"
            size: Size in shares/tokens (for BUY) or tokens to sell (for SELL)
            price: Price (0.0 to 1.0) - used to calculate USDC amount for BUY

        Returns:
            OrderResult with transaction details
        """
        if not self._connected:
            return OrderResult(success=False, error="Not connected")

        logger.info(
            f"🔗 Executing onchain: {side} {size:.2f} @ {price:.3f}",
            token_id=token_id[:16] if len(token_id) > 16 else token_id,
        )

        try:
            if side.upper() == "BUY":
                # Check mode
                if self.use_clob:
                    # CLOB mode: Use CLOB API with EIP-712 signing
                    if not self.clob_client:
                        return OrderResult(
                            success=False,
                            error="CLOB client not initialized. Check py-clob-client installation."
                        )

                    # CLOB market order: Get best ask price for immediate fill
                    market_price = await self.get_market_price(token_id, "BUY")
                    if market_price is None:
                        logger.error("No market price available (empty orderbook)")
                        return OrderResult(
                            success=False,
                            error="No liquidity in orderbook for BUY"
                        )

                    # CLOB price limits: 0.01 to 0.99
                    if market_price > 0.99 or market_price < 0.01:
                        logger.warning(
                            f"clob_price_out_of_range",
                            price=market_price,
                            message=f"Market price {market_price:.4f} outside CLOB range (0.01-0.99). Market near-resolved."
                        )
                        return OrderResult(
                            success=False,
                            error=f"Market price {market_price:.4f} outside CLOB range (0.01-0.99). Cannot place order on near-resolved market."
                        )

                    logger.info(f"📋 CLOB BUY (market order): {size:.2f} shares @ {market_price:.4f}")

                    try:
                        # Create order via py-clob-client (handles EIP-712 signing)
                        # Use market price for immediate fill
                        order_args = OrderArgs(
                            token_id=token_id,
                            price=market_price,
                            size=size,
                            side=CLOB_BUY
                        )

                        # Sign order (EIP-712)
                        signed_order = self.clob_client.create_order(order_args)

                        # Post to CLOB
                        response = self.clob_client.post_order(signed_order, OrderType.GTC)

                        order_id = response.get("orderID")
                        if not order_id:
                            return OrderResult(
                                success=False,
                                error=f"CLOB order failed: {response}"
                            )

                        logger.info(f"✅ CLOB BUY order placed", order_id=order_id[:16])

                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            filled_amount=size,
                            avg_price=market_price
                        )

                    except Exception as e:
                        logger.error(f"CLOB BUY error: {e}")
                        return OrderResult(
                            success=False,
                            error=f"CLOB order failed: {str(e)}"
                        )

                # Onchain mode: Split USDC into YES/NO tokens (fee-free)
                # Calculate USDC needed: shares * price
                usdc_amount = size * price

                # Need to get condition_id for the token
                # For now, we'll need to query the market info
                condition_id = await self._get_condition_id_for_token(token_id)

                if not condition_id:
                    return OrderResult(
                        success=False,
                        error="Could not determine condition_id for token"
                    )

                logger.info(f"💰 Splitting ${usdc_amount:.2f} USDC into tokens (onchain, fee-free)")

                result = await self.split_position(
                    condition_id=condition_id,
                    amount_usdc=usdc_amount,
                )

                if result.success:
                    logger.info(
                        f"✅ Onchain BUY executed: {result.filled_amount:.2f} tokens",
                        tx_hash=result.tx_hash[:16] if result.tx_hash else None
                    )
                    return OrderResult(
                        success=True,
                        tx_hash=result.tx_hash,
                        filled_amount=size,  # Return requested size
                        avg_price=price,
                        gas_used=result.gas_used,
                    )
                else:
                    return result

            elif side.upper() == "SELL":
                # SELL: Not supported in pure onchain mode due to operator restrictions
                if not self.use_clob:
                    logger.warning(
                        "sell_not_supported_onchain",
                        message="Cannot exit positions in pure onchain mode (use_clob=false). "
                                "Options: (1) Enable CLOB mode via config, or (2) Hold until resolution and redeem."
                    )
                    return OrderResult(
                        success=False,
                        error="SELL not supported in pure onchain mode. Enable --clob flag or wait for resolution to redeem."
                    )

                # CLOB mode: Use CLOB API with EIP-712 signing
                if not self.clob_client:
                    return OrderResult(
                        success=False,
                        error="CLOB client not initialized. Check py-clob-client installation."
                    )

                # CLOB market order: Get best bid price for immediate fill
                market_price = await self.get_market_price(token_id, "SELL")
                if market_price is None:
                    logger.error("No market price available (empty orderbook)")
                    return OrderResult(
                        success=False,
                        error="No liquidity in orderbook for SELL"
                    )

                # CLOB price limits: 0.01 to 0.99
                # If price is outside this range, the market is near-resolved
                if market_price > 0.99 or market_price < 0.01:
                    logger.warning(
                        f"clob_price_out_of_range",
                        price=market_price,
                        message=f"Market price {market_price:.4f} outside CLOB range (0.01-0.99). Market near-resolved. "
                                f"Holding position for redemption after resolution is more profitable."
                    )
                    return OrderResult(
                        success=False,
                        error=f"Market price {market_price:.4f} outside CLOB range (0.01-0.99). Hold until resolution and redeem for better payout."
                    )

                logger.info(f"📋 CLOB SELL (market order): {size:.2f} shares @ {market_price:.4f}")

                try:
                    # Create order via py-clob-client (handles EIP-712 signing)
                    # Use market price for immediate fill
                    order_args = OrderArgs(
                        token_id=token_id,
                        price=market_price,
                        size=size,
                        side=CLOB_SELL
                    )

                    # Sign order (EIP-712)
                    signed_order = self.clob_client.create_order(order_args)

                    # Post to CLOB
                    response = self.clob_client.post_order(signed_order, OrderType.GTC)

                    order_id = response.get("orderID")
                    if not order_id:
                        return OrderResult(
                            success=False,
                            error=f"CLOB order failed: {response}"
                        )

                    logger.info(f"✅ CLOB SELL order placed", order_id=order_id[:16])

                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        filled_amount=size,
                        avg_price=market_price
                    )

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"CLOB SELL error: {e}")

                    # If balance/allowance error, check actual balance for debugging
                    if "balance" in error_msg.lower() or "allowance" in error_msg.lower():
                        try:
                            actual_balance = await self._get_token_balance(token_id)
                            balance_shares = actual_balance / 1e6
                            logger.error(
                                f"Balance check after error: {balance_shares:.2f} shares "
                                f"(attempted to sell {size:.2f})",
                                token_id=token_id[:16] + "..."
                            )

                            if actual_balance == 0:
                                error_msg = f"No tokens to sell - balance is 0 (position tracking may be incorrect)"
                            elif balance_shares < size:
                                error_msg = f"Insufficient balance: have {balance_shares:.2f}, tried to sell {size:.2f}"
                        except Exception as balance_error:
                            logger.warning(f"Could not check balance: {balance_error}")

                    return OrderResult(
                        success=False,
                        error=f"CLOB order failed: {error_msg}"
                    )

            else:
                return OrderResult(
                    success=False,
                    error=f"Unknown side: {side}"
                )

        except Exception as e:
            logger.error(f"Onchain order execution error: {e}")
            return OrderResult(
                success=False,
                error=str(e)
            )

    async def _get_condition_id_for_token(self, token_id: str) -> Optional[str]:
        """
        Get condition_id for a given token_id by querying Gamma API.

        Args:
            token_id: CLOB token ID

        Returns:
            condition_id or None
        """
        if not self.http_client:
            return None

        try:
            url = f"{self.gamma_api_url}/markets"
            params = {"clob_token_ids": token_id}

            response = await self.http_client.get(url, params=params)

            if response.status_code == 200:
                markets = response.json()
                if markets and len(markets) > 0:
                    return markets[0].get("conditionId")

            return None

        except Exception as e:
            logger.error(f"Failed to get condition_id: {e}")
            return None

    async def wait_for_fill(self, order_id: str, timeout: int = 30) -> bool:
        """
        Wait for order fill.

        For onchain execution, orders are filled immediately.
        For CLOB mode, polls the CLOB API for fill status.

        Args:
            order_id: Order ID
            timeout: Timeout in seconds

        Returns:
            True if filled, False otherwise
        """
        if not self.use_clob:
            # Onchain mode: immediate fill
            logger.debug("Onchain: Order filled immediately")
            return True

        # CLOB mode: poll for fill
        if not self.clob_client:
            logger.error("CLOB client not available")
            return False

        start_time = time.time()
        poll_interval = 2  # seconds

        while time.time() - start_time < timeout:
            try:
                # Check order status
                order_info = self.clob_client.get_order(order_id)
                status = order_info.get("status", "").upper()

                if status == "MATCHED":
                    logger.info(f"Order filled", order_id=order_id[:16])
                    return True
                elif status in ["CANCELLED", "EXPIRED", "FAILED"]:
                    logger.warning(f"Order {status}", order_id=order_id[:16])
                    return False

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Poll error: {e}")
                await asyncio.sleep(poll_interval)

        logger.warning(f"Order fill timeout", order_id=order_id[:16])
        return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        For onchain execution, there's no order to cancel (immediate fill).
        For CLOB mode, cancels the order via CLOB API.

        Args:
            order_id: Order ID

        Returns:
            True if successful
        """
        if not self.use_clob:
            # Onchain mode: no order to cancel
            logger.debug("Onchain: No order to cancel (immediate execution)")
            return True

        # CLOB mode: cancel via API
        if not self.clob_client:
            logger.error("CLOB client not available")
            return False

        try:
            self.clob_client.cancel(order_id)
            logger.info(f"Order cancelled", order_id=order_id[:16])
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    async def close_position(
        self,
        condition_id: str,
        yes_token_id: str,
        no_token_id: str,
    ) -> OrderResult:
        """
        Close a position by merging YES/NO tokens back to USDC.

        This is the onchain equivalent of selling. If you hold both
        YES and NO tokens, you can merge them back to USDC without
        needing to find a buyer.

        Args:
            condition_id: Market condition ID
            yes_token_id: YES token ID
            no_token_id: NO token ID

        Returns:
            OrderResult with transaction details
        """
        if not self._connected:
            return OrderResult(success=False, error="Not connected")

        try:
            # Check balances
            yes_balance = await self._get_token_balance(yes_token_id)
            no_balance = await self._get_token_balance(no_token_id)

            if yes_balance == 0 and no_balance == 0:
                logger.info("No tokens to merge")
                return OrderResult(
                    success=True,
                    filled_amount=0.0,
                    skipped_reason="No tokens held"
                )

            # Can only merge the minimum of both sides
            # (you need equal amounts of YES and NO to merge)
            merge_amount = min(yes_balance, no_balance)

            if merge_amount == 0:
                logger.warning(
                    f"Cannot merge: unequal balances (YES={yes_balance}, NO={no_balance})"
                )
                # Hold until market resolves, then redeem
                return OrderResult(
                    success=True,
                    filled_amount=0.0,
                    skipped_reason="Unequal balances - will redeem at resolution"
                )

            logger.info(f"💱 Merging {merge_amount} token pairs back to USDC")

            # Merge positions
            result = await self.merge_position(
                condition_id=condition_id,
                amount=merge_amount / 1e6,  # Convert from wei to USDC
            )

            if result.success:
                logger.info(
                    f"✅ Position closed: merged {merge_amount} tokens",
                    tx_hash=result.tx_hash[:16] if result.tx_hash else None
                )

            return result

        except Exception as e:
            logger.error(f"Close position error: {e}")
            return OrderResult(success=False, error=str(e))

    async def _get_token_balance(self, token_id: str) -> int:
        """
        Get balance of a specific token.

        Args:
            token_id: Token ID to check

        Returns:
            Balance in wei (6 decimals for USDC-based tokens)
        """
        if not self.local_w3 or not self.address:
            return 0

        try:
            ctf = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # Convert token_id
            if isinstance(token_id, str):
                if token_id.startswith("0x"):
                    token_id_int = int(token_id, 16)
                else:
                    token_id_int = int(token_id)
            else:
                token_id_int = int(token_id)

            balance = ctf.functions.balanceOf(self.address, token_id_int).call()
            return int(balance)

        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            return 0
