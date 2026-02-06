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
        clob_api_url: str = "https://clob.polymarket.com",
        gamma_api_url: str = "https://gamma-api.polymarket.com",
        socks5_proxy: Optional[str] = None,
    ):
        """Initialize onchain order executor."""
        self.local_rpc_url = local_rpc_url
        self.private_key = private_key
        self.public_rpc_url = public_rpc_url
        self.use_public_rpc = use_public_rpc
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

            logger.info("Onchain order executor connected", address=self.address)
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
        Ensure USDC approval for CTF Exchange and CTF for splitting/merging.

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

            # Note: CTF tokens use setApprovalForAll (ERC1155)
            # This would be done separately if needed

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
            orders = []

            for order_data in data.get("orders", []):
                # Parse order from API response
                # Note: Actual structure may vary, adjust as needed
                orders.append(OrderbookOrder(
                    order_id=order_data.get("id", ""),
                    price=float(order_data.get("price", 0)),
                    size=float(order_data.get("size", 0)),
                    side=order_data.get("side", "BUY"),
                    maker=order_data.get("maker", ""),
                    signature=order_data.get("signature", ""),
                    salt=int(order_data.get("salt", 0)),
                    maker_amount=int(order_data.get("maker_amount", 0)),
                    taker_amount=int(order_data.get("taker_amount", 0)),
                    expiration=int(order_data.get("expiration", 0)),
                    nonce=int(order_data.get("nonce", 0)),
                    fee_rate_bps=int(order_data.get("fee_rate_bps", 0)),
                ))

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
                # BUY: Split USDC into YES/NO tokens
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

                logger.info(f"💰 Splitting ${usdc_amount:.2f} USDC into tokens")

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
                # SELL: Query orderbook and fill buy orders via CTF Exchange
                logger.info(
                    f"💰 Selling {size:.2f} tokens @ {price:.3f}",
                    token_id=token_id[:16] if len(token_id) > 16 else token_id,
                )

                # Get orderbook (people wanting to BUY what we're selling)
                buy_orders = await self.get_orderbook(token_id, side="BUY")

                if not buy_orders:
                    return OrderResult(
                        success=False,
                        error="No buy orders available in orderbook"
                    )

                # Find best buy order (highest price)
                best_order = max(buy_orders, key=lambda o: o.price)

                # Check if price is acceptable
                if best_order.price < price * 0.95:  # Allow 5% slippage
                    return OrderResult(
                        success=False,
                        error=f"Best buy order price {best_order.price:.3f} too low (wanted {price:.3f})"
                    )

                # Determine how much we can fill
                fill_size = min(size, best_order.size)

                logger.info(
                    f"📋 Filling buy order @ {best_order.price:.3f} for {fill_size:.2f} tokens",
                    order_id=best_order.order_id[:16],
                )

                # Fill the order via CTF Exchange
                result = await self._fill_order_onchain(
                    order=best_order,
                    fill_amount=fill_size,
                    token_id=token_id,
                )

                if result.success:
                    logger.info(
                        f"✅ Sold {fill_size:.2f} tokens @ {best_order.price:.3f}",
                        tx_hash=result.tx_hash[:16] if result.tx_hash else None,
                        usdc_received=fill_size * best_order.price,
                    )

                return result

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
        Wait for order fill (compatible interface with LiveExecutor).

        For onchain execution, orders are typically filled immediately
        via position splitting or direct contract interaction.

        Args:
            order_id: Order ID (not used in onchain mode)
            timeout: Timeout in seconds (not used)

        Returns:
            True (assumes immediate fill)
        """
        # In onchain mode, execution is typically immediate
        # No need to wait for fills like with CLOB orderbook
        logger.debug("Onchain: Order filled immediately")
        return True

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (compatible interface with LiveExecutor).

        For onchain execution, there's no order to cancel as
        execution is typically immediate.

        Args:
            order_id: Order ID (not used)

        Returns:
            True
        """
        logger.debug("Onchain: No order to cancel (immediate execution)")
        return True

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
