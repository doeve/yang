"""
Onchain Executor for Polymarket.

Handles direct blockchain interactions:
- Token balance queries
- Position redemption after market resolution
- Automatic token redemption with proper balance checking

Ported and improved from d3v's force-redeem-public.ts with fixes for:
- Avoiding redemption attempts on already-redeemed positions
- Checking token balances before redemption to prevent reverts
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Set, Dict, Any, List
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


# Minimal ABIs
CTF_ABI = [
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
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
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "getOutcomeSlotCount",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

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
class RedemptionResult:
    """Result of a redemption attempt."""
    success: bool
    condition_id: str
    tx_hash: Optional[str] = None
    usdc_redeemed: float = 0.0
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass
class MarketInfo:
    """Market information from Gamma API."""
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    resolved: bool = False


class OnchainExecutor:
    """
    Onchain executor for Polymarket trading.

    Handles direct blockchain interactions without relying on CLOB API:
    - Queries token balances
    - Redeems winning positions from resolved markets
    - Tracks processed conditions to avoid redundant operations
    """

    def __init__(
        self,
        local_rpc_url: str,
        private_key: str,
        public_rpc_url: str = "https://polygon-rpc.com",
        use_public_rpc: bool = True,
        gamma_api_url: str = "https://gamma-api.polymarket.com",
        clob_api_url: str = "https://clob.polymarket.com",
        socks5_proxy: Optional[str] = None,
    ):
        """
        Initialize onchain executor.

        Args:
            local_rpc_url: Local Polygon RPC URL (for fast reads)
            private_key: Wallet private key
            public_rpc_url: Public Polygon RPC URL (for transaction broadcast)
            use_public_rpc: Use public RPC for transactions (more reliable)
            gamma_api_url: Polymarket Gamma API URL
            clob_api_url: Polymarket CLOB API URL
            socks5_proxy: Optional SOCKS5 proxy URL
        """
        self.local_rpc_url = local_rpc_url
        self.private_key = private_key
        self.public_rpc_url = public_rpc_url
        self.use_public_rpc = use_public_rpc
        self.gamma_api_url = gamma_api_url
        self.clob_api_url = clob_api_url
        self.socks5_proxy = socks5_proxy

        # Web3 providers
        self.local_w3: Optional[Web3] = None
        self.public_w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.address: Optional[str] = None

        # HTTP client for API calls
        self.http_client: Optional[httpx.AsyncClient] = None

        # Track processed conditions to avoid redundant operations
        self._processed_conditions: Set[str] = set()

        # Connected state
        self._connected = False

    async def connect(self) -> bool:
        """Initialize providers, wallet, and HTTP client."""
        try:
            # Setup local provider (for fast reads)
            self.local_w3 = Web3(Web3.HTTPProvider(self.local_rpc_url))
            # Inject POA middleware for Polygon
            self.local_w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            # Setup public provider (for transactions)
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

            logger.info("Onchain executor connected", address=self.address)
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Onchain executor connection failed: {e}")
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
        if not self.local_w3 or not self.address:
            return 0.0

        try:
            usdc = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                abi=ERC20_ABI,
            )
            balance_wei = usdc.functions.balanceOf(self.address).call()
            return balance_wei / 1e6
        except Exception as e:
            logger.error(f"Failed to get USDC balance: {e}")
            return 0.0

    async def get_token_balance(self, token_id: str) -> int:
        """
        Get balance of a specific conditional token.

        Args:
            token_id: The token ID to check

        Returns:
            Token balance as integer
        """
        if not self.local_w3 or not self.address:
            return 0

        # Validate token_id
        if not token_id or token_id == '':
            logger.debug("Empty token_id, skipping balance check")
            return 0

        try:
            ctf = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # Convert token_id to uint256
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
            logger.error(f"Failed to get token balance for {token_id}: {e}")
            return 0

    async def check_resolution(self, condition_id: str) -> bool:
        """
        Check if a condition has been resolved.

        Returns True if payoutDenominator > 0 (resolved).
        """
        if not self.local_w3:
            return False

        try:
            ctf = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # Ensure condition_id is bytes32
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            condition_bytes = bytes.fromhex(condition_id[2:])

            denominator = ctf.functions.payoutDenominator(condition_bytes).call()
            return denominator > 0

        except Exception as e:
            logger.error(f"Check resolution error: {e}")
            return False

    async def get_market_info(self, token_id: str) -> Optional[MarketInfo]:
        """
        Fetch market information from Gamma API.

        Args:
            token_id: CLOB token ID

        Returns:
            MarketInfo or None if not found
        """
        if not self.http_client:
            return None

        try:
            url = f"{self.gamma_api_url}/markets?clob_token_ids={token_id}"
            response = await self.http_client.get(url)

            if response.status_code != 200:
                return None

            markets = response.json()
            if not markets or len(markets) == 0:
                return None

            market = markets[0]
            return MarketInfo(
                condition_id=market.get("conditionId", ""),
                question=market.get("question", ""),
                yes_token_id=market.get("tokens", [{}])[0].get("token_id", "") if market.get("tokens") else "",
                no_token_id=market.get("tokens", [{}])[1].get("token_id", "") if len(market.get("tokens", [])) > 1 else "",
                resolved=market.get("closed", False),
            )

        except Exception as e:
            logger.error(f"Failed to get market info: {e}")
            return None

    async def get_token_ids_from_chain(self, from_block: int = 0, to_block: str = 'latest') -> Set[str]:
        """
        Query blockchain directly for token IDs this wallet has received.

        Uses ERC1155 Transfer events from the CTF contract.

        Args:
            from_block: Starting block (0 = genesis, or use recent block)
            to_block: Ending block ('latest' or block number)

        Returns:
            Set of token IDs (as strings)
        """
        if not self.local_w3 or not self.address:
            return set()

        try:
            logger.info("Querying blockchain for token transfers...")

            ctf = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # If from_block is 0, use a recent block to avoid huge queries
            if from_block == 0:
                current_block = self.local_w3.eth.block_number
                # Look back ~30 days (assuming 2 sec blocks = ~1.3M blocks/month)
                from_block = max(0, current_block - 1_300_000)
                logger.info(f"Scanning from block {from_block} to {to_block}")

            token_ids = set()

            # Query TransferSingle events where 'to' is our address
            # event TransferSingle(address indexed operator, address indexed from, address indexed to, uint256 id, uint256 value)

            # Get logs
            # Topic 0: event signature
            event_signature = Web3.keccak(text='TransferSingle(address,address,address,uint256,uint256)')
            # Topic 3: 'to' address (our address, padded to 32 bytes)
            to_address_topic = '0x' + self.address[2:].lower().zfill(64)

            logs = self.local_w3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': Web3.to_checksum_address(CONTRACTS["CTF"]),
                'topics': [
                    '0x' + event_signature.hex(),
                    None,  # operator (any)
                    None,  # from (any)
                    to_address_topic,  # to (our address)
                ]
            })

            logger.info(f"Found {len(logs)} transfer events")

            # Extract token IDs from logs
            for log in logs:
                # Token ID is the 4th topic (index 3), but it's in the data field for TransferSingle
                # Actually for TransferSingle, id and value are in the data field
                data_hex = log['data'].hex() if isinstance(log['data'], bytes) else log['data']

                if len(data_hex) >= 64:
                    # First 32 bytes (64 hex chars) = id, next 32 bytes = value
                    # Remove '0x' prefix if present
                    if data_hex.startswith('0x'):
                        data_hex = data_hex[2:]

                    token_id_hex = '0x' + data_hex[:64]  # First 32 bytes
                    token_id = str(int(token_id_hex, 16))
                    token_ids.add(token_id)

            logger.info(f"Found {len(token_ids)} unique token IDs")
            return token_ids

        except Exception as e:
            logger.error(f"Failed to query chain for token IDs: {e}")
            import traceback
            traceback.print_exc()
            return set()

    async def redeem_position(
        self,
        condition_id: str,
        yes_token_id: str,
        no_token_id: str,
        gas_price_gwei: int = 600,
        max_priority_fee_gwei: int = 50,
    ) -> RedemptionResult:
        """
        Redeem winning positions for a resolved condition.

        This function checks token balances before attempting redemption
        to avoid unnecessary reverts.

        Args:
            condition_id: The condition ID (bytes32 hex string)
            yes_token_id: YES token ID
            no_token_id: NO token ID
            gas_price_gwei: Max fee per gas in gwei
            max_priority_fee_gwei: Max priority fee in gwei

        Returns:
            RedemptionResult with success status
        """
        if not self._connected or not self.public_w3 or not self.account:
            return RedemptionResult(
                success=False,
                condition_id=condition_id,
                error="Not connected"
            )

        # Check if already processed
        if condition_id in self._processed_conditions:
            logger.debug(f"Condition already processed, skipping", condition_id=condition_id[:16])
            return RedemptionResult(
                success=True,
                condition_id=condition_id,
                skipped_reason="Already processed"
            )

        # Check if resolved
        if not await self.check_resolution(condition_id):
            return RedemptionResult(
                success=False,
                condition_id=condition_id,
                error="Condition not resolved"
            )

        # Check token balances BEFORE attempting redemption
        # This is the fix for the issue in the original script
        yes_balance = await self.get_token_balance(yes_token_id) if yes_token_id else 0
        no_balance = await self.get_token_balance(no_token_id) if no_token_id else 0

        # If we don't have token IDs, we can't check individual balances
        # but we can still try to redeem (it will fail gracefully if no tokens)
        if yes_token_id and no_token_id:
            if yes_balance == 0 and no_balance == 0:
                logger.debug(
                    f"No tokens to redeem for this condition",
                    condition_id=condition_id[:16]
                )
                self._processed_conditions.add(condition_id)
                return RedemptionResult(
                    success=True,
                    condition_id=condition_id,
                    skipped_reason="No tokens held"
                )
        else:
            logger.debug(f"Missing token IDs, will attempt redemption anyway")

        if yes_balance > 0 or no_balance > 0:
            logger.info(
                f"Redeeming position",
                condition_id=condition_id[:16],
                yes_balance=yes_balance / 1e6 if yes_balance else 0,
                no_balance=no_balance / 1e6 if no_balance else 0,
            )
        else:
            logger.info(
                f"Attempting redemption (unknown balance)",
                condition_id=condition_id[:16],
            )

        try:
            ctf = self.public_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["CTF"]),
                abi=CTF_ABI,
            )

            # Ensure condition_id is bytes32
            if not condition_id.startswith("0x"):
                condition_id = "0x" + condition_id
            condition_bytes = bytes.fromhex(condition_id[2:])

            # Parent collection ID (zero for top-level)
            parent_collection_id = bytes(32)

            # Index sets: [1, 2] for binary market (YES=1, NO=2)
            index_sets = [1, 2]

            # Get USDC balance before
            usdc = self.public_w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACTS["USDC"]),
                abi=ERC20_ABI,
            )
            balance_before = usdc.functions.balanceOf(self.address).call()

            # Build transaction
            nonce = self.public_w3.eth.get_transaction_count(self.address, "latest")

            tx = ctf.functions.redeemPositions(
                Web3.to_checksum_address(CONTRACTS["USDC"]),
                parent_collection_id,
                condition_bytes,
                index_sets,
            ).build_transaction({
                "from": self.address,
                "nonce": nonce,
                "maxFeePerGas": Web3.to_wei(gas_price_gwei, "gwei"),
                "maxPriorityFeePerGas": Web3.to_wei(max_priority_fee_gwei, "gwei"),
                "gas": 300000,
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(f"Redemption tx sent", tx_hash=tx_hash_hex)

            # Wait for confirmation
            receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                # Mark as processed
                self._processed_conditions.add(condition_id)

                # Calculate USDC redeemed
                balance_after = usdc.functions.balanceOf(self.address).call()
                usdc_redeemed = (balance_after - balance_before) / 1e6

                logger.info(
                    f"Redemption successful",
                    tx_hash=tx_hash_hex,
                    block=receipt["blockNumber"],
                    usdc_redeemed=usdc_redeemed,
                )

                return RedemptionResult(
                    success=True,
                    condition_id=condition_id,
                    tx_hash=tx_hash_hex,
                    usdc_redeemed=usdc_redeemed,
                )
            else:
                return RedemptionResult(
                    success=False,
                    condition_id=condition_id,
                    tx_hash=tx_hash_hex,
                    error="Transaction reverted",
                )

        except Exception as e:
            error_msg = str(e)

            # Handle common revert reasons
            if "payout is zero" in error_msg.lower():
                # No winning tokens - this shouldn't happen since we check balances
                self._processed_conditions.add(condition_id)
                logger.info(f"No winning tokens to redeem", condition_id=condition_id[:16])
                return RedemptionResult(
                    success=True,
                    condition_id=condition_id,
                    skipped_reason="No winning tokens"
                )

            logger.error(f"Redemption error: {error_msg}")
            return RedemptionResult(
                success=False,
                condition_id=condition_id,
                error=error_msg
            )

    async def redeem_all_resolved_positions(self, from_block: int = 0) -> List[RedemptionResult]:
        """
        Automatically find and redeem all resolved positions by querying the blockchain.

        This queries the CTF contract directly for Transfer events to find
        which tokens this wallet has held, then redeems resolved positions.

        Args:
            from_block: Starting block for event scanning (0 = auto, scans last ~30 days)

        Returns:
            List of RedemptionResult objects
        """
        if not self._connected:
            logger.error("Not connected")
            return []

        logger.info("Starting automatic redemption of resolved positions")

        # Get initial balance
        initial_balance = await self.get_usdc_balance()
        logger.info(f"Initial USDC balance: ${initial_balance:.2f}")

        # Get all token IDs from blockchain
        token_ids = await self.get_token_ids_from_chain(from_block=from_block)

        if not token_ids:
            logger.info("No token transfers found")
            return []

        logger.info(f"Found {len(token_ids)} unique token IDs from blockchain")

        # Track unique conditions to avoid processing the same market twice
        seen_conditions: Set[str] = set()
        results: List[RedemptionResult] = []

        # For each token ID, we need to derive the condition_id
        # Token IDs in CTF contract encode position info, but we need to group by condition
        # We'll check each token's balance and try to find pairs

        # Group tokens by checking balances
        token_balances = {}
        for token_id in token_ids:
            balance = await self.get_token_balance(token_id)
            if balance > 0:
                token_balances[token_id] = balance

        logger.info(f"Found {len(token_balances)} tokens with non-zero balance")

        # Try to get market info for each token to find condition IDs
        for token_id in token_balances.keys():
            try:
                # Query Gamma API to map token_id -> condition_id
                market_info = await self.get_market_info(token_id)
                if not market_info:
                    # If API fails, skip this token
                    logger.debug(f"Could not get market info for token {token_id[:10]}...")
                    continue

                condition_id = market_info.condition_id

                # Skip if already seen
                if condition_id in seen_conditions:
                    continue
                seen_conditions.add(condition_id)

                # Check if resolved
                is_resolved = await self.check_resolution(condition_id)
                if not is_resolved:
                    logger.debug(
                        f"Market not resolved: {market_info.question[:50]}...",
                        condition_id=condition_id[:16]
                    )
                    continue

                logger.info(
                    f"✅ RESOLVED: {market_info.question[:50]}...",
                    condition_id=condition_id[:16]
                )

                # Get token IDs - prefer from market_info, fallback to the token we found
                yes_token = market_info.yes_token_id if market_info.yes_token_id else token_id
                no_token = market_info.no_token_id if market_info.no_token_id else ""

                # Attempt redemption
                result = await self.redeem_position(
                    condition_id,
                    yes_token,
                    no_token,
                )
                results.append(result)

                # Log result
                if result.success and result.tx_hash:
                    logger.info(f"   🎉 Redeemed ${result.usdc_redeemed:.2f}")
                elif result.skipped_reason:
                    logger.debug(f"   Skipped: {result.skipped_reason}")

                # Small delay between redemptions
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error processing asset {asset_id[:10]}: {e}")

        # Get final balance
        final_balance = await self.get_usdc_balance()
        total_redeemed = final_balance - initial_balance

        # Summary
        successful = sum(1 for r in results if r.success and r.tx_hash)
        skipped = sum(1 for r in results if r.skipped_reason)
        failed = sum(1 for r in results if not r.success and not r.skipped_reason)

        logger.info("=" * 50)
        logger.info(f"Redemption summary:")
        logger.info(f"  Final USDC: ${final_balance:.2f}")
        logger.info(f"  Total recovered: ${total_redeemed:.2f}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Failed: {failed}")
        logger.info("=" * 50)

        return results

    def mark_processed(self, condition_id: str) -> None:
        """Manually mark a condition as already processed."""
        self._processed_conditions.add(condition_id)

    def get_processed_conditions(self) -> Set[str]:
        """Get set of already-processed condition IDs."""
        return self._processed_conditions.copy()
