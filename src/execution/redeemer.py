"""
Redeemer for Polymarket conditional tokens.

Handles redemption of winning shares after market resolution.
Ported from d3v's force-redeem-public.ts.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Set, Dict, Any

import structlog
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

logger = structlog.get_logger(__name__)


# Contract addresses (Polygon Mainnet)
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

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
    tx_hash: Optional[str] = None
    usdc_redeemed: float = 0.0
    error: Optional[str] = None
    skipped: bool = False  # True if already redeemed


class Redeemer:
    """
    Handles redemption of winning conditional token shares.
    
    Uses public RPC for reliable transaction broadcast.
    Tracks already-redeemed conditions to avoid redundant calls.
    """
    
    def __init__(
        self,
        local_rpc_url: str,
        private_key: str,
        public_rpc_url: str = "https://polygon-rpc.com",
        use_public_rpc: bool = True,
    ):
        self.local_rpc_url = local_rpc_url
        self.private_key = private_key
        self.public_rpc_url = public_rpc_url
        self.use_public_rpc = use_public_rpc
        
        # Web3 providers
        self.local_w3: Optional[Web3] = None
        self.public_w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.address: Optional[str] = None
        
        # Track already redeemed conditions to avoid redundant calls
        self._redeemed_conditions: Set[str] = set()
        
        # Connected state
        self._connected = False
    
    async def connect(self) -> bool:
        """Initialize providers and wallet."""
        try:
            # Setup local provider (for reads)
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
            
            logger.info(f"Redeemer connected", address=self.address)
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Redeemer connection failed: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def check_resolution(self, condition_id: str) -> bool:
        """
        Check if a condition has been resolved.
        
        Returns True if payoutDenominator > 0 (resolved).
        """
        if not self.local_w3:
            return False
        
        try:
            ctf = self.local_w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
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
    
    async def redeem_positions(
        self,
        condition_id: str,
        gas_price_gwei: int = 100,
        max_priority_fee_gwei: int = 30,
    ) -> RedemptionResult:
        """
        Redeem winning positions for a resolved condition.
        
        Args:
            condition_id: The condition ID (bytes32 hex string)
            gas_price_gwei: Max fee per gas in gwei
            max_priority_fee_gwei: Max priority fee in gwei
        
        Returns:
            RedemptionResult with success status
        """
        if not self._connected or not self.public_w3 or not self.account:
            return RedemptionResult(success=False, error="Not connected")
        
        # Check if already redeemed
        if condition_id in self._redeemed_conditions:
            logger.info(f"Condition already redeemed, skipping", condition_id=condition_id[:16])
            return RedemptionResult(success=True, skipped=True)
        
        # Check if resolved
        if not await self.check_resolution(condition_id):
            return RedemptionResult(success=False, error="Condition not resolved")
        
        try:
            ctf = self.public_w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
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
                address=Web3.to_checksum_address(USDC_ADDRESS),
                abi=ERC20_ABI,
            )
            balance_before = usdc.functions.balanceOf(self.address).call()
            
            # Build transaction
            nonce = self.public_w3.eth.get_transaction_count(self.address, "latest")
            
            tx = ctf.functions.redeemPositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                parent_collection_id,
                condition_bytes,
                index_sets,
            ).build_transaction({
                "from": self.address,
                "nonce": nonce,
                "maxFeePerGas": Web3.to_wei(gas_price_gwei, "gwei"),
                "maxPriorityFeePerGas": Web3.to_wei(max_priority_fee_gwei, "gwei"),
                "gas": 300000,  # Estimate
            })
            
            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(f"Redemption tx sent", tx_hash=tx_hash_hex)
            
            # Wait for confirmation
            receipt = self.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt["status"] == 1:
                # Mark as redeemed
                self._redeemed_conditions.add(condition_id)
                
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
                    tx_hash=tx_hash_hex,
                    usdc_redeemed=usdc_redeemed,
                )
            else:
                return RedemptionResult(
                    success=False,
                    tx_hash=tx_hash_hex,
                    error="Transaction reverted",
                )
                
        except Exception as e:
            error_msg = str(e)
            
            # Handle common revert reasons
            if "payout is zero" in error_msg.lower() or "execution reverted" in error_msg.lower():
                # No winning tokens for this outcome - mark as handled
                self._redeemed_conditions.add(condition_id)
                logger.info(f"No winning tokens to redeem", condition_id=condition_id[:16])
                return RedemptionResult(success=True, skipped=True)
            
            logger.error(f"Redemption error: {error_msg}")
            return RedemptionResult(success=False, error=error_msg)
    
    def mark_redeemed(self, condition_id: str) -> None:
        """Manually mark a condition as already redeemed."""
        self._redeemed_conditions.add(condition_id)
    
    def get_redeemed_conditions(self) -> Set[str]:
        """Get set of already-redeemed condition IDs."""
        return self._redeemed_conditions.copy()
