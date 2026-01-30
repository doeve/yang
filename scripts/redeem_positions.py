#!/usr/bin/env python3
"""
Redeem Resolved Positions

Automatically finds and redeems all winning positions from resolved markets.
This is the Python equivalent of d3v's force-redeem-public.ts with fixes for:
- Checking token balances before attempting redemption
- Avoiding redundant redemption attempts
- Proper error handling

Configuration is loaded from:
1. .env file (copy from .env.example)
2. config.yaml
3. Command-line arguments (optional overrides)
"""

import asyncio
import argparse
import logging
from pathlib import Path

import structlog

from src.config import load_config
from src.execution.onchain_executor import OnchainExecutor

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)


async def main():
    """Main entry point for position redemption."""
    parser = argparse.ArgumentParser(
        description="Redeem all resolved Polymarket positions (reads config from .env and config.yaml)"
    )
    parser.add_argument(
        "--gas-price",
        type=int,
        default=600,
        help="Max gas price in gwei (default: 600)",
    )
    parser.add_argument(
        "--priority-fee",
        type=int,
        default=50,
        help="Max priority fee in gwei (default: 50)",
    )

    args = parser.parse_args()

    # Load configuration from .env and config.yaml
    logger.info("📋 Loading configuration from .env and config.yaml...")
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.info("Make sure you have .env file (copy from .env.example) with required settings")
        return

    # Validate required settings
    if not config.eth_private_key:
        logger.error("ETH_PRIVATE_KEY not set in .env file")
        logger.info("Copy .env.example to .env and set your private key")
        return

    logger.info("🚀 Starting Position Redemption")
    logger.info(f"  Wallet: Loading from config...")
    logger.info(f"  Local RPC: {config.polygon_rpc_url}")
    logger.info(f"  Public RPC: {config.public_rpc_url}")
    logger.info("")

    # Create executor using config
    executor = OnchainExecutor(
        local_rpc_url=config.polygon_rpc_url,
        private_key=config.eth_private_key,
        public_rpc_url=config.public_rpc_url,
        use_public_rpc=config.execution.use_public_rpc_for_redeem,
        socks5_proxy=config.socks5_proxy,
    )

    # Connect
    if not await executor.connect():
        logger.error("Failed to connect to Polygon network")
        logger.info("Check your RPC URLs in .env file")
        return

    try:
        # Run automatic redemption
        results = await executor.redeem_all_resolved_positions()

        # Display detailed results
        logger.info("")
        logger.info("Detailed Results:")
        for i, result in enumerate(results, 1):
            if result.tx_hash:
                logger.info(
                    f"  {i}. ✅ Redeemed ${result.usdc_redeemed:.2f}",
                    condition=result.condition_id[:16],
                    tx_hash=result.tx_hash[:16],
                )
            elif result.skipped_reason:
                logger.debug(
                    f"  {i}. ⏭️  Skipped: {result.skipped_reason}",
                    condition=result.condition_id[:16],
                )
            else:
                logger.warning(
                    f"  {i}. ❌ Failed: {result.error}",
                    condition=result.condition_id[:16],
                )

    finally:
        await executor.disconnect()
        logger.info("Disconnected")


if __name__ == "__main__":
    # Fix for asyncio on some systems
    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
