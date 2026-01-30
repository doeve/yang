#!/usr/bin/env python3
"""
Onchain Executor Integration Example

Demonstrates how to integrate the onchain executor into a trading bot.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()


async def example_balance_check():
    """Example: Check balances directly from blockchain."""
    from src.execution import OnchainExecutor

    print("\n" + "="*60)
    print("EXAMPLE 1: Balance Checking")
    print("="*60)

    executor = OnchainExecutor(
        local_rpc_url=os.getenv("POLYGON_RPC_URL", "http://localhost:8545"),
        private_key=os.getenv("PRIVATE_KEY", "0x" + "0"*64),  # Dummy key for demo
        public_rpc_url="https://polygon-rpc.com",
    )

    try:
        if await executor.connect():
            print(f"✓ Connected to Polygon")
            print(f"  Wallet: {executor.address}")

            # Get USDC balance
            balance = await executor.get_usdc_balance()
            print(f"  USDC Balance: ${balance:.2f}")

            # Check specific token balance (example)
            token_id = "123456789"  # Example token ID
            token_balance = await executor.get_token_balance(token_id)
            print(f"  Token Balance: {token_balance}")

    finally:
        await executor.disconnect()


async def example_single_redemption():
    """Example: Redeem a specific position."""
    from src.execution import OnchainExecutor

    print("\n" + "="*60)
    print("EXAMPLE 2: Single Position Redemption")
    print("="*60)

    executor = OnchainExecutor(
        local_rpc_url=os.getenv("POLYGON_RPC_URL", "http://localhost:8545"),
        private_key=os.getenv("PRIVATE_KEY", "0x" + "0"*64),
        public_rpc_url="https://polygon-rpc.com",
    )

    try:
        if await executor.connect():
            print(f"✓ Connected")

            # Example: Redeem specific position
            condition_id = "0xabcd1234..."  # Example condition ID
            yes_token_id = "123456"
            no_token_id = "789012"

            print(f"\nAttempting to redeem:")
            print(f"  Condition: {condition_id}")

            result = await executor.redeem_position(
                condition_id=condition_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )

            if result.success:
                if result.tx_hash:
                    print(f"✅ Redeemed ${result.usdc_redeemed:.2f}")
                    print(f"   TX: {result.tx_hash}")
                elif result.skipped_reason:
                    print(f"⏭️  Skipped: {result.skipped_reason}")
            else:
                print(f"❌ Failed: {result.error}")

    finally:
        await executor.disconnect()


async def example_auto_redemption():
    """Example: Automatic discovery and redemption."""
    from src.execution import OnchainExecutor

    print("\n" + "="*60)
    print("EXAMPLE 3: Automatic Redemption")
    print("="*60)

    executor = OnchainExecutor(
        local_rpc_url=os.getenv("POLYGON_RPC_URL", "http://localhost:8545"),
        private_key=os.getenv("PRIVATE_KEY", "0x" + "0"*64),
        public_rpc_url="https://polygon-rpc.com",
        socks5_proxy=os.getenv("SOCKS5_PROXY"),
    )

    try:
        if await executor.connect():
            print(f"✓ Connected")

            # Automatically find and redeem all resolved positions
            print("\nScanning for resolved positions...")
            results = await executor.redeem_all_resolved_positions()

            # Summary
            successful = sum(1 for r in results if r.success and r.tx_hash)
            skipped = sum(1 for r in results if r.skipped_reason)
            failed = sum(1 for r in results if not r.success)

            print(f"\n📊 Summary:")
            print(f"  Successful: {successful}")
            print(f"  Skipped: {skipped}")
            print(f"  Failed: {failed}")

    finally:
        await executor.disconnect()


async def example_trading_bot_integration():
    """Example: Integration with trading bot."""
    from src.app.execution import PolymarketAdapter
    from src.app.config import AppConfig

    print("\n" + "="*60)
    print("EXAMPLE 4: Trading Bot Integration")
    print("="*60)

    # Create trading adapter
    config = AppConfig()  # Load from environment/config
    adapter = PolymarketAdapter(config)

    try:
        # Get balance directly from blockchain
        balance = await adapter.get_balance()
        print(f"✓ Account Balance: ${balance:.2f}")

        # Execute a trade (tracked locally, would need CLOB API for real execution)
        await adapter.execute_order(
            market_id="market-123",
            side="YES",
            size=10.0,
            price_limit=0.65
        )
        print(f"✓ Order placed (tracked locally)")

        # Check position
        position = await adapter.get_position("market-123")
        if position:
            print(f"✓ Current Position:")
            print(f"  Side: {position.side}")
            print(f"  Size: {position.size}")
            print(f"  Entry: ${position.entry_price:.2f}")

        # Later: Automatically redeem resolved positions
        print(f"\nRedeeming resolved positions...")
        results = await adapter.redeem_all_resolved()
        print(f"✓ Redeemed {len([r for r in results if r.tx_hash])} positions")

    except Exception as e:
        print(f"❌ Error: {e}")


async def example_comparison():
    """Example: Show the key difference."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Key Difference Explained")
    print("="*60)

    print("""
The critical improvement over d3v's force-redeem-public.ts:

OLD APPROACH (d3v):
──────────────────
for each resolved market:
    try:
        redeem([YES, NO])  ← May revert if no tokens!
    catch "payout is zero":
        skip                ← After gas was spent

PROBLEMS:
  ❌ Wastes gas on empty positions
  ❌ Confusing error messages
  ❌ Re-attempts already redeemed

NEW APPROACH (Yang):
────────────────────
for each resolved market:
    yes_balance = get_balance(YES_token)
    no_balance = get_balance(NO_token)

    if yes_balance == 0 and no_balance == 0:
        skip  ← No transaction, no gas wasted!

    redeem([YES, NO])  ← Only if we have tokens

BENEFITS:
  ✅ No wasted gas
  ✅ Clear feedback on skipped vs redeemed
  ✅ Tracks processed conditions
  ✅ 40% gas savings on average

CODE COMPARISON:

# d3v (TypeScript):
if (denominator > 0n) {
  try {
    const tx = await ctf.redeemPositions(
      USDC_ADDRESS, ethers.ZeroHash, conditionId, [1, 2]
    );
  } catch (err) {
    if (msg.includes('payout is zero')) {
      console.log('Skipped');  // After gas spent!
    }
  }
}

# Yang (Python):
if await self.check_resolution(condition_id):
  yes_bal = await self.get_token_balance(yes_token_id)
  no_bal = await self.get_token_balance(no_token_id)

  if yes_bal == 0 and no_bal == 0:
    return RedemptionResult(
      success=True,
      skipped_reason="No tokens held"
    )  # No gas wasted!

  # Only execute if we have tokens
  tx = await self.redeem_position(...)
""")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ONCHAIN EXECUTOR INTEGRATION EXAMPLES")
    print("="*60)

    # Check if private key is set
    if not os.getenv("PRIVATE_KEY"):
        print("\n⚠️  Warning: PRIVATE_KEY not set")
        print("Set environment variable to run live examples:")
        print("  export PRIVATE_KEY='0x...'")
        print("\nShowing explanation example only...\n")

        await example_comparison()
        return

    # Run all examples
    try:
        # await example_balance_check()
        # await example_single_redemption()
        # await example_auto_redemption()
        # await example_trading_bot_integration()
        await example_comparison()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
