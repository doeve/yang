"""
Comparison: Old vs New Redemption Approach

This example demonstrates the key difference between the d3v force-redeem-public.ts
script and the improved Yang onchain executor.
"""

import asyncio
from web3 import Web3
from eth_account import Account


# Simulated contract for demonstration
class SimulatedCTF:
    """Simulates the ConditionalTokens contract behavior."""

    def __init__(self):
        self.balances = {
            # condition_id -> {yes_balance, no_balance}
            "0xabcd...": {"yes": 100, "no": 0},  # Has YES tokens
            "0x1234...": {"yes": 0, "no": 0},    # No tokens (already redeemed)
            "0x5678...": {"yes": 0, "no": 50},   # Has NO tokens
        }

    def balanceOf(self, address: str, token_id: str) -> int:
        """Simulate token balance check."""
        # In reality, this queries the blockchain
        return 0  # Simplified for demo

    def redeemPositions(self, collateral, parent_id, condition_id, index_sets):
        """Simulate redemption transaction."""
        balances = self.balances.get(condition_id, {"yes": 0, "no": 0})

        if balances["yes"] == 0 and balances["no"] == 0:
            # This is what happens in the blockchain
            raise Exception("execution reverted: payout is zero")

        return {"hash": "0xabcd...", "status": 1}


# ============================================================================
# OLD APPROACH (d3v force-redeem-public.ts)
# ============================================================================

async def old_approach_redeem(ctf: SimulatedCTF, condition_ids: list):
    """
    Original d3v approach - tries to redeem without checking balances.

    Problems:
    1. Attempts redemption on all resolved markets
    2. Doesn't check if wallet holds tokens
    3. Results in reverted transactions and wasted gas
    """
    print("\n" + "="*60)
    print("OLD APPROACH (d3v force-redeem-public.ts)")
    print("="*60)

    successful = 0
    failed = 0

    for condition_id in condition_ids:
        try:
            print(f"\nProcessing {condition_id}")

            # Check if resolved (this part is the same)
            is_resolved = True  # Assume all are resolved

            if is_resolved:
                print(f"  ✓ Market is resolved")

                # OLD: Immediately try to redeem WITHOUT checking balance
                print(f"  → Attempting redemption...")
                tx = ctf.redeemPositions(
                    collateral="USDC",
                    parent_id="0x00",
                    condition_id=condition_id,
                    index_sets=[1, 2],  # Always tries both YES and NO
                )
                print(f"  ✓ Redeemed! TX: {tx['hash']}")
                successful += 1

        except Exception as e:
            # This catches the revert, but gas was already spent!
            if "payout is zero" in str(e):
                print(f"  ✗ Skipped (no winning tokens)")
            else:
                print(f"  ✗ Error: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULT: {successful} successful, {failed} failed/reverted")
    print(f"Gas wasted on {failed} reverted transactions! 💸")
    print(f"{'='*60}\n")


# ============================================================================
# NEW APPROACH (Yang onchain executor)
# ============================================================================

async def new_approach_redeem(ctf: SimulatedCTF, condition_ids: list):
    """
    Improved Yang approach - checks balances BEFORE attempting redemption.

    Improvements:
    1. Queries token balances first
    2. Only attempts redemption if tokens are held
    3. Avoids wasted gas on empty positions
    4. Provides clear feedback on skipped vs redeemed
    """
    print("\n" + "="*60)
    print("NEW APPROACH (Yang onchain executor)")
    print("="*60)

    successful = 0
    skipped = 0

    for condition_id in condition_ids:
        try:
            print(f"\nProcessing {condition_id}")

            # Check if resolved (same as before)
            is_resolved = True  # Assume all are resolved

            if is_resolved:
                print(f"  ✓ Market is resolved")

                # NEW: Check balances BEFORE attempting redemption
                yes_balance = ctf.balances.get(condition_id, {}).get("yes", 0)
                no_balance = ctf.balances.get(condition_id, {}).get("no", 0)
                print(f"  → Checking balances: YES={yes_balance}, NO={no_balance}")

                if yes_balance == 0 and no_balance == 0:
                    # Skip without wasting gas!
                    print(f"  ⏭  Skipped (no tokens held)")
                    skipped += 1
                    continue

                # Only redeem if we actually have tokens
                print(f"  → Attempting redemption...")
                tx = ctf.redeemPositions(
                    collateral="USDC",
                    parent_id="0x00",
                    condition_id=condition_id,
                    index_sets=[1, 2],
                )
                print(f"  ✓ Redeemed! TX: {tx['hash']}")
                successful += 1

        except Exception as e:
            # Should rarely happen now since we check balances
            print(f"  ✗ Unexpected error: {e}")

    print(f"\n{'='*60}")
    print(f"RESULT: {successful} successful, {skipped} skipped")
    print(f"No wasted gas! All transactions were necessary. ✅")
    print(f"{'='*60}\n")


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def main():
    """Run comparison demonstration."""
    print("\n" + "="*60)
    print("POLYMARKET REDEMPTION: OLD vs NEW APPROACH")
    print("="*60)

    # Simulated conditions
    ctf = SimulatedCTF()
    condition_ids = [
        "0xabcd...",  # Has YES tokens - should redeem
        "0x1234...",  # No tokens - should skip (already redeemed)
        "0x5678...",  # Has NO tokens - should redeem
    ]

    print("\nScenario:")
    print("  - 3 resolved markets")
    print("  - Market 1: Wallet holds 100 YES tokens")
    print("  - Market 2: Wallet holds 0 tokens (already redeemed)")
    print("  - Market 3: Wallet holds 50 NO tokens")

    # Run old approach
    await old_approach_redeem(ctf, condition_ids)

    # Run new approach
    await new_approach_redeem(ctf, condition_ids)

    # Summary
    print("\n" + "="*60)
    print("KEY DIFFERENCE")
    print("="*60)
    print("""
OLD APPROACH:
  1. Check if market is resolved ✓
  2. Try to redeem
  3. Transaction reverts if no tokens ✗
  4. Gas wasted 💸

NEW APPROACH:
  1. Check if market is resolved ✓
  2. Check token balances ✓
  3. Skip if no tokens (no transaction) ✓
  4. Only redeem if tokens held ✓
  5. No wasted gas ✅

CODE COMPARISON:

# Old (d3v):
if (denominator > 0n) {
    const tx = await ctf.redeemPositions(...)  // May revert!
}

# New (Yang):
if await self.check_resolution(condition_id):
    yes_bal = await self.get_token_balance(yes_token_id)
    no_bal = await self.get_token_balance(no_token_id)

    if yes_bal == 0 and no_bal == 0:
        return  # Skip - no gas wasted!

    tx = await self.redeem_position(...)  // Only if we have tokens
""")


if __name__ == "__main__":
    asyncio.run(main())
