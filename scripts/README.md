# Yang Trading Scripts

Utility scripts for Polymarket trading operations.

## Position Redemption

### redeem_positions.py

Automatically finds and redeems all winning positions from resolved markets.

**Features:**
- Discovers all markets you've traded in
- Checks which markets are resolved
- **Verifies token balances before attempting redemption** (avoids wasted gas)
- Tracks processed conditions to avoid redundant operations
- Provides detailed summary of redemptions

**Usage:**

```bash
# Basic usage (uses environment variables)
python scripts/redeem_positions.py

# With custom RPC endpoints
python scripts/redeem_positions.py \
    --local-rpc http://localhost:8545 \
    --public-rpc https://polygon-rpc.com

# With custom gas settings
python scripts/redeem_positions.py \
    --gas-price 600 \
    --priority-fee 50

# See all options
python scripts/redeem_positions.py --help
```

**Environment Variables:**

```bash
# Required
export PRIVATE_KEY="0x..."

# Optional (with defaults)
export POLYGON_RPC_URL="http://localhost:8545"  # for reads
export PUBLIC_RPC_URL="https://polygon-rpc.com"  # for transactions
export SOCKS5_PROXY="socks5://127.0.0.1:1080"  # if needed
```

**Example Output:**

```
🚀 Starting Position Redemption

[INFO] Onchain executor connected address=0x1234...
[INFO] Initial USDC balance: $1234.56
[INFO] Found 15 unique traded assets
[INFO] ✅ RESOLVED: Will BTC hit $100k by end of year?...
[INFO] Redeeming position condition_id=0xabcd... yes_balance=100 no_balance=0
[INFO] Redemption tx sent tx_hash=0x5678...
[INFO]    🎉 Redeemed $95.00
[INFO] ✅ RESOLVED: Will ETH flip BTC?...
[DEBUG] Skipped: No tokens held

==================================================
Redemption summary:
  Final USDC: $1329.56
  Total recovered: $95.00
  Successful: 1
  Skipped: 14
  Failed: 0
==================================================
```

## Key Improvements Over d3v Script

The `redeem_positions.py` script is based on d3v's `force-redeem-public.ts` but with critical fixes:

### Problem in Original Script
```typescript
// Original: Always tries to redeem without checking balance
for (const assetId of assetIds) {
  if (denominator > 0n) {
    // Attempts redemption even if balance is 0
    const tx = await ctf.redeemPositions(
      USDC_ADDRESS,
      ethers.ZeroHash,
      conditionId,
      [1, 2],  // Always tries both outcomes
    );
  }
}
```

This causes:
- ❌ Reverted transactions when you don't hold tokens
- ❌ Wasted gas fees
- ❌ Confusing error messages
- ❌ Re-attempts on already redeemed positions

### Solution in This Script
```python
# Check balances BEFORE attempting redemption
yes_balance = await self.get_token_balance(yes_token_id)
no_balance = await self.get_token_balance(no_token_id)

if yes_balance == 0 and no_balance == 0:
    # Skip - no tokens to redeem
    logger.debug("No tokens to redeem")
    return RedemptionResult(success=True, skipped_reason="No tokens held")

# Only redeem if we actually have tokens
result = await self.redeem_position(...)
```

This provides:
- ✅ No wasted gas on empty positions
- ✅ Clear feedback on what was actually redeemed
- ✅ Proper tracking of processed conditions
- ✅ Better error reporting

## Security Notes

- **Private Key**: Never commit your private key to version control
- **Environment Variables**: Use `.env` file (not tracked in git) or export variables
- **Gas Settings**: Adjust based on network conditions to avoid overpaying
- **SOCKS5 Proxy**: Only needed if accessing Polymarket APIs from restricted regions

## Troubleshooting

**"Private key not provided"**
```bash
export PRIVATE_KEY="0x..."
```

**"Failed to connect to Polygon network"**
- Check RPC URLs are accessible
- Try different public RPC: `--public-rpc https://polygon-mainnet.infura.io/v3/YOUR_KEY`
- Verify network connectivity

**"Failed to fetch trades"**
- CLOB API may require authentication for trade history
- Script will still work if you manually provide asset IDs

**Gas too expensive**
- Lower gas price: `--gas-price 300`
- Check current Polygon gas: https://polygonscan.com/gastracker

## Related Documentation

- [Onchain Executor Documentation](../docs/ONCHAIN_EXECUTOR.md)
- [Polymarket CTF Contracts](https://docs.polymarket.com/)
