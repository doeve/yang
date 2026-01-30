# Onchain Executor

The Onchain Executor provides direct blockchain interaction for Polymarket trading, focusing on position redemption without relying solely on the CLOB API.

## Overview

This implementation is ported from `d3v/scripts/force-redeem-public.ts` with significant improvements:

### Key Improvements Over Original Script

1. **Balance Checking Before Redemption**
   - The original script tried to redeem ALL resolved markets
   - Problem: It would attempt redemption even if wallet held zero tokens
   - Fix: Check token balances before attempting redemption
   - Result: Avoids unnecessary gas costs and reverted transactions

2. **Proper Condition Tracking**
   - Tracks processed conditions to avoid redundant operations
   - Prevents re-processing the same market multiple times
   - Reduces unnecessary API calls and blockchain queries

3. **Improved Error Handling**
   - Distinguishes between different error types
   - Skips markets where wallet has no tokens (not an error)
   - Properly logs and reports actual failures

## Architecture

```
OnchainExecutor
├── Connection Management
│   ├── Local RPC (fast reads)
│   └── Public RPC (reliable tx broadcast)
├── Balance Queries
│   ├── USDC balance
│   └── Conditional token balances
├── Market Discovery
│   ├── Fetch traded assets from CLOB API
│   └── Get market info from Gamma API
├── Resolution Checking
│   └── Query payoutDenominator from CTF contract
└── Redemption
    ├── Check balances first (KEY FIX)
    ├── Build & sign transaction
    └── Track processed conditions
```

## Key Difference: Balance Checking

### Original Script (force-redeem-public.ts)
```typescript
// Tries to redeem without checking balance
const tx = await ctf.redeemPositions(
  USDC_ADDRESS,
  ethers.ZeroHash,
  conditionId,
  [1, 2],  // Always tries both YES and NO
);
```

Problem: If wallet doesn't hold tokens for this outcome, transaction reverts with "payout is zero"

### Onchain Executor (Fixed)
```python
# Check balances BEFORE attempting redemption
yes_balance = await self.get_token_balance(yes_token_id)
no_balance = await self.get_token_balance(no_token_id)

if yes_balance == 0 and no_balance == 0:
    # Skip - no tokens to redeem
    self._processed_conditions.add(condition_id)
    return RedemptionResult(
        success=True,
        skipped_reason="No tokens held"
    )

# Only redeem if we have tokens
result = await self.redeem_position(...)
```

## Usage

### Command Line
```bash
# Set environment variables
export PRIVATE_KEY="0x..."
export POLYGON_RPC_URL="http://localhost:8545"
export PUBLIC_RPC_URL="https://polygon-rpc.com"
export SOCKS5_PROXY="socks5://127.0.0.1:1080"  # optional

# Run redemption script
python scripts/redeem_positions.py

# With custom settings
python scripts/redeem_positions.py \
    --gas-price 600 \
    --priority-fee 50
```

### Programmatic Usage
```python
from src.execution import OnchainExecutor

executor = OnchainExecutor(
    local_rpc_url="http://localhost:8545",
    private_key="0x...",
    public_rpc_url="https://polygon-rpc.com",
    use_public_rpc=True,
    socks5_proxy="socks5://127.0.0.1:1080",
)

await executor.connect()

# Automatically redeem all resolved positions
results = await executor.redeem_all_resolved_positions()

# Or redeem specific position
result = await executor.redeem_position(
    condition_id="0x123...",
    yes_token_id="123456",
    no_token_id="789012",
)

await executor.disconnect()
```

### Integration with Trading App
```python
from src.app.execution import PolymarketAdapter

adapter = PolymarketAdapter(config)

# Get balance directly from blockchain
balance = await adapter.get_balance()

# Redeem all resolved positions
results = await adapter.redeem_all_resolved()
```

## Contract Addresses (Polygon Mainnet)

- **ConditionalTokens**: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
- **CTF Exchange**: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- **USDC**: `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`

## Flow Diagram

```
1. Get Traded Assets
   └─> CLOB API: /trades?maker={address}

2. For Each Asset:
   └─> Gamma API: /markets?clob_token_ids={asset_id}
       └─> Extract: condition_id, yes_token_id, no_token_id

3. Check Resolution
   └─> CTF Contract: payoutDenominator(condition_id)
       └─> If > 0: Market is resolved

4. Check Token Balances (KEY FIX)
   └─> CTF Contract: balanceOf(address, yes_token_id)
   └─> CTF Contract: balanceOf(address, no_token_id)
       └─> If both == 0: Skip this market

5. Redeem (Only if we have tokens)
   └─> CTF Contract: redeemPositions(
         collateral=USDC,
         parentCollection=0x00...00,
         conditionId=condition_id,
         indexSets=[1, 2]
       )
```

## Benefits Over CLOB API Approach

1. **Direct Blockchain Access**
   - No dependency on CLOB API for redemption
   - Works even if CLOB API is down
   - True ownership verification via on-chain balances

2. **Automatic Discovery**
   - Scans all traded assets automatically
   - No manual tracking of positions needed
   - Catches all resolved markets

3. **Gas Efficiency**
   - Only redeems when tokens are held
   - Avoids reverted transactions
   - Batch processes multiple markets

4. **Transparency**
   - All operations verifiable on-chain
   - Clear transaction history
   - No hidden fees

## Limitations

- **Order Placement**: Still requires CLOB API or DEX for placing new orders
  - Polymarket's orderbook is off-chain (managed by CLOB API)
  - Direct on-chain trading would require finding counter-parties via mempool or DEX
  - This executor focuses on position management and redemption

- **Trade History**: Uses CLOB API to discover traded assets
  - Could be replaced with on-chain event indexing
  - Future improvement: Index TradeExecuted events from CTF Exchange

## Future Enhancements

1. **Event Indexing**
   ```python
   # Replace CLOB API trade history with on-chain event indexing
   async def get_traded_assets_from_events(self):
       ctf_exchange = self.local_w3.eth.contract(
           address=CTF_EXCHANGE,
           abi=CTF_EXCHANGE_ABI
       )

       # Query TradeExecuted events
       events = ctf_exchange.events.TradeExecuted.get_logs(
           fromBlock="earliest",
           argument_filters={"maker": self.address}
       )

       # Extract token IDs from events
       return {event.args.tokenId for event in events}
   ```

2. **Position Tracking**
   - Track positions in local database
   - Monitor position changes via events
   - Alert on market resolution

3. **Batch Redemption**
   - Combine multiple redemptions in one transaction
   - Requires custom contract or use of multicall

## Comparison with d3v Implementation

| Feature | d3v (TypeScript) | Yang (Python) |
|---------|------------------|---------------|
| Balance Check | ❌ No | ✅ Yes |
| Condition Tracking | ❌ No | ✅ Yes |
| Error Handling | Basic | Comprehensive |
| Skip Logic | Catches reverts | Prevents reverts |
| Gas Efficiency | Lower | Higher |
| Logging | Basic | Structured |
| Integration | Standalone script | Integrated adapter |

## Troubleshooting

### "Transaction reverted" errors
- Likely trying to redeem tokens you don't hold
- Check token balances before redemption (this should be automatic)
- Verify market is actually resolved

### "Not connected" errors
- Ensure RPC URLs are correct
- Check network connectivity
- Verify SOCKS5 proxy is running (if used)

### "Failed to fetch trades" warnings
- CLOB API may require authentication
- Try using authenticated requests
- Fall back to manual asset ID list

### Gas price too high/low
- Adjust `--gas-price` and `--priority-fee` arguments
- Check current Polygon gas prices
- Default: 600 gwei max, 50 gwei priority

## Summary

The Onchain Executor provides a robust, gas-efficient way to manage Polymarket positions directly on Polygon. The key improvement over the original d3v script is **checking token balances before attempting redemption**, which prevents wasted gas on reverted transactions and provides clearer feedback about which markets actually had redeemable tokens.
