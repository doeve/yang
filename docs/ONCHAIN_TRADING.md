# Onchain Trading - Fee-Free Polymarket Execution

## Overview

The onchain trading system allows you to trade on Polymarket **without paying CLOB API fees** by executing directly on the blockchain.

## How It Works

### Traditional CLOB API Approach (Fees)

```
User → CLOB API → CTF Exchange Contract → Settlement
         ↑
      Fees charged here!
```

### Onchain Approach (No Fees)

```
User → CTF Exchange Contract → Settlement
  ↑
  No CLOB fees!
```

## Trading Strategies

### 1. Position Splitting/Merging (Recommended for Long Positions)

**Create Long Position:**
```
1. Approve USDC for CTF contract
2. Call splitPosition() to get YES + NO tokens
3. Sell unwanted side (optional)
```

**Close Position:**
```
1. Hold both YES and NO tokens
2. Call mergePositions() to get USDC back
```

**Benefits:**
- No trading fees
- Guaranteed execution
- Full control

**Cost:**
- Only gas fees (~$0.01-0.10 on Polygon)

### 2. Direct Order Filling (For Market Orders)

**Strategy:**
```
1. Query orderbook from CLOB API (read-only, free)
2. Find best available orders
3. Fill orders directly via CTF Exchange contract
4. Pay only gas, no CLOB fees
```

**Benefits:**
- Access to existing liquidity
- No CLOB trading fees
- Better prices (no fee markup)

**Requirements:**
- Understanding of order structure
- Proper signature validation
- Gas management

## Implementation

### Onchain Order Executor

Located in: `src/execution/onchain_order_executor.py`

**Key Methods:**
- `split_position()` - Create YES/NO tokens from USDC
- `merge_position()` - Combine YES/NO back to USDC
- `ensure_approvals()` - Setup contract approvals
- `get_orderbook()` - Read CLOB orderbook (free)

### Integration in Paper Trading

The paper trading script (`src/paper_trade_unified_new.py`) now uses:

1. **OnchainOrderExecutor** for trading
   - Replaces LiveExecutor (which used CLOB API)
   - Executes trades directly onchain
   - No CLOB fees

2. **OnchainExecutor** for redemption
   - Automatically redeems after each market close
   - Checks balances before redemption
   - Prevents wasted gas

## Usage Example

```python
from src.execution import OnchainOrderExecutor

# Initialize
executor = OnchainOrderExecutor(
    local_rpc_url="http://localhost:8545",
    private_key="0x...",
    public_rpc_url="https://polygon-rpc.com",
)

await executor.connect()

# Ensure approvals (one-time setup)
await executor.ensure_approvals()

# Split position (buy both YES and NO)
result = await executor.split_position(
    condition_id="0xabcd...",
    amount_usdc=10.0,  # $10 worth
)

# Later: Merge back to USDC
result = await executor.merge_position(
    condition_id="0xabcd...",
    amount=10.0,
)
```

## Automatic Redemption

After each market closes and settles, the system automatically:

1. Checks if market is resolved
2. Checks token balances (YES and NO)
3. Only redeems if you hold tokens
4. Updates wallet balance

**No manual intervention required!**

## Cost Comparison

### CLOB API Trading (Old)
```
Trade Size: $100
CLOB Fee: 2% = $2.00
Gas Fee: ~$0.05
Total Cost: $2.05
```

### Onchain Trading (New)
```
Trade Size: $100
CLOB Fee: $0.00
Gas Fee: ~$0.05
Total Cost: $0.05

Savings: $2.00 (97.5% reduction!)
```

## Limitations

1. **Liquidity Access**
   - Position splitting gives you tokens but you need to find buyers
   - For immediate liquidity, you still need the orderbook
   - Best for: markets where you can wait for fills

2. **Market Orders**
   - Direct order filling requires finding existing orders
   - Not suitable for limit orders (those need orderbook matching)
   - Best for: taking existing liquidity

3. **Complexity**
   - More complex than simple API calls
   - Requires gas management
   - Need to understand contract interactions

## When to Use Each Approach

### Use Onchain (Fee-Free)
- ✅ You're okay waiting for fills
- ✅ Trading larger amounts (fees add up)
- ✅ You want full control
- ✅ You understand smart contracts

### Use CLOB API (Traditional)
- ✅ Need immediate fills
- ✅ Want limit order functionality
- ✅ Trading small amounts
- ✅ Prefer simplicity

## Configuration

In `config.yaml`:

```yaml
trading_mode: live

execution:
  # Use public RPC for reliable tx broadcast
  use_public_rpc_for_redeem: true
  public_rpc_url: "https://polygon-rpc.com"

  # Gas settings
  order_timeout_seconds: 30
  poll_interval_seconds: 5
```

Environment variables:

```bash
export PRIVATE_KEY="0x..."
export POLYGON_RPC_URL="http://localhost:8545"
export PUBLIC_RPC_URL="https://polygon-rpc.com"
```

## Paper Trading with Onchain Execution

Run the paper trader with live onchain execution:

```bash
python src/paper_trade_unified_new.py \
    --model logs/market_predictor_v1 \
    --live
```

Features:
- Real blockchain transactions
- No CLOB fees
- Automatic redemption after each candle
- Position splitting/merging
- Gas-efficient execution

## Security Notes

1. **Private Key Protection**
   - Never commit private keys
   - Use environment variables
   - Consider hardware wallets for production

2. **Gas Price Management**
   - Monitor Polygon gas prices
   - Set reasonable limits
   - Account for gas in P&L calculations

3. **Approval Management**
   - Approvals are permanent until revoked
   - Only approve trusted contracts
   - Consider using limited approvals

## Advanced: Custom Trading Strategies

### Example: Market Making Without Fees

```python
# 1. Split position to get inventory
await executor.split_position(condition_id, amount=100)

# 2. You now have 100 YES and 100 NO tokens
# 3. Sell YES at premium via DEX/P2P
# 4. Sell NO at premium via DEX/P2P
# 5. Profit = premiums - gas costs

# No CLOB fees means you can offer better prices
# and still make profit!
```

### Example: Arbitrage

```python
# 1. Find mispricing on CLOB orderbook
# 2. Fill order directly (save fees)
# 3. Split/merge to close position
# 4. Profit from spread - gas

# Lower costs = more arbitrage opportunities
```

## Troubleshooting

**"Transaction reverted"**
- Check USDC balance
- Ensure approvals are set
- Verify condition_id is correct

**"Gas too high"**
- Adjust gas price settings
- Use Polygon gas tracker
- Consider transaction timing

**"No tokens to redeem"**
- This is normal! It means you didn't hold those tokens
- Redemption is properly skipped
- No action needed

## Summary

Onchain trading provides a **fee-free alternative** to the CLOB API by interacting directly with Polymarket's smart contracts. While it requires more technical knowledge, the **97.5% cost savings** make it worthwhile for active traders.

The automatic redemption feature ensures you get your USDC back immediately after market resolution, without manual intervention.

**Best for:** Cost-conscious traders, high-volume trading, technical users
**Saves:** ~$2 per $100 traded (vs CLOB API fees)
