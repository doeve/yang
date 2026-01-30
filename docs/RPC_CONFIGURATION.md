# RPC Configuration Guide

## Issue: Public RPC Rate Limiting

Public Polygon RPCs often have rate limits. You'll see errors like:
```
Too many requests, reason: call rate limit exhausted, retry in 10s
```

## Solution: Use Local Polygon Node

The best solution is to use your **local Polygon node** for all operations.

### Setup

**In your .env file:**
```bash
# Use local node for BOTH reads and writes
POLYGON_RPC_URL=http://localhost:8545
PUBLIC_RPC_URL=http://localhost:8545  # Use local instead of public

# Or comment out PUBLIC_RPC_URL to use POLYGON_RPC_URL for both
# PUBLIC_RPC_URL=
```

**In config.yaml:**
```yaml
execution:
  use_public_rpc_for_redeem: false  # Use local RPC
  # public_rpc_url will be ignored
```

## Alternative: Fallback Public RPCs

If you don't have a local node, try these alternatives:

### Free Public RPCs

```bash
# Option 1: Official Polygon RPC
PUBLIC_RPC_URL=https://rpc-mainnet.matic.network

# Option 2: Matic Vigil
PUBLIC_RPC_URL=https://rpc-mainnet.maticvigil.com

# Option 3: Polygon RPC (sometimes rate limited)
PUBLIC_RPC_URL=https://polygon-rpc.com

# Option 4: Ankr
PUBLIC_RPC_URL=https://rpc.ankr.com/polygon
```

### Paid/Premium RPCs (No Rate Limits)

```bash
# Infura (requires API key)
PUBLIC_RPC_URL=https://polygon-mainnet.infura.io/v3/YOUR_API_KEY

# Alchemy (requires API key)
PUBLIC_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY

# QuickNode (requires account)
PUBLIC_RPC_URL=https://your-endpoint.matic.quiknode.pro/YOUR_KEY
```

## Current Optimization

The system is now optimized to minimize public RPC calls:

1. **Reads from Local RPC** (fast, no limits)
   - Balance checks
   - Approval checks
   - Token balances
   - Market data queries

2. **Writes to Public RPC** (only when needed)
   - Transaction broadcasting
   - Only if `use_public_rpc_for_redeem=true`

## Recommended Configuration

### If you have a local Polygon node:

```bash
# .env
POLYGON_RPC_URL=http://localhost:8545
PUBLIC_RPC_URL=http://localhost:8545  # Same as local
```

```yaml
# config.yaml
execution:
  use_public_rpc_for_redeem: false  # Use local for everything
```

**Benefits:**
- ✅ No rate limits
- ✅ Faster responses
- ✅ More reliable
- ✅ Free

### If you DON'T have a local node:

```bash
# .env
POLYGON_RPC_URL=https://rpc-mainnet.matic.network
PUBLIC_RPC_URL=https://rpc-mainnet.matic.network  # Same
```

```yaml
# config.yaml
execution:
  use_public_rpc_for_redeem: true
  public_rpc_url: "https://rpc-mainnet.matic.network"
```

**Alternative:** Get a premium RPC:
- Infura: https://infura.io/ (free tier available)
- Alchemy: https://www.alchemy.com/ (free tier available)

## Testing Your RPC

```bash
# Test connection
curl -X POST $PUBLIC_RPC_URL \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Should return:
# {"jsonrpc":"2.0","id":1,"result":"0x..."}
```

## Troubleshooting

### "Rate limit exhausted"
- Switch to local node
- Try alternative public RPC
- Get premium RPC with higher limits

### "Connection refused"
- Check local node is running: `systemctl status polygon`
- Verify RPC URL is correct
- Check firewall settings

### "Timeout"
- Local node may be syncing
- Try public RPC temporarily
- Increase timeout in code if needed

## Rate Limit Comparison

| RPC Provider | Free Tier Limit | Cost |
|--------------|-----------------|------|
| Local Node | Unlimited | Free (requires running node) |
| polygon-rpc.com | ~25 req/sec | Free |
| rpc-mainnet.matic.network | ~25 req/sec | Free |
| Infura Free | 100,000 req/day | Free |
| Alchemy Free | 300M compute units/month | Free |
| Premium | Unlimited | $50-200/month |

## Current System Behavior

The trading system now:

1. **Checks approval status** using local RPC (fast, no rate limit)
2. **Only approves if needed** (skips if already approved)
3. **Uses local RPC for all reads** (balances, market data)
4. **Uses public RPC only for tx broadcast** (if configured)

This minimizes public RPC usage and avoids rate limits!

## Summary

**Best Setup:**
```bash
# Use local Polygon node for everything
POLYGON_RPC_URL=http://localhost:8545
PUBLIC_RPC_URL=http://localhost:8545
```

**Alternative:**
```bash
# Use premium RPC provider
POLYGON_RPC_URL=https://polygon-mainnet.infura.io/v3/YOUR_KEY
PUBLIC_RPC_URL=https://polygon-mainnet.infura.io/v3/YOUR_KEY
```

**Last Resort:**
```bash
# Try different free public RPCs
PUBLIC_RPC_URL=https://rpc-mainnet.matic.network
# or
PUBLIC_RPC_URL=https://rpc-mainnet.maticvigil.com
```
