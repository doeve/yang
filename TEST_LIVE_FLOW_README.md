# Live Trading Flow Test

This script tests the complete live trading flow for Polymarket onchain execution.

## What It Tests

1. **BUY** - Splits $1 USDC into YES/NO tokens using onchain position splitting
2. **SELL** - Merges YES/NO tokens back to USDC using onchain position merging
3. **REDEEM** - Redeems winning positions from resolved markets

## Features

- ✅ Tests real onchain transactions (small $1 amount)
- ✅ No CLOB API fees (direct contract interaction)
- ✅ Comprehensive error handling and logging
- ✅ Balance tracking before/after each operation
- ✅ Gas usage reporting

## Requirements

1. **USDC Balance**: At least $1 USDC + gas on Polygon
2. **MATIC Balance**: For gas fees (~$0.01 per transaction)
3. **RPC Access**: Local Polygon node or public RPC
4. **SOCKS5 Proxy**: Optional, for Polymarket API access

## Setup

1. **Configure Environment**:
```bash
cp .env.example .env
# Edit .env and set:
# - ETH_PRIVATE_KEY (your Polygon wallet private key)
# - POLYGON_RPC_URL (your local node or public RPC)
# - SOCKS5_PROXY (optional, for API access)
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Test (with redemption)
```bash
python test_live_flow.py
```

### Skip Redemption Test
```bash
python test_live_flow.py --skip-redeem
```

This is useful if you don't have any resolved positions to redeem.

### Disable SOCKS5 Proxy
```bash
python test_live_flow.py --no-proxy
```

Use this if you don't have a SOCKS5 proxy or want to use direct connection.

### Combined Flags
```bash
python test_live_flow.py --skip-redeem --no-proxy
```

## What Happens

### Step 1: Setup
- Connects to Polygon blockchain
- Initializes onchain executors
- Checks USDC approvals for CTF contract
- Verifies sufficient balance

### Step 2: Buy Test
- Fetches an active Polymarket market
- Splits $1 USDC into YES/NO tokens
- Verifies tokens were received
- Reports transaction hash and gas used

### Step 3: Sell Test
- Merges YES/NO tokens back to USDC
- Verifies USDC was recovered
- Reports transaction hash and gas used

### Step 4: Redeem Test (optional)
- Scans all traded assets for resolved markets
- Redeems winning positions automatically
- Reports total USDC recovered

## Expected Output

```
Setting up test environment...
✅ Connected! USDC Balance: $10.50

Selected Market:
  Question: Will Bitcoin reach $100k by end of 2024?
  Condition ID: 0x1234567890abcdef

TEST 1: BUY Position ($1.0)
Splitting $1.0 USDC into YES/NO tokens...
✅ BUY successful!
   TX: 0xabcdef...
   Gas used: 150000
   USDC spent: $1.00
   YES tokens: 1.00
   NO tokens: 1.00

TEST 2: SELL Position (merge to USDC)
Merging 1.00 token pairs back to USDC...
✅ SELL successful!
   TX: 0x123456...
   Gas used: 120000
   USDC recovered: $1.00

TEST 3: REDEEM Resolved Positions
Searching for resolved positions to redeem...
✅ Redemption test complete!
   Total redeemed: $0.00
   Successful: 0
   Skipped: 5
   Failed: 0

🎉 ALL TESTS PASSED! 🎉

Final USDC Balance: $10.48
```

## Cost

Each test run costs approximately:
- Gas fees: ~$0.02-0.05 (depends on Polygon gas prices)
- No CLOB trading fees (onchain execution)
- Test amount: $1.00 (recovered after sell)

**Net cost**: Just gas fees (~$0.02-0.05)

## Troubleshooting

### "Insufficient balance"
Make sure you have at least $1 USDC + $0.10 MATIC on Polygon.

### "Failed to connect"
Check your RPC URL in `.env`. Try using `https://polygon-rpc.com` if your local node isn't working.

### "Approval error"
The script will automatically approve USDC for the CTF contract on first run. This requires gas.

### "No active markets found"
This is rare. The script will retry with different parameters.

### "Merge failed: unequal balances"
This happens if the split didn't create equal YES/NO tokens. The script will handle this gracefully.

## Integration with Main Script

The main trading script `src/paper_trade_unified_new.py` uses the same executors when `trading_mode: live` is set in `config.yaml`.

To enable live trading in the main script:
1. Edit `config.yaml`
2. Set `trading_mode: live`
3. Run: `python -m src.paper_trade_unified_new --live`

## Safety Notes

⚠️ **This executes REAL on-chain transactions!**

- Start with small amounts ($1)
- Test on a burner wallet first
- Monitor gas prices (high gas = expensive tests)
- Keep private keys secure
- Never commit `.env` file to git

## Questions?

Check the main documentation:
- `ONCHAIN_TRADING_SUMMARY.md` - Onchain execution overview
- `docs/ONCHAIN_EXECUTOR.md` - Executor API details
- `QUICK_REFERENCE.md` - Trading quick reference
