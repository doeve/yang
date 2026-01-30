# Live Trading Testing Guide

I've created a comprehensive testing suite for your live trading flow. Here's what you have:

## 📁 Files Created

### 1. `test_live_flow.py` - Main Test Script
**Purpose**: Tests the complete buy → sell → redeem flow with $1 USDC

**What it does**:
- ✅ Buys position by splitting $1 USDC into YES/NO tokens (onchain)
- ✅ Sells position by merging YES/NO tokens back to USDC (onchain)
- ✅ Redeems resolved positions automatically
- ✅ Reports gas costs, transaction hashes, and balance changes
- ✅ Comprehensive error handling and logging

**Usage**:
```bash
# Full test (buy + sell + redeem)
python test_live_flow.py

# Skip redemption (if no resolved positions)
python test_live_flow.py --skip-redeem
```

### 2. `check_live_setup.py` - Diagnostic Tool
**Purpose**: Verifies your environment is correctly configured

**What it checks**:
- ✅ Environment variables (.env file)
- ✅ RPC connection to Polygon
- ✅ Wallet MATIC and USDC balances
- ✅ USDC approvals for CTF contract
- ✅ Polymarket API access

**Usage**:
```bash
python check_live_setup.py
```

**Example output**:
```
✅ PASS - Environment Variables
✅ PASS - RPC Connection
✅ PASS - Wallet Balances
⚠️  FAIL - USDC Approvals (needs approval)
✅ PASS - Market API Access

Run test_live_flow.py to set approvals automatically
```

### 3. `TEST_LIVE_FLOW_README.md` - Documentation
Complete guide with setup instructions, troubleshooting, and expected costs.

## 🚀 Quick Start

### Step 1: Check Your Setup
```bash
# With SOCKS5 proxy
python check_live_setup.py

# Or without proxy
python check_live_setup.py --no-proxy
```

Fix any issues reported (see troubleshooting below).

### Step 2: Run Test
```bash
# With SOCKS5 proxy
python test_live_flow.py

# Or without proxy (direct connection)
python test_live_flow.py --no-proxy

# Skip redemption and no proxy
python test_live_flow.py --skip-redeem --no-proxy
```

This will:
1. Ask for confirmation (real transactions!)
2. Find an active market
3. Buy $1 position
4. Sell it immediately
5. Try to redeem any resolved positions
6. Report results and costs

### Step 3: Review Results
Look for:
- ✅ All tests passed
- Transaction hashes on Polygonscan
- Gas costs (should be ~$0.02-0.05 total)
- USDC balance recovered (minus gas)

## 🔧 Troubleshooting

### Common Issues

#### 1. "ETH_PRIVATE_KEY not set"
**Fix**:
```bash
cp .env.example .env
# Edit .env and add your private key
nano .env
```

#### 2. "Failed to connect to RPC"
**Options**:
- **Use local node**: Make sure your Polygon node is running on port 8545
- **Use public RPC**: Set `POLYGON_RPC_URL=https://polygon-rpc.com` in .env

#### 3. "Insufficient balance"
**Fix**:
- Need at least $1 USDC on Polygon
- Need at least 0.1 MATIC for gas
- Bridge from Ethereum or buy on exchange

#### 4. "Market API access failed"
**Options**:
- **With SOCKS5**: Set `SOCKS5_PROXY=socks5://127.0.0.1:1080` in .env
- **Without proxy**: Try direct connection (may work depending on location)

#### 5. "Approval failed"
**Fix**:
- Make sure you have enough MATIC for gas
- The script will retry automatically
- First run always needs approval (one-time gas cost)

#### 6. Live trading in main script doesn't work
**Check**:
```bash
# Verify config
cat config.yaml | grep trading_mode
# Should show: trading_mode: live

# Run diagnostic
python check_live_setup.py

# Check if executor is initialized
# Look for this in logs:
# "🔴 LIVE TRADING MODE (ONCHAIN - NO POLYMARKET FEES)"
```

## 💰 Costs

### Test Script (`test_live_flow.py`)
- **Gas fees**: ~$0.02-0.05 (3-4 transactions)
- **Test amount**: $1.00 USDC (recovered after)
- **Net cost**: Just gas (~$0.02-0.05)

### Each Transaction
- Split position (BUY): ~$0.01-0.02
- Merge position (SELL): ~$0.01-0.02
- Redeem position: ~$0.01-0.02

**Note**: Costs vary with Polygon gas prices. Usually very cheap!

## 🎯 Integration with Main Script

The main trading script (`src/paper_trade_unified_new.py`) uses the same executors.

### Enable Live Trading
```bash
# 1. Edit config
nano config.yaml
# Set: trading_mode: live

# 2. Verify setup
python check_live_setup.py

# 3. Run trading script
python -m src.paper_trade_unified_new --live
```

### What Happens
- Model makes trading decisions
- OnchainOrderExecutor executes via position splitting/merging
- No CLOB API fees (direct contract interaction)
- Auto-redemption on market resolution
- Real-time balance tracking

## 📊 Test Output Example

```
Setting up test environment...
Checking USDC approvals...
✅ Approvals confirmed
✅ Connected! USDC Balance: $10.50

Selected Market:
  Question: Will Bitcoin reach $100k by end of 2024?
  Condition ID: 0x1234567890abcdef

TEST 1: BUY Position ($1.0)
Market: Will Bitcoin reach $100k by end of 2024?...
Condition ID: 0x123456789012...
Balance before: $10.50
Splitting $1.0 USDC into YES/NO tokens...
✅ BUY successful!
   TX: 0xabcdef123456...
   Gas used: 150000
   USDC spent: $1.00
   YES tokens: 1.00
   NO tokens: 1.00

Waiting 5 seconds for blockchain state to update...

TEST 2: SELL Position (merge to USDC)
USDC balance before: $9.48
YES tokens before: 1.00
NO tokens before: 1.00
Merging 1.00 token pairs back to USDC...
✅ SELL successful!
   TX: 0x123456abcdef...
   Gas used: 120000
   USDC recovered: $1.00

TEST 3: REDEEM Resolved Positions
Searching for resolved positions to redeem...
Found 10 unique traded assets
✅ RESOLVED: Will Trump win 2024 election?...
   🎉 Redeemed $5.23
✅ Redemption test complete!
   Total redeemed: $5.23

════════════════════════════════════════════════════════════
🎉 ALL TESTS PASSED! 🎉
════════════════════════════════════════════════════════════

Final USDC Balance: $14.68
```

## ⚠️ Safety Notes

1. **Start small**: Test with $1-10 first
2. **Use burner wallet**: Don't use your main wallet for testing
3. **Check gas prices**: High gas = expensive tests
4. **Never commit .env**: Keep private keys secure
5. **Monitor transactions**: Check Polygonscan for confirmations

## 🔗 Useful Links

- **Polygonscan**: https://polygonscan.com/
- **Polygon Gas Tracker**: https://polygonscan.com/gastracker
- **Polymarket Docs**: https://docs.polymarket.com/

## 📝 Next Steps

1. Run diagnostic: `python check_live_setup.py`
2. Fix any issues
3. Run test: `python test_live_flow.py`
4. Review results
5. Enable live trading in main script if tests pass

## 🐛 Debug Mode

For detailed logging, set environment variable:
```bash
export PYTHONUNBUFFERED=1
python test_live_flow.py 2>&1 | tee test_output.log
```

This saves all output to `test_output.log` for debugging.

## 💡 Tips

- **First run takes longer**: USDC approval transaction needed
- **Test during low gas**: Weekends typically cheaper
- **Keep MATIC balance**: Need for all transactions
- **Monitor positions**: Use Polymarket UI to verify
- **Redemption is optional**: Can skip if no resolved markets

## ❓ Questions?

Check the documentation:
- `TEST_LIVE_FLOW_README.md` - Detailed test documentation
- `ONCHAIN_TRADING_SUMMARY.md` - How onchain execution works
- `docs/ONCHAIN_EXECUTOR.md` - Executor API reference
- `QUICK_REFERENCE.md` - Trading commands reference
