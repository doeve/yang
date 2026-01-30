# Setup Guide - Onchain Trading & Automatic Redemption

## Quick Setup (5 minutes)

### 1. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
nano .env  # or your preferred editor
```

### 2. Set Required Variables in .env

```bash
# Wallet (REQUIRED for live trading)
ETH_PRIVATE_KEY=your_private_key_here  # Without 0x prefix

# Polygon RPC URLs (REQUIRED)
POLYGON_RPC_URL=http://localhost:8545     # Your local Polygon node
PUBLIC_RPC_URL=https://polygon-rpc.com    # Public RPC for transactions

# SOCKS5 Proxy (if needed for Polymarket API access)
SOCKS5_PROXY=socks5://127.0.0.1:1080

# Polymarket API (optional - only needed for CLOB API features)
# Not required for onchain trading!
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_PASSPHRASE=
```

### 3. Configure Trading Settings in config.yaml

```yaml
# Trading mode
trading_mode: live  # or "paper" for testing

# Risk controls
risk:
  max_daily_loss_pct: 20.0
  max_position_size_usdc: 100.0
  min_balance_usdc: 10.0

# Execution (onchain settings)
execution:
  use_public_rpc_for_redeem: true
  public_rpc_url: "https://polygon-rpc.com"  # Can be overridden by .env
  order_timeout_seconds: 30
  poll_interval_seconds: 5

# Model
model:
  path: "./logs/market_predictor_v1"
  min_confidence: 0.3
  min_expected_return: 0.02
```

### 4. Run Trading

```bash
# Paper trading (safe testing)
python src/paper_trade_unified_new.py

# Live trading with onchain execution (no CLOB fees!)
python src/paper_trade_unified_new.py --live
```

### 5. Redeem Positions Manually (Optional)

```bash
# Automatic redemption runs after each market close
# But you can also manually redeem all resolved positions:
python scripts/redeem_positions.py
```

## Configuration Hierarchy

Settings are loaded in this priority order:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (.env file)
3. **Config file** (config.yaml)
4. **Defaults** (lowest priority)

### Example: RPC URL Resolution

```
# If you set PUBLIC_RPC_URL in .env:
PUBLIC_RPC_URL=https://polygon-rpc.com

# It will be used instead of config.yaml setting
# Command-line args would override both
```

## What Gets Loaded From Where

### From .env File
- ✅ ETH_PRIVATE_KEY (your wallet)
- ✅ POLYGON_RPC_URL (local node)
- ✅ PUBLIC_RPC_URL (transaction broadcast)
- ✅ SOCKS5_PROXY (if needed)
- ✅ POLYMARKET_API_* (optional, not needed for onchain)

### From config.yaml
- ✅ trading_mode (paper or live)
- ✅ Risk settings (max loss, position sizes)
- ✅ Execution settings (timeouts, RPC preferences)
- ✅ Model settings (path, confidence thresholds)
- ✅ Logging settings

### From CLI Arguments
- ✅ --live (force live mode)
- ✅ --model (model path)
- ✅ --balance (starting balance)
- ✅ --min-confidence, --min-return

## Features Enabled

### ✅ Onchain Trading (No CLOB Fees)
- Uses OnchainOrderExecutor
- Position splitting/merging
- Direct contract interaction
- Saves ~97.5% on fees

### ✅ Automatic Redemption
- After each market close
- Checks balances before redemption
- No wasted gas
- Updates wallet balance automatically

### ✅ Smart Configuration
- No manual environment exports needed
- Just edit .env and config.yaml
- All settings in one place

## Validation

The system automatically validates your configuration:

```python
# When you run with --live, it checks:
✅ ETH_PRIVATE_KEY is set
✅ POLYGON_RPC_URL is accessible
✅ PUBLIC_RPC_URL is valid

# If anything is missing, you'll see:
❌ Config Error: ETH_PRIVATE_KEY required for live trading
💡 Copy .env.example to .env and set required variables
```

## Security Best Practices

### 1. Never Commit Secrets
```bash
# .env is in .gitignore - keep it that way!
# Never commit .env to git
```

### 2. Use Environment Variables for Secrets
```bash
# Good: In .env file (not tracked)
ETH_PRIVATE_KEY=abc123...

# Bad: Hardcoded in config.yaml (tracked by git)
# Don't do this!
```

### 3. Limit Exposure
```bash
# Consider using a dedicated trading wallet
# Not your main wallet with all your funds
```

## Example Setup Session

```bash
# 1. Clone and enter repo
cd /home/dave/projects/yang

# 2. Copy example config
cp .env.example .env

# 3. Edit .env
nano .env
# Set: ETH_PRIVATE_KEY=your_key_here

# 4. Verify config.yaml is set to live mode
cat config.yaml
# Should show: trading_mode: live

# 5. Test with paper trading first
python src/paper_trade_unified_new.py
# Should show: 📝 PAPER TRADING MODE

# 6. Run live trading
python src/paper_trade_unified_new.py --live
# Should show: 🔴 LIVE TRADING MODE (ONCHAIN - NO POLYMARKET FEES)

# 7. Watch automatic redemption happen
# After each 15-min candle closes, you'll see:
# 🎁 Auto-redeeming resolved position...
# ✅ Redeemed $X.XX USDC
```

## Troubleshooting

### "ETH_PRIVATE_KEY not set"
```bash
# Check your .env file exists
ls -la .env

# If not, copy from example
cp .env.example .env

# Edit and add your private key
nano .env
```

### "Failed to connect to Polygon network"
```bash
# Check your RPC URLs in .env
cat .env | grep RPC

# Test RPC connection
curl -X POST $POLYGON_RPC_URL \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

### "Config file not found"
```bash
# Make sure you're in the project root
pwd
# Should be: /home/dave/projects/yang

# Check config.yaml exists
ls -la config.yaml
```

### "Redemption failed"
```bash
# Check you have tokens to redeem
# The system automatically checks balances
# "No tokens held" is normal - not an error

# If you get other errors:
# 1. Check gas settings in config.yaml
# 2. Verify PUBLIC_RPC_URL is accessible
# 3. Ensure wallet has MATIC for gas
```

## Advanced: Override Config Per Run

```bash
# Use different model
python src/paper_trade_unified_new.py --model ./other_model

# Adjust confidence thresholds
python src/paper_trade_unified_new.py --min-confidence 0.5 --min-return 0.03

# Change starting balance (paper mode)
python src/paper_trade_unified_new.py --balance 5000

# Custom log directory
python src/paper_trade_unified_new.py --log-dir ./logs/custom
```

## Summary

**Setup Steps:**
1. ✅ Copy `.env.example` → `.env`
2. ✅ Set `ETH_PRIVATE_KEY` in `.env`
3. ✅ Verify `config.yaml` settings
4. ✅ Run `python src/paper_trade_unified_new.py --live`

**No Need To:**
- ❌ Export environment variables manually
- ❌ Set PUBLIC_RPC_URL every time
- ❌ Configure POLYMARKET_API_* (onchain bypasses CLOB)
- ❌ Run separate redemption scripts (it's automatic!)

Everything is configured once in `.env` and `config.yaml`, then just run the trading script!
