# Quick Reference - Onchain Trading

## 🚀 Quick Start (30 seconds)

```bash
# 1. Setup (one-time)
cp .env.example .env
nano .env  # Set ETH_PRIVATE_KEY

# 2. Run
python src/paper_trade_unified_new.py --live
```

Done! Automatic redemption included. 🎉

## 📁 Configuration Files

### .env (Your Secrets)
```bash
ETH_PRIVATE_KEY=your_key_here          # Required for live trading
POLYGON_RPC_URL=http://localhost:8545  # Your local node
PUBLIC_RPC_URL=https://polygon-rpc.com # Public RPC (optional)
SOCKS5_PROXY=socks5://127.0.0.1:1080  # If needed (optional)
```

### config.yaml (Trading Settings)
```yaml
trading_mode: live  # or "paper"

risk:
  max_daily_loss_pct: 20.0
  max_position_size_usdc: 100.0

execution:
  use_public_rpc_for_redeem: true
  public_rpc_url: "https://polygon-rpc.com"

model:
  path: "./logs/market_predictor_v1"
```

## 🎯 Common Commands

```bash
# Paper trading (testing)
python src/paper_trade_unified_new.py

# Live trading (real money)
python src/paper_trade_unified_new.py --live

# Manual redemption (optional - it's automatic!)
python scripts/redeem_positions.py

# With custom model
python src/paper_trade_unified_new.py --live --model ./other_model
```

## 💰 Cost Savings

| Method | Fees | Gas | Total | Savings |
|--------|------|-----|-------|---------|
| CLOB API | $2.00 | $0.05 | **$2.05** | - |
| Onchain | $0.00 | $0.05 | **$0.05** | **97.5%** |

Per $100 trade on Polygon.

## ✨ Features

- ✅ **No CLOB Fees** - Direct onchain execution
- ✅ **Automatic Redemption** - After each market close
- ✅ **Smart Gas** - Checks balances before redemption
- ✅ **Auto Config** - Loads from .env and config.yaml
- ✅ **Safe** - Paper mode for testing

## 🔧 Troubleshooting

**"ETH_PRIVATE_KEY not set"**
```bash
cp .env.example .env
nano .env  # Add your key
```

**"Failed to connect"**
```bash
# Check RPC in .env
cat .env | grep RPC
```

**"No tokens to redeem"**
```
This is normal! It means you don't hold tokens for that outcome.
Not an error - redemption properly skipped.
```

## 📊 What Happens Automatically

```
Market Opens
    ↓
Trade (onchain, no fees)
    ↓
Market Closes (15 min)
    ↓
Settlement
    ↓
Auto-Redeem ← Automatic!
    ↓
USDC in wallet ✅
```

## 🔐 Security Checklist

- ✅ .env is in .gitignore
- ✅ Never commit .env
- ✅ Use dedicated trading wallet
- ✅ Keep private key secure

## 📚 Documentation

- `SETUP_GUIDE.md` - Complete setup
- `ONCHAIN_QUICKSTART.md` - Quick start
- `ONCHAIN_TRADING.md` - Trading guide
- `CONFIG_UPDATE_SUMMARY.md` - Config details

## 🎓 Example Session

```bash
# First time setup
cp .env.example .env
nano .env  # Set ETH_PRIVATE_KEY=abc123...

# Run paper trading (testing)
python src/paper_trade_unified_new.py
# Shows: 📝 PAPER TRADING MODE

# Run live trading
python src/paper_trade_unified_new.py --live
# Shows: 🔴 LIVE TRADING MODE (ONCHAIN - NO POLYMARKET FEES)
#        🔵 Using onchain execution (bypassing CLOB fees)
#        🔵 Auto-redemption enabled

# Watch it trade and redeem automatically!
# After market closes:
# 🎁 Auto-redeeming resolved position...
# ✅ Redeemed $95.00 USDC
```

## 💡 Tips

1. **Test First**: Run paper mode before live
2. **Check Balance**: Ensure MATIC for gas
3. **Monitor**: Watch the console output
4. **Be Patient**: Onchain takes ~10-30 seconds per tx
5. **Gas Settings**: Adjust in config.yaml if needed

## ⚡ Key Differences

### Old (CLOB API)
- Requires API keys
- Pays 2% fee per trade
- Fast execution
- Manual redemption

### New (Onchain)
- No API keys needed
- No trading fees (only gas)
- Slower (blockchain)
- **Automatic redemption** ✨

## 🎯 When to Use

**Use Onchain (New):**
- Want to save on fees
- Trading larger amounts
- Okay with ~30 sec execution
- Want automatic redemption

**Use CLOB API (Old):**
- Need instant fills
- Trading small amounts
- Want limit orders
- Need traditional orderbook

## 📞 Support

**Issues?**
- Check `SETUP_GUIDE.md` first
- Review error messages carefully
- Verify .env settings
- Test with paper mode

**Good to know:**
- Config loaded automatically
- Redemption happens automatically
- No manual setup needed
- Just edit .env and run!

---

**Remember:** All config in `.env` and `config.yaml` - no manual exports needed! 🚀
