# Configuration Update Summary

## What Changed

The system now **automatically loads all configuration** from `.env` and `config.yaml` files. No more manual environment variable exports!

## Changes Made

### 1. Added `PUBLIC_RPC_URL` to Configuration

**File: `.env.example`**
```bash
# NEW: Public RPC for transaction broadcast
PUBLIC_RPC_URL=https://polygon-rpc.com
```

**File: `src/config.py`**
```python
# Added public_rpc_url field
public_rpc_url: str = ""

# Load from environment with fallback to config.yaml
config.public_rpc_url = os.getenv("PUBLIC_RPC_URL", config.execution.public_rpc_url)
```

### 2. Updated Executors to Use Config

**File: `src/paper_trade_unified_new.py`**
```python
# OLD: Hardcoded RPC URLs
self.executor = OnchainOrderExecutor(
    public_rpc_url="https://polygon-rpc.com",  # Hardcoded
)

# NEW: From config
self.executor = OnchainOrderExecutor(
    public_rpc_url=self.trading_config.public_rpc_url,  # From .env or config.yaml
)
```

### 3. Simplified Redemption Script

**File: `scripts/redeem_positions.py`**

**Before:**
```python
# Required manual arguments
parser.add_argument("--local-rpc", default=os.getenv(...))
parser.add_argument("--public-rpc", default=os.getenv(...))
parser.add_argument("--private-key", default=os.getenv(...))
# etc...

executor = OnchainExecutor(
    local_rpc_url=args.local_rpc,
    private_key=args.private_key,
    # ...
)
```

**After:**
```python
# Load config automatically
config = load_config()

# No arguments needed - everything from .env and config.yaml!
executor = OnchainExecutor(
    local_rpc_url=config.polygon_rpc_url,
    private_key=config.eth_private_key,
    public_rpc_url=config.public_rpc_url,
    use_public_rpc=config.execution.use_public_rpc_for_redeem,
    socks5_proxy=config.socks5_proxy,
)
```

### 4. New Documentation

**Created:**
- `SETUP_GUIDE.md` - Complete setup instructions
- Updated `ONCHAIN_QUICKSTART.md` - Simplified quick start

## How It Works Now

### Configuration Hierarchy

```
1. Command-line arguments (highest priority)
   ↓
2. .env file (your secrets and endpoints)
   ↓
3. config.yaml (trading settings)
   ↓
4. Defaults (fallback values)
```

### Example: PUBLIC_RPC_URL

```bash
# Option 1: Set in .env (recommended for per-environment settings)
PUBLIC_RPC_URL=https://polygon-rpc.com

# Option 2: Set in config.yaml (good for shared settings)
execution:
  public_rpc_url: "https://polygon-rpc.com"

# Option 3: Use default (https://polygon-rpc.com)

# Priority: .env > config.yaml > default
```

## Before vs After

### Before: Manual Exports

```bash
# Had to export every time
export PRIVATE_KEY="0x..."
export POLYGON_RPC_URL="http://localhost:8545"
export PUBLIC_RPC_URL="https://polygon-rpc.com"
export SOCKS5_PROXY="socks5://127.0.0.1:1080"

# Then run
python scripts/redeem_positions.py
```

### After: Automatic Loading

```bash
# One-time setup (copy and edit)
cp .env.example .env
nano .env  # Set ETH_PRIVATE_KEY

# Just run - config loaded automatically!
python scripts/redeem_positions.py

# Or for trading
python src/paper_trade_unified_new.py --live
```

## Setup Process

### Old Way (Manual)
1. Set ETH_PRIVATE_KEY env var
2. Set POLYGON_RPC_URL env var
3. Set PUBLIC_RPC_URL env var
4. Set SOCKS5_PROXY env var
5. Remember to export before each session
6. Pass arguments to scripts

### New Way (Automatic)
1. Copy `.env.example` → `.env`
2. Edit `.env` with your settings
3. Run scripts - everything loaded automatically!

## Files Modified

### Configuration System
- ✅ `.env.example` - Added PUBLIC_RPC_URL
- ✅ `src/config.py` - Added public_rpc_url field and loading
- ✅ `src/paper_trade_unified_new.py` - Use config.public_rpc_url

### Scripts
- ✅ `scripts/redeem_positions.py` - Simplified to use config system

### Documentation
- ✅ `SETUP_GUIDE.md` - New comprehensive setup guide
- ✅ `ONCHAIN_QUICKSTART.md` - Updated quick start
- ✅ `CONFIG_UPDATE_SUMMARY.md` - This file

## Migration Guide

If you were using the old manual export method:

### Step 1: Create .env File
```bash
cp .env.example .env
```

### Step 2: Transfer Your Settings
```bash
# OLD: You had these exports in your shell
export ETH_PRIVATE_KEY="abc123..."
export POLYGON_RPC_URL="http://localhost:8545"
export PUBLIC_RPC_URL="https://polygon-rpc.com"

# NEW: Put them in .env file
echo 'ETH_PRIVATE_KEY=abc123...' >> .env
echo 'POLYGON_RPC_URL=http://localhost:8545' >> .env
echo 'PUBLIC_RPC_URL=https://polygon-rpc.com' >> .env
```

### Step 3: Remove Manual Exports
```bash
# Remove from .bashrc or .zshrc:
# export ETH_PRIVATE_KEY=...
# export POLYGON_RPC_URL=...
# etc.

# No longer needed - loaded from .env automatically!
```

### Step 4: Run Scripts Normally
```bash
# No arguments needed
python scripts/redeem_positions.py

# Config loaded from .env and config.yaml automatically!
```

## Benefits

### ✅ Simpler Usage
- No manual exports
- No long command lines
- Just run the script

### ✅ Safer Configuration
- Secrets in .env (gitignored)
- Settings in config.yaml (version controlled)
- Clear separation

### ✅ Consistent Behavior
- Same config everywhere
- No forgot-to-export errors
- Predictable results

### ✅ Better Organization
- All settings in two files
- Easy to review
- Easy to share (without secrets)

## Validation

The system validates your configuration automatically:

```python
# When running in live mode:
try:
    config.validate()
except ValueError as e:
    print(f"Config Error: {e}")
    print("Copy .env.example to .env and set required variables")
```

**Checks:**
- ✅ ETH_PRIVATE_KEY is set
- ✅ POLYMARKET_API_* are set (if using CLOB API)
- ✅ Files exist and are readable

## Testing

### Verify Config Loading

```python
from src.config import load_config

# Load config
config = load_config()

# Check values
print(f"Polygon RPC: {config.polygon_rpc_url}")
print(f"Public RPC: {config.public_rpc_url}")
print(f"Trading Mode: {config.trading_mode}")
print(f"Private Key: {'✅ Set' if config.eth_private_key else '❌ Missing'}")
```

### Test Redemption

```bash
# Should load config automatically and show:
python scripts/redeem_positions.py

# Output:
# 📋 Loading configuration from .env and config.yaml...
# 🚀 Starting Position Redemption
#   Wallet: Loading from config...
#   Local RPC: http://localhost:8545
#   Public RPC: https://polygon-rpc.com
```

## Summary

**What You Need:**
1. ✅ `.env` file with ETH_PRIVATE_KEY
2. ✅ `config.yaml` with trading settings
3. ✅ Run scripts - config loaded automatically!

**What You Don't Need:**
- ❌ Manual export commands
- ❌ Long command-line arguments
- ❌ Remembering environment setup

**Result:** Clean, simple, automatic configuration! 🎉
