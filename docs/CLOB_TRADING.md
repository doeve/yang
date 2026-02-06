# CLOB Trading Implementation

This document explains how to use the CLOB (Central Limit Order Book) trading feature with full EIP-712 signing support.

## Overview

The trading bot supports two execution modes:

### 1. **Pure Onchain Mode** (Default - `use_clob=false`)
- ✅ **BUY**: Fee-free via position splitting
- ❌ **SELL**: Not supported (must hold until resolution)
- **Best for**: Long-term positions, minimal fees
- **Limitation**: Cannot exit positions early

### 2. **CLOB API Mode** (`use_clob=true`)
- ✅ **BUY**: Via CLOB orderbook (~2% fees)
- ✅ **SELL**: Via CLOB orderbook (~2% fees)
- **Best for**: Active trading, dynamic position management
- **Benefit**: Full trading functionality with early exits

## Installation

### Install py-clob-client

```bash
# From the project root
pip install -e .

# Or install py-clob-client directly
pip install py-clob-client>=0.25.0
```

### Verify Installation

```bash
python -c "import py_clob_client; print('✓ py-clob-client installed')"
```

## Usage

### Command Line

#### Enable CLOB Mode

```bash
# Enable CLOB for all operations
python src/paper_trade_unified_new.py --live --clob
```

#### Default Onchain Mode

```bash
# Fee-free onchain mode (no early exits)
python src/paper_trade_unified_new.py --live
```

### Configuration File

Edit `config.yaml`:

```yaml
execution:
  use_clob: true  # Enable CLOB API trading
  # ... other settings
```

### Interactive Config Editor

1. Launch the trading bot
2. Press `C` to open config editor
3. Select option `9` - Use CLOB API (vs onchain)
4. Toggle enabled/disabled
5. Press `S` to save to `config.yaml`

## How It Works

### CLOB Mode Execution Flow

#### BUY Order
1. Creates `OrderArgs` with token_id, price, size, side=BUY
2. Signs order using EIP-712 (handled by py-clob-client)
3. Submits signed order to CLOB API
4. Returns order_id for tracking
5. Polls for fill confirmation

#### SELL Order
1. Creates `OrderArgs` with token_id, price, size, side=SELL
2. Signs order using EIP-712
3. Submits to CLOB API
4. Returns order_id
5. Polls for fill

### Authentication

The CLOB client automatically:
- Generates API credentials from your private key
- Creates HMAC signatures for requests
- Handles EIP-712 order signing
- Manages nonces and salts

**No manual API key setup required** - everything is derived from your `ETH_PRIVATE_KEY`.

## Technical Details

### EIP-712 Domain

```python
domain = {
    "name": "Polymarket CTF Exchange",
    "version": "1",
    "chainId": 137,  # Polygon Mainnet
    "verifyingContract": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
}
```

### Order Structure

```python
Order = {
    "salt": "uint256",           # Random uniqueness
    "maker": "address",          # Your wallet
    "signer": "address",         # Your wallet
    "taker": "address",          # 0x0 (anyone can fill)
    "tokenId": "uint256",        # Market token ID
    "makerAmount": "uint256",    # Amount you're trading
    "takerAmount": "uint256",    # Amount you want
    "expiration": "uint256",     # Order expiry time
    "nonce": "uint256",          # Order nonce
    "feeRateBps": "uint256",     # Fee (basis points)
    "side": "uint8",             # 1=BUY, 2=SELL
    "signatureType": "uint8"     # 0=EOA
}
```

### Order Types

- **GTC** (Good-Til-Cancelled): Default, stays active until filled or cancelled
- **FOK** (Fill-Or-Kill): Must fill immediately or cancel
- **GTD** (Good-Til-Date): Active until expiration time

## Code Integration

### Initialization

```python
from src.execution.onchain_order_executor import OnchainOrderExecutor

executor = OnchainOrderExecutor(
    local_rpc_url="http://localhost:8545",
    private_key=private_key,
    public_rpc_url="https://polygon-rpc.com",
    use_clob=True,  # Enable CLOB mode
)

await executor.connect()
```

### Place Order

```python
# BUY order
result = await executor.place_order(
    token_id="123456789...",
    side="BUY",
    size=10.0,  # 10 shares
    price=0.65  # 65% probability
)

if result.success:
    print(f"Order placed: {result.order_id}")

    # Wait for fill
    filled = await executor.wait_for_fill(result.order_id, timeout=30)
    if filled:
        print("Order filled!")
```

### Cancel Order

```python
cancelled = await executor.cancel_order(order_id)
```

## Fee Comparison

### Pure Onchain Mode
- **BUY**: ~$0.01-0.10 (gas only)
- **SELL**: Not available
- **Total**: ~$0.01-0.10 per trade

### CLOB Mode
- **BUY**: ~2% + gas
- **SELL**: ~2% + gas
- **Total**: ~4% round-trip + gas

**Example**: Trading $100
- Onchain: $0.10 (entry only, hold until resolution)
- CLOB: $4.00 (entry + exit with dynamic trading)

## Troubleshooting

### py-clob-client Not Installed

```
ImportError: No module named 'py_clob_client'
```

**Solution**:
```bash
pip install py-clob-client>=0.25.0
```

### CLOB Client Initialization Failed

```
Failed to initialize CLOB client: ...
```

**Possible causes**:
1. Invalid private key format
2. Network connectivity issues
3. CLOB API unavailable

**Solution**:
- Check `.env` file has valid `ETH_PRIVATE_KEY`
- Verify internet connection
- Check CLOB API status at https://status.polymarket.com

### Order Not Filling

**Possible causes**:
1. Price too far from market
2. Insufficient liquidity
3. Market closed/paused

**Solution**:
- Check current market price
- Adjust order price closer to midpoint
- Cancel and retry with better price

### "SELL not supported in pure onchain mode"

**Cause**: Trying to sell with `use_clob=false`

**Solution**:
- Enable CLOB mode: `--clob` flag or config editor
- Or hold position until resolution and redeem

## Monitoring

### Logs

CLOB operations are logged with structlog:

```json
{
  "event": "clob_buy_order_placed",
  "order_id": "abc123...",
  "token_id": "456789...",
  "price": 0.65,
  "size": 10.0,
  "timestamp": "2026-02-06T..."
}
```

### Order Status

Check order status in logs:
- `LIVE`: Active on orderbook
- `MATCHED`: Filled
- `CANCELLED`: Cancelled by user
- `EXPIRED`: Expired before fill
- `FAILED`: Failed to place

## Best Practices

### When to Use CLOB Mode

✅ **Use CLOB when**:
- You need to exit positions dynamically
- The model predicts market movements
- You're actively managing risk
- Trading based on signals/indicators

❌ **Don't use CLOB when**:
- You plan to hold until resolution
- Fees would eat profits (small positions)
- Model confidence is low
- Market has low liquidity

### Risk Management

1. **Fee Awareness**: ~4% round-trip fees add up quickly
2. **Liquidity Check**: Ensure sufficient orderbook depth
3. **Slippage**: Allow reasonable price tolerance
4. **Timeout**: Set appropriate fill timeouts
5. **Position Sizing**: Account for fees in sizing

### Performance Optimization

1. **Batch Operations**: Group trades when possible
2. **Price Updates**: Monitor midpoint before placing orders
3. **Fill Monitoring**: Use efficient polling intervals
4. **Order Cancellation**: Clean up unfilled orders promptly

## Environment Variables

Required in `.env`:

```bash
# Wallet & Keys (required)
ETH_PRIVATE_KEY=your_private_key_here

# RPC Endpoints
POLYGON_RPC_URL=http://localhost:8545  # Optional: local node
PUBLIC_RPC_URL=https://polygon-rpc.com  # Fallback RPC

# Proxy (optional)
SOCKS5_PROXY=socks5://127.0.0.1:1080
```

## References

- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [Polymarket CLOB API Docs](https://docs.polymarket.com/developers/CLOB/)
- [EIP-712 Specification](https://eips.ethereum.org/EIPS/eip-712)
- [Polymarket CTF Exchange](https://github.com/Polymarket/ctf-exchange)

## Support

For issues:
1. Check logs for error messages
2. Verify py-clob-client installation
3. Test with small amounts first
4. Report bugs with full logs
