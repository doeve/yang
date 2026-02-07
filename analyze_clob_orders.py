#!/usr/bin/env python3
"""
Analyze CLOB orders to find actual execution prices and fees.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Monkey-patch httpx with SOCKS proxy BEFORE importing py_clob_client
socks5_proxy = os.getenv("SOCKS5_PROXY", "socks5://127.0.0.1:1080")
if socks5_proxy:
    try:
        import httpx
        from httpx_socks import SyncProxyTransport, AsyncProxyTransport

        # Store original transports
        _original_client_init = httpx.Client.__init__
        _original_async_client_init = httpx.AsyncClient.__init__

        def patched_client_init(self, *args, **kwargs):
            if 'transport' not in kwargs and 'mounts' not in kwargs:
                kwargs['transport'] = SyncProxyTransport.from_url(socks5_proxy)
            return _original_client_init(self, *args, **kwargs)

        def patched_async_client_init(self, *args, **kwargs):
            if 'transport' not in kwargs and 'mounts' not in kwargs:
                kwargs['transport'] = AsyncProxyTransport.from_url(socks5_proxy)
            return _original_async_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched_client_init
        httpx.AsyncClient.__init__ = patched_async_client_init
        print(f"✓ SOCKS5 proxy configured: {socks5_proxy}")
    except ImportError:
        print("Warning: httpx-socks not available, proxy not configured")

from py_clob_client.client import ClobClient

# Order IDs from the session (extracted from logs)
order_ids = [
    "0xbba10c10d6c2deac7323cf9660fc168b9ff29d43283e4f1a47db1dd88c9385c2",  # BUY
    "0x5e132ddfbafece2060587f5d84250a27e834e636a66251e9a05c2ecc84f98cd4",  # SELL
    "0xe9f4f2edd055aaddc78b4c28c07e39ad5867a35fe93bd869d08d6e696a4811bc",  # BUY
    "0x4c5b746a835c0b433b05742f2f0f966348cb0468345b9cc1619fc58f9252bdd7",  # SELL (timed out?)
    "0x9766cabd8002ac6b3ef48fce5dec7879da26ff289d4956e31a4c051469f257a7",  # BUY
    "0x5aab65c3e699c9da4c088f17f77d5536b2ca8ce3ba61c5ca94bd0bad076ab6b7",  # SELL
    "0xe5c5f0fb04e13a52ca7d1fd9f6df07c2c2e9ecef65f44bc6b0a71dd0e1df8b3d",  # BUY
    "0x0d7d3e02bb1c6e34a40e5f9f97de7b1b58e86e4ac3c54d0d5dfb9e0b1b5d0bad",  # SELL
]

# Initialize CLOB client with full credentials
host = "https://clob.polymarket.com"
private_key = os.getenv("ETH_PRIVATE_KEY")
api_key = os.getenv("POLYMARKET_API_KEY")
api_secret = os.getenv("POLYMARKET_API_SECRET")
api_passphrase = os.getenv("POLYMARKET_PASSPHRASE")
chain_id = 137

if not private_key:
    print("Error: ETH_PRIVATE_KEY not found in .env")
    sys.exit(1)

# Create client with both signing key and API credentials
client = ClobClient(
    host=host,
    key=private_key,
    chain_id=chain_id,
    creds={
        "api_key": api_key,
        "api_secret": api_secret,
        "api_passphrase": api_passphrase
    } if api_key and api_secret and api_passphrase else None
)

print("\n" + "=" * 80)
print("CLOB ORDER ANALYSIS - Session 2026-02-07 02:16-02:38")
print("=" * 80)
print()

total_fees = 0
total_slippage = 0
orders_analyzed = 0

for order_id in order_ids:
    try:
        # Get order details
        order = client.get_order(order_id)

        if not order:
            print(f"Order ID: {order_id[:16]}... - NOT FOUND")
            print()
            continue

        orders_analyzed += 1
        side = order.get('side', 'N/A')
        original_price = float(order.get('price', 0))
        original_size = float(order.get('original_size', 0))
        size_matched = float(order.get('size_matched', 0))
        status = order.get('status', 'N/A')

        print(f"Order #{orders_analyzed}: {order_id[:16]}...")
        print(f"  Side: {side}")
        print(f"  Original Price: {original_price:.4f}")
        print(f"  Size Requested: {original_size:.2f}")
        print(f"  Size Matched: {size_matched:.2f}")
        print(f"  Status: {status}")

        # Get trades for this order to see actual execution
        trades = order.get('trades', [])
        if trades:
            print(f"  Executions: {len(trades)}")
            order_total_cost = 0
            order_total_fee = 0

            for i, trade in enumerate(trades, 1):
                trade_price = float(trade.get('price', 0))
                trade_size = float(trade.get('size', 0))
                fee_rate_bps = float(trade.get('fee_rate_bps', 0))
                fee = float(trade.get('fee', 0))

                trade_cost = trade_price * trade_size
                order_total_cost += trade_cost
                order_total_fee += fee

                slippage = abs(trade_price - original_price)

                print(f"    Execution {i}:")
                print(f"      Price: {trade_price:.4f} (slippage: {slippage:.4f})")
                print(f"      Size: {trade_size:.2f}")
                print(f"      Cost: ${trade_cost:.6f}")
                print(f"      Fee: ${fee:.6f} ({fee_rate_bps:.1f} bps)")

                total_slippage += slippage * trade_size

            # Calculate average execution price
            avg_price = order_total_cost / size_matched if size_matched > 0 else 0
            print(f"  Average Execution Price: {avg_price:.4f}")
            print(f"  Total Order Cost: ${order_total_cost:.6f}")
            print(f"  Total Fees: ${order_total_fee:.6f}")

            total_fees += order_total_fee
        else:
            print("  No executions found (order not filled)")

        print()
    except Exception as e:
        print(f"Order ID: {order_id[:16]}...")
        print(f"  Error: {e}")
        print()

print("=" * 80)
print(f"Orders Analyzed: {orders_analyzed}")
print(f"Total Fees Paid: ${total_fees:.6f}")
if orders_analyzed > 0:
    print(f"Average Fee per Order: ${total_fees/orders_analyzed:.6f}")
print("=" * 80)
