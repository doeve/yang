#!/usr/bin/env python3
"""
Analyze CLOB orders using direct API calls.
"""
import os
import requests
import urllib3
from dotenv import load_dotenv

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Configure SOCKS5 proxy
socks5_proxy = os.getenv("SOCKS5_PROXY", "socks5://127.0.0.1:1080")
proxies = {
    'http': socks5_proxy,
    'https': socks5_proxy
}

# Order IDs from the session
order_ids = [
    "0xbba10c10d6c2deac7323cf9660fc168b9ff29d43283e4f1a47db1dd88c9385c2",
    "0x5e132ddfbafece2060587f5d84250a27e834e636a66251e9a05c2ecc84f98cd4",
    "0xe9f4f2edd055aaddc78b4c28c07e39ad5867a35fe93bd869d08d6e696a4811bc",
    "0x4c5b746a835c0b433b05742f2f0f966348cb0468345b9cc1619fc58f9252bdd7",
    "0x9766cabd8002ac6b3ef48fce5dec7879da26ff289d4956e31a4c051469f257a7",
]

print("\n" + "=" * 80)
print("CLOB ORDER ANALYSIS - Direct API Calls")
print("=" * 80)
print()

base_url = "https://clob.polymarket.com"
total_fees = 0
orders_analyzed = 0

for order_id in order_ids:
    try:
        # Query order details
        url = f"{base_url}/data/order/{order_id}"
        response = requests.get(url, proxies=proxies, timeout=10, verify=False)

        if response.status_code != 200:
            print(f"Order {order_id[:16]}... - HTTP {response.status_code}")
            print()
            continue

        order = response.json()
        orders_analyzed += 1

        side = order.get('side', 'N/A')
        price = float(order.get('price', 0))
        original_size = float(order.get('original_size', 0))
        size_matched = float(order.get('size_matched', 0))
        status = order.get('status', 'N/A')

        print(f"Order #{orders_analyzed}: {order_id[:16]}...")
        print(f"  Side: {side}")
        print(f"  Price: {price:.4f}")
        print(f"  Size: {original_size:.2f} (matched: {size_matched:.2f})")
        print(f"  Status: {status}")

        # Get trades/fills for this order
        trades = order.get('associate_trades', [])
        if trades:
            print(f"  Fills: {len(trades)}")
            order_total = 0
            for i, trade in enumerate(trades, 1):
                trade_price = float(trade.get('price', 0))
                trade_size = float(trade.get('size', 0))
                trade_cost = trade_price * trade_size

                print(f"    Fill {i}: {trade_size:.2f} @ {trade_price:.4f} = ${trade_cost:.6f}")
                order_total += trade_cost

            print(f"  Total Cost: ${order_total:.6f}")

        print()

    except Exception as e:
        print(f"Order {order_id[:16]}... - Error: {e}")
        print()

print("=" * 80)
print(f"Orders Analyzed: {orders_analyzed}/{len(order_ids)}")
print("=" * 80)
