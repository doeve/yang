#!/usr/bin/env python3
"""
Fetch YES/NO token prices from local Bor node (Polygon).

Uses OrderFilled events from Polymarket CTF Exchange contract.

Sources:
- https://yzc.me/x01Crypto/decoding-polymarket
- https://polygonscan.com/address/0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e
"""

import json
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from web3 import Web3
from eth_abi import decode

# Contract addresses
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Binary markets
NEGRISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Multi-outcome
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Event signatures
# OrderFilled on NegRisk: OrderFilled(bytes32 indexed orderHash, address indexed maker, address indexed taker, uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"

# FeeCharged on CTF Exchange (less useful for prices)
FEE_CHARGED_TOPIC = "0xacffcc86834d0f1a64b0d5a675798deed6ff0bcfc2231edd3480e7288dba7ff4"


def connect_to_node(rpc_url: str = "http://localhost:8545") -> Web3:
    """Connect to local Bor node."""
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to {rpc_url}")

    chain_id = w3.eth.chain_id
    block_num = w3.eth.block_number
    print(f"Connected to chain {chain_id} at block {block_num}")
    return w3


def normalize_token_id(token_id: str) -> str:
    """Normalize token ID to hex format for comparison."""
    if token_id.startswith("0x"):
        return token_id.lower()
    else:
        # Assume decimal string
        try:
            return hex(int(token_id)).lower()
        except:
            return token_id.lower()


def get_order_filled_events(
    w3: Web3,
    contract_address: str,
    from_block: int,
    to_block: int,
    token_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch OrderFilled events from CTF Exchange.

    Returns list of events with price info.
    """
    events = []

    # Normalize token_id for comparison
    if token_id:
        token_id = normalize_token_id(token_id)

    # Build filter
    filter_params = {
        "fromBlock": from_block,
        "toBlock": to_block,
        "address": Web3.to_checksum_address(contract_address),
        "topics": [ORDER_FILLED_TOPIC],
    }

    try:
        logs = w3.eth.get_logs(filter_params)
        print(f"Found {len(logs)} OrderFilled events")

        for log in logs:
            event = decode_order_filled(log)
            if event:
                # Filter by token_id if specified
                if token_id:
                    event_token = event.get("token_id", "").lower()
                    if event_token != token_id:
                        continue
                events.append(event)

    except Exception as e:
        print(f"Error fetching logs: {e}")

    return events


def decode_order_filled(log: Dict) -> Optional[Dict[str, Any]]:
    """Decode OrderFilled event log."""
    try:
        # Indexed params from topics (4 topics for NegRisk OrderFilled)
        if len(log["topics"]) < 4:
            return None

        order_hash = log["topics"][1].hex() if hasattr(log["topics"][1], 'hex') else log["topics"][1]
        maker = Web3.to_checksum_address("0x" + (log["topics"][2].hex() if hasattr(log["topics"][2], 'hex') else log["topics"][2])[-40:])
        taker = Web3.to_checksum_address("0x" + (log["topics"][3].hex() if hasattr(log["topics"][3], 'hex') else log["topics"][3])[-40:])

        # Non-indexed params from data
        # (uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)
        data_hex = log["data"].hex() if hasattr(log["data"], 'hex') else log["data"]
        if data_hex.startswith("0x"):
            data_hex = data_hex[2:]
        data = bytes.fromhex(data_hex)

        if len(data) < 160:
            return None

        decoded = decode(
            ["uint256", "uint256", "uint256", "uint256", "uint256"],
            data
        )

        maker_asset_id_int = decoded[0]
        taker_asset_id_int = decoded[1]
        maker_asset_id = hex(maker_asset_id_int)
        taker_asset_id = hex(taker_asset_id_int)
        maker_amount = decoded[2]
        taker_amount = decoded[3]
        fee = decoded[4]

        # Calculate price
        # Asset ID 0 = USDC (collateral)
        # Both USDC and CTF tokens have 6 decimals
        price = None
        token_id = None
        side = None

        if maker_asset_id_int == 0:
            # Maker gave USDC, got tokens (BUY)
            usdc = maker_amount / 1e6
            tokens = taker_amount / 1e6
            if tokens > 0:
                price = usdc / tokens
            token_id = taker_asset_id
            side = "buy"
        elif taker_asset_id_int == 0:
            # Taker gave USDC for maker's tokens (SELL)
            usdc = taker_amount / 1e6
            tokens = maker_amount / 1e6
            if tokens > 0:
                price = usdc / tokens
            token_id = maker_asset_id
            side = "sell"

        return {
            "block_number": log["blockNumber"],
            "tx_hash": log["transactionHash"].hex(),
            "order_hash": order_hash,
            "maker": maker,
            "taker": taker,
            "maker_asset_id": maker_asset_id,
            "taker_asset_id": taker_asset_id,
            "maker_amount": maker_amount,
            "taker_amount": taker_amount,
            "fee": fee,
            "price": price,
            "token_id": token_id,
            "side": side,
        }

    except Exception as e:
        print(f"Error decoding log: {e}")
        return None


def get_recent_prices(
    w3: Web3,
    token_id: str,
    blocks_back: int = 1000,
) -> List[Dict[str, Any]]:
    """Get recent prices for a specific token."""
    current_block = w3.eth.block_number
    from_block = max(0, current_block - blocks_back)

    print(f"Fetching prices for token {token_id}")
    print(f"Blocks {from_block} to {current_block}")

    # Check both exchanges
    events = []

    for exchange in [CTF_EXCHANGE, NEGRISK_CTF_EXCHANGE]:
        print(f"Checking {exchange}...")
        exchange_events = get_order_filled_events(
            w3, exchange, from_block, current_block, token_id
        )
        events.extend(exchange_events)

    # Sort by block
    events.sort(key=lambda x: x["block_number"])

    return events


def find_btc_15min_markets(w3: Web3, blocks_back: int = 500) -> List[str]:
    """
    Find active BTC 15-min market token IDs by looking at recent trades.
    Uses NegRisk exchange which has OrderFilled events.
    """
    current_block = w3.eth.block_number
    from_block = max(0, current_block - blocks_back)

    print(f"Scanning for active markets in last {blocks_back} blocks...")
    print(f"Querying NegRisk exchange (OrderFilled events)...")

    token_ids = set()

    # Only query NegRisk exchange - that's where OrderFilled events are
    events = get_order_filled_events(w3, NEGRISK_CTF_EXCHANGE, from_block, current_block)
    for event in events:
        if event and event.get("token_id"):
            token_ids.add(event["token_id"])

    print(f"Found {len(token_ids)} unique tokens traded")
    return list(token_ids)


def collect_price_history(
    w3: Web3,
    token_id: str,
    blocks_back: int = 10000,
    batch_size: int = 500,
    add_timestamps: bool = False,
) -> List[Dict[str, Any]]:
    """
    Collect historical prices for a token from on-chain events.
    Returns list of price events with timestamps.
    """
    current_block = w3.eth.block_number
    all_events = []

    print(f"Collecting price history for {token_id[:20]}...")
    print(f"Scanning {blocks_back} blocks in batches of {batch_size}")

    for start in range(current_block - blocks_back, current_block, batch_size):
        end = min(start + batch_size, current_block)

        for exchange in [NEGRISK_CTF_EXCHANGE, CTF_EXCHANGE]:
            events = get_order_filled_events(w3, exchange, start, end, token_id)
            all_events.extend(events)

        print(f"  Blocks {start}-{end}: {len(all_events)} events so far")

    # Add timestamps if requested (slower)
    if add_timestamps:
        print("Fetching block timestamps...")
        block_cache = {}
        for event in all_events:
            bn = event["block_number"]
            if bn not in block_cache:
                try:
                    block = w3.eth.get_block(bn)
                    block_cache[bn] = block["timestamp"]
                except:
                    block_cache[bn] = None
            event["timestamp"] = block_cache.get(bn)
    else:
        # Estimate timestamps from block numbers (Polygon ~2 sec/block)
        current_time = int(datetime.now().timestamp())
        for event in all_events:
            blocks_ago = current_block - event["block_number"]
            event["timestamp"] = current_time - (blocks_ago * 2)  # ~2 sec/block

    return sorted(all_events, key=lambda x: (x.get("timestamp") or 0, x["block_number"]))


def get_btc_15min_token_ids(w3: Web3, candle_timestamp: int) -> Optional[Dict[str, str]]:
    """
    Get YES/NO token IDs for a BTC 15-min candle.
    Uses the Polymarket gamma API to get market info.
    """
    import httpx

    slug = f"btc-updown-15m-{candle_timestamp}"
    url = f"https://gamma-api.polymarket.com/events/slug/{slug}"

    try:
        # Use SOCKS proxy if available
        try:
            import httpx_socks
            transport = httpx_socks.SyncProxyTransport.from_url("socks5://127.0.0.1:1080")
            client = httpx.Client(transport=transport, timeout=30, verify=False)
        except ImportError:
            client = httpx.Client(timeout=30)

        response = client.get(url)
        if response.status_code == 200:
            data = response.json()
            markets = data.get("markets", [])
            if markets:
                market = markets[0]
                clob_tokens = market.get("clobTokenIds", "[]")
                if isinstance(clob_tokens, str):
                    import json
                    clob_tokens = json.loads(clob_tokens)

                if len(clob_tokens) >= 2:
                    return {
                        "yes_token_id": clob_tokens[0],
                        "no_token_id": clob_tokens[1],
                        "slug": slug,
                    }
        client.close()
    except Exception as e:
        print(f"Error fetching market info: {e}")

    return None


def save_price_data(events: List[Dict[str, Any]], output_path: str):
    """Save price events to JSON file."""
    import json
    from pathlib import Path

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = []
    for e in events:
        data.append({
            "timestamp": e.get("timestamp"),
            "block_number": e["block_number"],
            "price": e.get("price"),
            "side": e.get("side"),
            "token_id": e.get("token_id"),
            "maker_amount": e.get("maker_amount"),
            "taker_amount": e.get("taker_amount"),
        })

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} events to {output}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket prices from local node")
    parser.add_argument("--rpc", default="http://localhost:8545", help="RPC URL")
    parser.add_argument("--token-id", help="Specific token ID to query")
    parser.add_argument("--blocks", type=int, default=1000, help="Blocks to look back")
    parser.add_argument("--scan", action="store_true", help="Scan for active markets")
    parser.add_argument("--collect", action="store_true", help="Collect full price history")
    parser.add_argument("--output", type=str, help="Output file for collected data")
    args = parser.parse_args()

    w3 = connect_to_node(args.rpc)

    if args.scan:
        tokens = find_btc_15min_markets(w3, args.blocks)
        print("\nActive token IDs:")
        for t in tokens[:20]:  # Show first 20
            print(f"  {t}")
        return

    if args.collect and args.token_id:
        events = collect_price_history(w3, args.token_id, args.blocks)
        print(f"\nCollected {len(events)} price events")

        if events:
            prices = [e["price"] for e in events if e.get("price")]
            if prices:
                print(f"Price range: ${min(prices):.4f} - ${max(prices):.4f}")

            if args.output:
                save_price_data(events, args.output)
        return

    # BTC 15-min market collection
    if args.collect and not args.token_id:
        # Get current 15-min candle timestamp
        now = int(datetime.now(timezone.utc).timestamp())
        candle_ts = (now // 900) * 900

        print(f"Looking up current BTC 15-min market (candle {candle_ts})...")
        market_info = get_btc_15min_token_ids(w3, candle_ts)

        if market_info:
            print(f"Found: {market_info['slug']}")
            print(f"  YES token: {market_info['yes_token_id'][:30]}...")
            print(f"  NO token: {market_info['no_token_id'][:30]}...")

            # Collect prices for both tokens
            yes_events = collect_price_history(w3, market_info["yes_token_id"], args.blocks)
            no_events = collect_price_history(w3, market_info["no_token_id"], args.blocks)

            print(f"\nYES prices: {len(yes_events)} events")
            print(f"NO prices: {len(no_events)} events")

            if args.output:
                combined = {
                    "candle_ts": candle_ts,
                    "slug": market_info["slug"],
                    "yes_token_id": market_info["yes_token_id"],
                    "no_token_id": market_info["no_token_id"],
                    "yes_prices": yes_events,
                    "no_prices": no_events,
                }
                import json
                with open(args.output, "w") as f:
                    json.dump(combined, f, indent=2, default=str)
                print(f"Saved to {args.output}")
        else:
            print("Could not find current BTC 15-min market")
        return

    if args.token_id:
        events = get_recent_prices(w3, args.token_id, args.blocks)

        print(f"\nFound {len(events)} price events:")
        for e in events[-20:]:  # Show last 20
            price_str = f"${e['price']:.4f}" if e['price'] else "N/A"
            print(f"  Block {e['block_number']}: {e['side']} @ {price_str}")
    else:
        # Show recent activity
        current_block = w3.eth.block_number
        from_block = max(0, current_block - args.blocks)

        print(f"\nRecent OrderFilled events:")
        for exchange in [CTF_EXCHANGE, NEGRISK_CTF_EXCHANGE]:
            print(f"\n{exchange}:")
            events = get_order_filled_events(w3, exchange, from_block, current_block)
            for e in events[-10:]:
                price_str = f"${e['price']:.4f}" if e['price'] else "N/A"
                print(f"  Block {e['block_number']}: token={e['token_id'][:18]}... {e['side']} @ {price_str}")


if __name__ == "__main__":
    main()
