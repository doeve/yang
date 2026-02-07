#!/usr/bin/env python3
"""
Fetch positions from Data API and force redemption attempts onchain.

Skips all resolution checks and attempts redemption directly.
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
import httpx
from httpx_socks import SyncProxyTransport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.execution import OnchainExecutor

console = Console()


def fetch_positions(wallet_address: str, proxy_url: str) -> List[Dict]:
    """Fetch all positions from Data API."""
    if not wallet_address.startswith("0x") or len(wallet_address) != 42:
        raise ValueError(f"Invalid wallet address: {wallet_address}")

    transport = SyncProxyTransport.from_url(proxy_url)
    client = httpx.Client(transport=transport, verify=False, timeout=30)

    base_url = "https://data-api.polymarket.com/positions"
    all_positions = []
    offset = 0
    limit = 500

    console.print(f"[cyan]Fetching positions for {wallet_address}...[/cyan]")

    try:
        while True:
            params = {
                "user": wallet_address,
                "limit": limit,
                "offset": offset,
                "sizeThreshold": 0.0001,
                "sortBy": "CURRENT",
                "sortDirection": "DESC",
            }

            console.print(f"[dim]Page {offset // limit + 1}...[/dim]")
            response = client.get(base_url, params=params)
            response.raise_for_status()
            positions = response.json()

            if not isinstance(positions, list):
                raise ValueError(f"Unexpected response format")

            num_positions = len(positions)
            all_positions.extend(positions)
            console.print(f"[dim]  Got {num_positions} positions (total: {len(all_positions)})[/dim]")

            if num_positions < limit:
                break

            offset += limit
    finally:
        client.close()

    console.print(f"[green]✓ Fetched {len(all_positions)} positions[/green]\n")
    return all_positions


def display_positions(positions: List[Dict]):
    """Display all positions."""
    if not positions:
        console.print("[yellow]No positions found[/yellow]")
        return

    total_value = sum(float(p.get('currentValue', 0)) for p in positions)

    console.print(f"[bold cyan]{'═'*70}[/bold cyan]")
    console.print(f"[bold cyan]ALL POSITIONS ({len(positions)})[/bold cyan]")
    console.print(f"[bold cyan]Total Value: ${total_value:.2f}[/bold cyan]")
    console.print(f"[bold cyan]{'═'*70}[/bold cyan]\n")

    table = Table()
    table.add_column("#", style="dim", width=3)
    table.add_column("Market", style="cyan", max_width=40)
    table.add_column("Outcome", justify="center", width=6)
    table.add_column("Value", justify="right", style="green", width=10)
    table.add_column("Redeemable", justify="center", width=10)

    for i, pos in enumerate(positions, 1):
        title = pos.get('title', 'Unknown')
        if len(title) > 37:
            title = title[:37] + "..."

        outcome = pos.get('outcome', '?')
        value = float(pos.get('currentValue', 0))
        redeemable = "✓" if pos.get('redeemable', False) else "✗"

        table.add_row(str(i), title, outcome, f"${value:.2f}", redeemable)

    console.print(table)
    console.print()


async def force_redeem_position(
    executor: OnchainExecutor,
    condition_id: str,
    yes_token_id: str,
    no_token_id: str,
    title: str,
    value: float,
):
    """
    Force redemption attempt without any checks.

    Attempts to call redeemPositions directly on the contract.
    """
    if not executor.public_w3 or not executor.account:
        return {"success": False, "error": "Not connected"}

    console.print(f"[cyan]Attempting: {title[:60]}...[/cyan]")
    console.print(f"[dim]  Expected Value: ${value:.2f}[/dim]")
    console.print(f"[dim]  Condition: {condition_id[:32]}...[/dim]")

    try:
        from web3 import Web3

        # CTF contract
        ctf = executor.public_w3.eth.contract(
            address=Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"),
            abi=[
                {
                    "inputs": [
                        {"name": "collateralToken", "type": "address"},
                        {"name": "parentCollectionId", "type": "bytes32"},
                        {"name": "conditionId", "type": "bytes32"},
                        {"name": "indexSets", "type": "uint256[]"},
                    ],
                    "name": "redeemPositions",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function",
                },
            ],
        )

        # USDC contract
        usdc = executor.public_w3.eth.contract(
            address=Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
            abi=[
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function",
                }
            ],
        )

        # Get balance before
        balance_before = usdc.functions.balanceOf(executor.address).call()

        # Prepare redemption parameters
        if not condition_id.startswith("0x"):
            condition_id = "0x" + condition_id
        condition_bytes = bytes.fromhex(condition_id[2:])

        parent_collection_id = bytes(32)
        index_sets = [1, 2]  # YES, NO

        # Build transaction
        nonce = executor.public_w3.eth.get_transaction_count(executor.address, "latest")

        latest_block = executor.public_w3.eth.get_block('latest')
        base_fee = latest_block.get('baseFeePerGas', 30 * 10**9)
        max_priority_fee = executor.public_w3.to_wei(50, 'gwei')
        max_fee = base_fee * 2 + max_priority_fee

        tx = ctf.functions.redeemPositions(
            Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
            parent_collection_id,
            condition_bytes,
            index_sets,
        ).build_transaction({
            "from": executor.address,
            "nonce": nonce,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": max_priority_fee,
            "gas": 300000,
        })

        # Sign and send
        console.print(f"[yellow]  → Sending transaction...[/yellow]")
        signed_tx = executor.account.sign_transaction(tx)
        tx_hash = executor.public_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash.hex()

        console.print(f"[dim]  TX: {tx_hash_hex}[/dim]")
        console.print(f"[yellow]  → Waiting for confirmation...[/yellow]")

        # Wait for receipt
        receipt = executor.public_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] == 1:
            # Calculate redeemed amount
            balance_after = usdc.functions.balanceOf(executor.address).call()
            usdc_redeemed = (balance_after - balance_before) / 1e6

            console.print(f"[green]  ✓ SUCCESS! Redeemed: ${usdc_redeemed:.2f}[/green]")
            return {
                "success": True,
                "tx_hash": tx_hash_hex,
                "usdc_redeemed": usdc_redeemed,
            }
        else:
            console.print(f"[red]  ✗ Transaction reverted[/red]")
            return {
                "success": False,
                "tx_hash": tx_hash_hex,
                "error": "Transaction reverted",
            }

    except Exception as e:
        error_msg = str(e)

        # Parse common errors
        if "execution reverted" in error_msg.lower():
            if "payout is zero" in error_msg.lower():
                console.print(f"[yellow]  ⊘ No payout (position has no value or already redeemed)[/yellow]")
            else:
                console.print(f"[yellow]  ⊘ Reverted: {error_msg[:100]}[/yellow]")
        else:
            console.print(f"[red]  ✗ Error: {error_msg[:100]}[/red]")

        return {"success": False, "error": error_msg}


async def redeem_all_positions(positions: List[Dict], executor: OnchainExecutor):
    """Attempt to redeem all positions."""
    # Filter positions: only redeemable and value > 0
    redeemable_positions = [
        pos for pos in positions
        if pos.get('redeemable', False) and float(pos.get('currentValue', 0)) > 0
    ]

    if not redeemable_positions:
        console.print("[yellow]No redeemable positions with value > 0 found[/yellow]")
        return

    skipped_count = len(positions) - len(redeemable_positions)
    if skipped_count > 0:
        console.print(f"[dim]Skipping {skipped_count} positions (not redeemable or value = 0)[/dim]\n")

    console.print(f"[cyan]{'═'*70}[/cyan]")
    console.print(f"[bold cyan]REDEMPTION ATTEMPTS ({len(redeemable_positions)} positions)[/bold cyan]")
    console.print(f"[cyan]{'═'*70}[/cyan]\n")

    initial_balance = await executor.get_usdc_balance()
    console.print(f"Initial USDC Balance: ${initial_balance:.2f}\n")

    results = {"success": 0, "failed": 0, "total_redeemed": 0.0}

    for i, pos in enumerate(redeemable_positions, 1):
        console.print(f"[bold][{i}/{len(redeemable_positions)}][/bold]")

        result = await force_redeem_position(
            executor,
            condition_id=pos.get('conditionId', ''),
            yes_token_id=pos.get('asset', ''),
            no_token_id=pos.get('oppositeAsset', ''),
            title=pos.get('title', 'Unknown'),
            value=float(pos.get('currentValue', 0)),
        )

        if result.get("success"):
            results["success"] += 1
            results["total_redeemed"] += result.get("usdc_redeemed", 0)
        else:
            results["failed"] += 1

        console.print()

        # Delay between attempts
        if i < len(redeemable_positions):
            await asyncio.sleep(2)

    # Final balance
    final_balance = await executor.get_usdc_balance()

    console.print(f"[cyan]{'═'*70}[/cyan]")
    console.print(f"[bold green]REDEMPTION COMPLETE[/bold green]")
    console.print(f"[cyan]{'═'*70}[/cyan]")
    console.print(f"  Successful: [green]{results['success']}[/green]")
    console.print(f"  Failed: [red]{results['failed']}[/red]")
    console.print(f"  Initial Balance: ${initial_balance:.2f}")
    console.print(f"  Final Balance: [green]${final_balance:.2f}[/green]")
    console.print(f"  Total Redeemed: [bold green]${results['total_redeemed']:+.2f}[/bold green]")
    console.print(f"[cyan]{'═'*70}[/cyan]")


async def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Force redeem Polymarket positions")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Automatically confirm redemption without prompting")
    args = parser.parse_args()

    load_dotenv()

    private_key = os.getenv("ETH_PRIVATE_KEY")
    socks5_proxy = os.getenv("SOCKS5_PROXY")
    local_rpc_url = os.getenv("POLYGON_RPC_URL", "http://localhost:8545")
    public_rpc_url = os.getenv("PUBLIC_RPC_URL", "https://polygon-rpc.com")

    if not private_key or not socks5_proxy:
        console.print("[red]Error: ETH_PRIVATE_KEY and SOCKS5_PROXY must be set[/red]")
        return 1

    if private_key.startswith("0x"):
        private_key = private_key[2:]

    from eth_account import Account
    wallet_address = Account.from_key(private_key).address

    console.print(f"[bold green]Polymarket Force Redeemer[/bold green]")
    console.print(f"[dim]Wallet: {wallet_address}[/dim]\n")

    try:
        # Fetch positions from Data API
        positions = fetch_positions(wallet_address, socks5_proxy)

        if not positions:
            console.print("[yellow]No positions found[/yellow]")
            return 0

        # Display positions
        display_positions(positions)

        # Count redeemable positions
        redeemable_count = sum(
            1 for pos in positions
            if pos.get('redeemable', False) and float(pos.get('currentValue', 0)) > 0
        )

        if redeemable_count == 0:
            console.print("[yellow]No redeemable positions with value > 0 found[/yellow]")
            return 0

        # Ask confirmation (skip if -y flag is set)
        if not args.yes:
            if not Confirm.ask(f"Attempt to redeem {redeemable_count} redeemable positions?"):
                console.print("[yellow]Cancelled[/yellow]")
                return 0
        else:
            console.print(f"[dim]Auto-confirming redemption of {redeemable_count} positions (--yes flag)[/dim]\n")

        # Connect executor
        console.print("\n[cyan]Connecting to blockchain...[/cyan]")
        executor = OnchainExecutor(
            local_rpc_url=local_rpc_url,
            private_key=private_key,
            public_rpc_url=public_rpc_url,
            use_public_rpc=True,
            socks5_proxy=socks5_proxy,
        )

        if not await executor.connect():
            console.print("[red]Failed to connect[/red]")
            return 1

        console.print("[green]✓ Connected[/green]\n")

        # Attempt redemptions
        await redeem_all_positions(positions, executor)

        await executor.disconnect()
        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(1)
