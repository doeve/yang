#!/usr/bin/env python3
"""
Quick diagnostic script to check if live trading setup is correct.

Verifies:
- Environment variables
- RPC connection
- Wallet balance
- USDC approvals
- Market data access
"""

import os
import sys
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

console = Console()


async def check_env_vars() -> bool:
    """Check required environment variables."""
    console.print("\n[bold cyan]1. Checking Environment Variables[/bold cyan]")

    required_vars = {
        "POLYGON_RPC_URL": "Local Polygon RPC",
        "ETH_PRIVATE_KEY": "Wallet Private Key",
    }

    optional_vars = {
        "PUBLIC_RPC_URL": "Public RPC (for tx broadcast)",
        "SOCKS5_PROXY": "SOCKS5 Proxy",
    }

    table = Table(show_header=True)
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Value Preview")

    all_ok = True

    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value:
            preview = value[:20] + "..." if len(value) > 20 else value
            if "KEY" in var:
                preview = "***" + value[-4:] if len(value) > 4 else "***"
            table.add_row(var, "✅ Set", preview)
        else:
            table.add_row(var, "❌ Missing", "")
            all_ok = False

    for var, desc in optional_vars.items():
        value = os.getenv(var)
        if value:
            preview = value[:30] + "..." if len(value) > 30 else value
            table.add_row(var, "✅ Set", preview)
        else:
            table.add_row(var, "⚠️  Not Set", "Using defaults")

    console.print(table)
    return all_ok


async def check_rpc_connection() -> bool:
    """Check RPC connection."""
    console.print("\n[bold cyan]2. Checking RPC Connection[/bold cyan]")

    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware

        rpc_url = os.getenv("POLYGON_RPC_URL", "http://localhost:8545")
        console.print(f"Connecting to: {rpc_url}")

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if w3.is_connected():
            block_number = w3.eth.block_number
            chain_id = w3.eth.chain_id
            console.print(f"[green]✅ Connected![/green]")
            console.print(f"   Chain ID: {chain_id} (should be 137 for Polygon)")
            console.print(f"   Block Number: {block_number}")
            return chain_id == 137
        else:
            console.print("[red]❌ Not connected[/red]")
            return False

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


async def check_wallet_balance() -> bool:
    """Check wallet balances."""
    console.print("\n[bold cyan]3. Checking Wallet Balances[/bold cyan]")

    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware

        rpc_url = os.getenv("POLYGON_RPC_URL", "http://localhost:8545")
        private_key = os.getenv("ETH_PRIVATE_KEY")

        if not private_key:
            console.print("[red]No private key set[/red]")
            return False

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        from eth_account import Account
        if private_key.startswith("0x"):
            private_key = private_key[2:]
        account = Account.from_key(private_key)
        address = account.address

        console.print(f"Wallet Address: {address}")

        # Check MATIC balance
        matic_balance_wei = w3.eth.get_balance(address)
        matic_balance = matic_balance_wei / 1e18

        # Check USDC balance
        usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        usdc_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function",
            }
        ]

        usdc = w3.eth.contract(address=Web3.to_checksum_address(usdc_address), abi=usdc_abi)
        usdc_balance_wei = usdc.functions.balanceOf(address).call()
        usdc_balance = usdc_balance_wei / 1e6

        table = Table(show_header=True)
        table.add_column("Asset", style="cyan")
        table.add_column("Balance", style="bold")
        table.add_column("Status")

        # MATIC
        matic_status = "✅ OK" if matic_balance >= 0.1 else "⚠️  Low (need for gas)"
        table.add_row("MATIC", f"{matic_balance:.4f}", matic_status)

        # USDC
        usdc_status = "✅ OK" if usdc_balance >= 1.0 else "⚠️  Low (need $1+ for test)"
        table.add_row("USDC", f"${usdc_balance:.2f}", usdc_status)

        console.print(table)

        return matic_balance >= 0.01 and usdc_balance >= 1.0

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def check_approvals() -> bool:
    """Check USDC approvals."""
    console.print("\n[bold cyan]4. Checking USDC Approvals[/bold cyan]")

    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
        from eth_account import Account

        rpc_url = os.getenv("POLYGON_RPC_URL", "http://localhost:8545")
        private_key = os.getenv("ETH_PRIVATE_KEY")

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if private_key.startswith("0x"):
            private_key = private_key[2:]
        account = Account.from_key(private_key)
        address = account.address

        usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        ctf_address = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

        usdc_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"},
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            }
        ]

        usdc = w3.eth.contract(address=Web3.to_checksum_address(usdc_address), abi=usdc_abi)
        allowance = usdc.functions.allowance(address, Web3.to_checksum_address(ctf_address)).call()
        allowance_usdc = allowance / 1e6

        if allowance > 0:
            console.print(f"[green]✅ USDC approved for CTF contract[/green]")
            console.print(f"   Allowance: ${allowance_usdc:.2f}")
            return True
        else:
            console.print(f"[yellow]⚠️  USDC not approved for CTF contract[/yellow]")
            console.print(f"   Run test_live_flow.py to approve automatically")
            return False

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


async def check_market_access(use_proxy: bool = True) -> bool:
    """Check Polymarket API access."""
    console.print("\n[bold cyan]5. Checking Market API Access[/bold cyan]")

    try:
        import httpx

        socks5_proxy = os.getenv("SOCKS5_PROXY") if use_proxy else None

        transport = None
        if socks5_proxy:
            try:
                import httpx_socks
                transport = httpx_socks.AsyncProxyTransport.from_url(socks5_proxy)
                console.print(f"Using SOCKS5 proxy: {socks5_proxy}")
            except ImportError:
                console.print("[yellow]httpx_socks not installed, using direct connection[/yellow]")

        if transport:
            client = httpx.AsyncClient(transport=transport, timeout=30, verify=False)
        else:
            client = httpx.AsyncClient(timeout=30)

        # Try Gamma API
        console.print("Testing Gamma API...")
        response = await client.get("https://gamma-api.polymarket.com/markets", params={"limit": 1})

        if response.status_code == 200:
            markets = response.json()
            console.print(f"[green]✅ Gamma API accessible[/green]")
            console.print(f"   Found {len(markets)} market(s)")

            if markets:
                console.print(f"   Example: {markets[0].get('question', 'N/A')[:60]}...")
        else:
            console.print(f"[red]❌ Gamma API returned {response.status_code}[/red]")
            await client.aclose()
            return False

        await client.aclose()
        return True

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


async def main():
    """Run all checks."""
    import argparse
    parser = argparse.ArgumentParser(description="Check live trading setup")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable SOCKS5 proxy (use direct connection)"
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Live Trading Setup Diagnostic[/bold cyan]\n\n"
        "This script checks if your environment is properly configured for live trading.",
        title="Setup Check"
    ))

    # Load .env
    load_dotenv()

    results = {
        "Environment Variables": await check_env_vars(),
        "RPC Connection": await check_rpc_connection(),
        "Wallet Balances": await check_wallet_balance(),
        "USDC Approvals": await check_approvals(),
        "Market API Access": await check_market_access(use_proxy=not args.no_proxy),
    }

    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]Summary:[/bold]\n")

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        console.print(f"{status} - {check}")

    console.print("="*60)

    if all(results.values()):
        console.print("\n[bold green]🎉 All checks passed! Ready for live trading.[/bold green]")
        console.print("\nNext steps:")
        console.print("1. Run: python test_live_flow.py")
        console.print("2. Or run main script: python -m src.paper_trade_unified_new --live")
        return 0
    else:
        console.print("\n[bold yellow]⚠️  Some checks failed. Please fix the issues above.[/bold yellow]")
        console.print("\nCommon fixes:")
        console.print("- Copy .env.example to .env and configure it")
        console.print("- Make sure Polygon RPC is running or use public RPC")
        console.print("- Fund wallet with USDC and MATIC")
        console.print("- Run test_live_flow.py to set approvals")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
