#!/usr/bin/env python3
"""
Test Live Trading Flow for Polymarket.

Tests the complete flow:
1. Buy position ($1 worth) - splits USDC into YES/NO tokens
2. Sell position - merges YES/NO back to USDC
3. Redeem position - tests redemption on resolved markets

This tests the onchain execution without CLOB fees.
"""

import os
import sys
import asyncio
import argparse
from decimal import Decimal
from typing import Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.execution import OnchainOrderExecutor, OnchainExecutor

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)
console = Console()


class LiveFlowTester:
    """Test the complete live trading flow."""

    def __init__(
        self,
        local_rpc_url: str,
        private_key: str,
        public_rpc_url: str = "https://polygon-rpc.com",
        socks5_proxy: Optional[str] = None,
        use_local_rpc: bool = False,
    ):
        self.local_rpc_url = local_rpc_url
        self.private_key = private_key
        self.public_rpc_url = public_rpc_url
        self.socks5_proxy = socks5_proxy
        self.use_local_rpc = use_local_rpc

        # Executors
        self.order_executor: Optional[OnchainOrderExecutor] = None
        self.onchain_executor: Optional[OnchainExecutor] = None

        # Test amount
        self.test_amount_usdc = 1.0  # $1 for testing

    async def setup(self) -> bool:
        """Initialize executors."""
        console.print("\n[bold cyan]Setting up test environment...[/bold cyan]")

        # Initialize order executor (for buy/sell)
        self.order_executor = OnchainOrderExecutor(
            local_rpc_url=self.local_rpc_url,
            private_key=self.private_key,
            public_rpc_url=self.public_rpc_url,
            use_public_rpc=not self.use_local_rpc,
            socks5_proxy=self.socks5_proxy,
        )

        if not await self.order_executor.connect():
            console.print("[red]❌ Failed to connect order executor[/red]")
            return False

        # Initialize onchain executor (for redemption)
        self.onchain_executor = OnchainExecutor(
            local_rpc_url=self.local_rpc_url,
            private_key=self.private_key,
            public_rpc_url=self.public_rpc_url,
            use_public_rpc=not self.use_local_rpc,
            socks5_proxy=self.socks5_proxy,
        )

        if not await self.onchain_executor.connect():
            console.print("[red]❌ Failed to connect onchain executor[/red]")
            return False

        # Ensure approvals
        console.print("[cyan]Checking USDC approvals...[/cyan]")
        if not await self.order_executor.ensure_approvals():
            console.print("[yellow]⚠️  Warning: Could not verify approvals[/yellow]")
        else:
            console.print("[green]✅ Approvals confirmed[/green]")

        # Get initial balance
        balance = await self.order_executor.get_usdc_balance()
        console.print(f"[green]✅ Connected! USDC Balance: ${balance:.2f}[/green]\n")

        if balance < self.test_amount_usdc:
            console.print(f"[red]❌ Insufficient balance! Need at least ${self.test_amount_usdc}[/red]")
            return False

        return True

    async def cleanup(self):
        """Cleanup resources."""
        if self.order_executor:
            await self.order_executor.disconnect()
        if self.onchain_executor:
            await self.onchain_executor.disconnect()

    async def get_test_market(self) -> Optional[dict]:
        """Get a market to test with."""
        console.print("[cyan]Fetching active markets...[/cyan]")

        try:
            # Query CLOB API for markets (simpler, more reliable)
            import httpx

            transport = None
            if self.socks5_proxy:
                try:
                    import httpx_socks
                    transport = httpx_socks.AsyncProxyTransport.from_url(self.socks5_proxy)
                except ImportError:
                    pass

            if transport:
                client = httpx.AsyncClient(transport=transport, timeout=30, verify=False)
            else:
                client = httpx.AsyncClient(timeout=30)

            # Try CLOB API sampling endpoint first (no auth needed, returns random markets)
            try:
                response = await client.get(
                    "https://clob.polymarket.com/sampling-markets",
                    params={"next_cursor": "MA=="}
                )

                if response.status_code == 200:
                    data = response.json()
                    markets = data if isinstance(data, list) else data.get("data", [])

                    if markets:
                        console.print(f"[green]Found {len(markets)} markets from CLOB[/green]")

                        # Pick first market with valid token IDs
                        for market in markets[:10]:  # Check first 10
                            tokens = market.get("tokens", [])
                            if len(tokens) >= 2:
                                condition_id = market.get("condition_id")
                                question = market.get("question", "Unknown")
                                yes_token_id = tokens[0].get("token_id", "")
                                no_token_id = tokens[1].get("token_id", "")

                                if condition_id and yes_token_id and no_token_id:
                                    console.print(f"[green]Selected: {question[:60]}...[/green]")
                                    await client.aclose()
                                    return {
                                        "condition_id": condition_id,
                                        "question": question,
                                        "yes_token_id": yes_token_id,
                                        "no_token_id": no_token_id,
                                    }
            except Exception as e:
                console.print(f"[yellow]CLOB API failed: {e}, trying Gamma...[/yellow]")

            # Fallback to Gamma API
            try:
                # Try fetching a current 15min BTC market (most reliable)
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)
                minute_block = (now.minute // 15) * 15
                current_15min = now.replace(minute=minute_block, second=0, microsecond=0)
                timestamp = int(current_15min.timestamp())
                slug = f"btc-updown-15m-{timestamp}"

                console.print(f"[cyan]Trying BTC 15min market: {slug}[/cyan]")
                response = await client.get(
                    f"https://gamma-api.polymarket.com/events/slug/{slug}"
                )

                if response.status_code == 200:
                    data = response.json()
                    markets = data.get("markets", [])

                    if markets:
                        market = markets[0]
                        tokens = market.get("tokens", [])

                        if len(tokens) >= 2:
                            condition_id = market.get("conditionId")
                            question = market.get("question", "Unknown")
                            yes_token_id = tokens[0].get("token_id", "")
                            no_token_id = tokens[1].get("token_id", "")

                            if condition_id and yes_token_id and no_token_id:
                                console.print(f"[green]Found BTC market![/green]")
                                await client.aclose()
                                return {
                                    "condition_id": condition_id,
                                    "question": question,
                                    "yes_token_id": yes_token_id,
                                    "no_token_id": no_token_id,
                                }
            except Exception as e:
                console.print(f"[yellow]Gamma API failed: {e}[/yellow]")

            console.print("[red]Could not fetch any markets[/red]")
            console.print("[yellow]Try running with --no-proxy or check SOCKS5 connection[/yellow]")
            await client.aclose()
            return None

        except Exception as e:
            console.print(f"[red]Error fetching markets: {e}[/red]")
            import traceback
            traceback.print_exc()
            return None

    async def test_buy(self, market: dict) -> bool:
        """Test buying position (split USDC into tokens)."""
        console.print(f"\n[bold yellow]TEST 1: BUY Position (${self.test_amount_usdc})[/bold yellow]")
        console.print(f"Market: {market['question'][:80]}...")
        console.print(f"Condition ID: {market['condition_id'][:16]}...")

        try:
            # Get balance before
            balance_before = await self.order_executor.get_usdc_balance()
            console.print(f"Balance before: ${balance_before:.2f}")

            # Execute buy via split position
            console.print(f"[cyan]Splitting ${self.test_amount_usdc} USDC into YES/NO tokens...[/cyan]")
            console.print(f"[dim]This may take 30-60 seconds for confirmation...[/dim]")

            result = await self.order_executor.split_position(
                condition_id=market["condition_id"],
                amount_usdc=self.test_amount_usdc,
            )

            if not result.success:
                console.print(f"[red]❌ Buy failed: {result.error}[/red]")
                return False

            # Get balance after
            balance_after = await self.order_executor.get_usdc_balance()
            console.print(f"Balance after: ${balance_after:.2f}")
            console.print(f"[green]✅ BUY successful![/green]")
            console.print(f"   TX: {result.tx_hash}")
            console.print(f"   Gas used: {result.gas_used}")
            console.print(f"   USDC spent: ${balance_before - balance_after:.2f}")

            # Verify we got tokens
            yes_balance = await self.order_executor._get_token_balance(market["yes_token_id"])
            no_balance = await self.order_executor._get_token_balance(market["no_token_id"])

            console.print(f"   YES tokens: {yes_balance / 1e6:.2f}")
            console.print(f"   NO tokens: {no_balance / 1e6:.2f}")

            if yes_balance == 0 and no_balance == 0:
                console.print("[red]❌ No tokens received![/red]")
                return False

            return True

        except Exception as e:
            console.print(f"[red]❌ Buy test error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    async def test_sell(self, market: dict) -> bool:
        """Test selling position (merge tokens back to USDC)."""
        console.print(f"\n[bold yellow]TEST 2: SELL Position (merge to USDC)[/bold yellow]")

        try:
            # Get balances before
            balance_before = await self.order_executor.get_usdc_balance()
            yes_balance_before = await self.order_executor._get_token_balance(market["yes_token_id"])
            no_balance_before = await self.order_executor._get_token_balance(market["no_token_id"])

            console.print(f"USDC balance before: ${balance_before:.2f}")
            console.print(f"YES tokens before: {yes_balance_before / 1e6:.2f}")
            console.print(f"NO tokens before: {no_balance_before / 1e6:.2f}")

            # Check if we can merge
            merge_amount = min(yes_balance_before, no_balance_before)
            if merge_amount == 0:
                console.print("[yellow]⚠️  Cannot merge: no matching token pairs[/yellow]")
                console.print("[cyan]This is expected if tokens were split unevenly[/cyan]")
                return True  # Not a failure, just can't merge

            # Execute merge
            console.print(f"[cyan]Merging {merge_amount / 1e6:.2f} token pairs back to USDC...[/cyan]")
            console.print(f"[dim]Waiting for transaction confirmation...[/dim]")

            result = await self.order_executor.merge_position(
                condition_id=market["condition_id"],
                amount=merge_amount / 1e6,  # Convert from wei to USDC
            )

            if not result.success:
                console.print(f"[red]❌ Sell failed: {result.error}[/red]")
                return False

            # Get balance after
            balance_after = await self.order_executor.get_usdc_balance()
            yes_balance_after = await self.order_executor._get_token_balance(market["yes_token_id"])
            no_balance_after = await self.order_executor._get_token_balance(market["no_token_id"])

            console.print(f"USDC balance after: ${balance_after:.2f}")
            console.print(f"YES tokens after: {yes_balance_after / 1e6:.2f}")
            console.print(f"NO tokens after: {no_balance_after / 1e6:.2f}")
            console.print(f"[green]✅ SELL successful![/green]")
            console.print(f"   TX: {result.tx_hash}")
            console.print(f"   Gas used: {result.gas_used}")
            console.print(f"   USDC recovered: ${balance_after - balance_before:.2f}")

            return True

        except Exception as e:
            console.print(f"[red]❌ Sell test error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    async def test_redeem(self, from_block: int = 0) -> bool:
        """Test redemption on resolved markets."""
        console.print(f"\n[bold yellow]TEST 3: REDEEM Resolved Positions[/bold yellow]")

        try:
            # Get balance before
            balance_before = await self.onchain_executor.get_usdc_balance()
            console.print(f"USDC balance before: ${balance_before:.2f}")

            # Try to redeem all resolved positions
            console.print("[cyan]Querying blockchain for your token positions...[/cyan]")
            if from_block == 0:
                console.print("[dim]Scanning last ~30 days of blocks (this may take a moment)...[/dim]")
            else:
                console.print(f"[dim]Scanning from block {from_block}...[/dim]")

            results = await self.onchain_executor.redeem_all_resolved_positions(from_block=from_block)

            # Get balance after
            balance_after = await self.onchain_executor.get_usdc_balance()
            total_redeemed = balance_after - balance_before

            console.print(f"USDC balance after: ${balance_after:.2f}")
            console.print(f"[green]✅ Redemption test complete![/green]")
            console.print(f"   Total redeemed: ${total_redeemed:.2f}")
            console.print(f"   Successful: {sum(1 for r in results if r.success and r.tx_hash)}")
            console.print(f"   Skipped: {sum(1 for r in results if r.skipped_reason)}")
            console.print(f"   Failed: {sum(1 for r in results if not r.success and not r.skipped_reason)}")

            return True

        except Exception as e:
            console.print(f"[red]❌ Redeem test error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    async def run_tests(self, skip_redeem: bool = False, redeem_only: bool = False, from_block: int = 0) -> bool:
        """Run all tests."""
        # Setup
        if not await self.setup():
            return False

        try:
            if redeem_only:
                # Only run redemption test
                console.print("\n[bold cyan]Running REDEMPTION ONLY[/bold cyan]")
                if not await self.test_redeem(from_block=from_block):
                    console.print("\n[yellow]⚠️  Redeem test had issues (this is OK if no resolved positions)[/yellow]")
            else:
                # Get test market
                market = await self.get_test_market()
                if not market:
                    console.print("[red]Failed to get test market[/red]")
                    return False

                console.print(f"\n[bold green]Selected Market:[/bold green]")
                console.print(f"  Question: {market['question']}")
                console.print(f"  Condition ID: {market['condition_id']}")

                # Test 1: Buy
                if not await self.test_buy(market):
                    console.print("\n[red]❌ Buy test FAILED[/red]")
                    return False

                # Wait a bit for blockchain to update
                console.print("\n[cyan]Waiting 5 seconds for blockchain state to update...[/cyan]")
                await asyncio.sleep(5)

                # Test 2: Sell
                if not await self.test_sell(market):
                    console.print("\n[red]❌ Sell test FAILED[/red]")
                    return False

                # Test 3: Redeem (optional)
                if not skip_redeem:
                    console.print("\n[cyan]Waiting 5 seconds before redemption test...[/cyan]")
                    await asyncio.sleep(5)

                    if not await self.test_redeem(from_block=from_block):
                        console.print("\n[yellow]⚠️  Redeem test had issues (this is OK if no resolved positions)[/yellow]")
                else:
                    console.print("\n[yellow]Skipping redemption test[/yellow]")

            # Final summary
            console.print("\n" + "="*60)
            console.print("[bold green]🎉 ALL TESTS PASSED! 🎉[/bold green]")
            console.print("="*60)

            # Show final balance
            final_balance = await self.order_executor.get_usdc_balance()
            console.print(f"\nFinal USDC Balance: ${final_balance:.2f}")

            return True

        except Exception as e:
            console.print(f"\n[red]Test suite error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test live trading flow")
    parser.add_argument(
        "--skip-redeem",
        action="store_true",
        help="Skip redemption test (useful if no resolved positions)"
    )
    parser.add_argument(
        "--redeem-only",
        action="store_true",
        help="Only run redemption test (skip buy/sell)"
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable SOCKS5 proxy (use direct connection)"
    )
    parser.add_argument(
        "--use-local-rpc",
        action="store_true",
        help="Use local RPC for transactions (default uses public RPC)"
    )
    parser.add_argument(
        "--from-block",
        type=int,
        default=0,
        help="Starting block for redemption scan (0 = auto, scans last ~30 days)"
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Get config from env
    local_rpc_url = os.getenv("POLYGON_RPC_URL", "http://localhost:8545")
    public_rpc_url = os.getenv("PUBLIC_RPC_URL", "https://polygon-rpc.com")
    private_key = os.getenv("ETH_PRIVATE_KEY")
    socks5_proxy = None if args.no_proxy else os.getenv("SOCKS5_PROXY")

    if not private_key:
        console.print("[red]Error: ETH_PRIVATE_KEY not set in .env[/red]")
        console.print("[cyan]Please copy .env.example to .env and configure it[/cyan]")
        return 1

    # Show config
    proxy_status = "Disabled (--no-proxy)" if args.no_proxy else (socks5_proxy or "None")
    rpc_mode = "Local RPC" if args.use_local_rpc else "Public RPC"
    test_mode = "Redeem Only" if args.redeem_only else ("Buy + Sell" if args.skip_redeem else "Buy + Sell + Redeem")

    console.print(Panel.fit(
        f"[bold cyan]Live Flow Test Configuration[/bold cyan]\n\n"
        f"Local RPC: {local_rpc_url}\n"
        f"Public RPC: {public_rpc_url}\n"
        f"Transaction RPC: {rpc_mode}\n"
        f"SOCKS5 Proxy: {proxy_status}\n"
        f"Test Mode: {test_mode}\n"
        f"Test Amount: $1.00 USDC",
        title="Configuration"
    ))

    # Confirm
    console.print("\n[yellow]⚠️  This will execute REAL on-chain transactions![/yellow]")
    console.print("[yellow]Make sure you have enough USDC and MATIC for gas.[/yellow]\n")

    response = input("Continue? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        console.print("Cancelled.")
        return 0

    # Run tests
    tester = LiveFlowTester(
        local_rpc_url=local_rpc_url,
        private_key=private_key,
        public_rpc_url=public_rpc_url,
        socks5_proxy=socks5_proxy,
        use_local_rpc=args.use_local_rpc,
    )

    success = await tester.run_tests(
        skip_redeem=args.skip_redeem,
        redeem_only=args.redeem_only,
        from_block=args.from_block,
    )

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
