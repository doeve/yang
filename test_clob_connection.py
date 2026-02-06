#!/usr/bin/env python3
"""Test CLOB client connection with SOCKS proxy support (monkey-patched)."""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("🔍 Testing CLOB Client Connection with SOCKS Proxy...\n")

# Check for required env vars
private_key = os.getenv("ETH_PRIVATE_KEY")
if not private_key:
    print("❌ ETH_PRIVATE_KEY not found in .env")
    sys.exit(1)

print(f"✓ Private key found (length: {len(private_key)})")

# Check for proxy
socks_proxy = os.getenv("SOCKS5_PROXY")
if socks_proxy:
    print(f"✓ SOCKS5 proxy configured: {socks_proxy}")
else:
    print("⚠️  No SOCKS5_PROXY set (will connect directly)")

# Test 1: Check httpx-socks
if socks_proxy:
    try:
        import httpx_socks
        print("✓ httpx-socks available for proxy support")
    except ImportError:
        print("❌ httpx-socks not installed")
        print("   Install with: pip install 'httpx-socks[asyncio]'")
        sys.exit(1)

# Test 2: Monkey-patch httpx BEFORE importing py-clob-client
if socks_proxy:
    try:
        import httpx
        import httpx_socks

        print("\n🔧 Monkey-patching httpx (sync + async)...")

        # Patch AsyncClient
        _original_async_init = httpx.AsyncClient.__init__

        def patched_async_init(self, *args, **kwargs):
            """Patched AsyncClient.__init__ that adds SOCKS proxy."""
            if 'transport' not in kwargs:
                try:
                    kwargs['transport'] = httpx_socks.AsyncProxyTransport.from_url(socks_proxy)
                    kwargs['verify'] = False
                    print(f"   → Added SOCKS proxy to AsyncClient")
                except Exception as e:
                    print(f"   ⚠️  Failed to add proxy to AsyncClient: {e}")
            _original_async_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = patched_async_init

        # Patch Client (SYNC - important for py-clob-client!)
        _original_sync_init = httpx.Client.__init__

        def patched_sync_init(self, *args, **kwargs):
            """Patched Client.__init__ that adds SOCKS proxy."""
            if 'transport' not in kwargs:
                try:
                    kwargs['transport'] = httpx_socks.SyncProxyTransport.from_url(socks_proxy)
                    kwargs['verify'] = False
                    print(f"   → Added SOCKS proxy to sync Client")
                except Exception as e:
                    print(f"   ⚠️  Failed to add proxy to sync Client: {e}")
            _original_sync_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched_sync_init

        print("✓ httpx monkey-patched successfully (sync + async)")

    except Exception as e:
        print(f"❌ Monkey-patch failed: {e}")
        sys.exit(1)

# Test 3: Import py-clob-client (AFTER monkey-patch)
try:
    from py_clob_client.client import ClobClient
    print("✓ py-clob-client imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

async def test_clob_connection():
    """Test CLOB connection with proxy support."""
    try:
        print("\n📡 Initializing CLOB client...")
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=0,
        )
        print("✓ Client initialized")

        # Test API credentials (this makes HTTP requests through the proxy)
        print("\n🔑 Generating API credentials (testing proxied connection)...")
        api_creds = client.create_or_derive_api_creds()
        print(f"✓ API Key: {api_creds.api_key[:8]}...")
        print(f"✓ Secret: {api_creds.api_secret[:8]}...")
        print(f"✓ Passphrase: {api_creds.api_passphrase[:8]}...")

        # Set credentials
        client.set_api_creds(api_creds)
        print("✓ Credentials set successfully")

        print("\n✅ CLOB client connection successful!")
        print(f"   Proxy used: {socks_proxy if socks_proxy else 'Direct connection'}")
        print("\nYou can now use CLOB mode in the trading bot.")
        return True

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")

        # Additional troubleshooting
        print("\n🔧 Troubleshooting:")
        print("1. Verify SOCKS proxy is running:")
        print(f"   nc -zv 127.0.0.1 1080")
        print("2. Test proxy manually:")
        print(f"   curl --socks5 {socks_proxy.replace('socks5://', '')} https://clob.polymarket.com")
        print("3. Check CLOB API status: https://status.polymarket.com")
        print("4. Check firewall settings")

        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        return False

# Run async test
try:
    success = asyncio.run(test_clob_connection())
    sys.exit(0 if success else 1)
except KeyboardInterrupt:
    print("\n\nTest cancelled by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
