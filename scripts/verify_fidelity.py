import asyncio
import httpx
import json
import os
from datetime import datetime
import httpx_socks

SOCKS5_PROXY_URL = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:1080")

async def check_fidelity():
    clob_api = "https://clob.polymarket.com"
    gamma_api = "https://gamma-api.polymarket.com"
    
    print(f"Using proxy: {SOCKS5_PROXY_URL}")
    transport = httpx_socks.AsyncProxyTransport.from_url(SOCKS5_PROXY_URL)
    
    async with httpx.AsyncClient(transport=transport, verify=False) as client:
        # Get multiple markets
        print("Fetching markets...")
        # Try active markets first
        resp = await client.get(f"{gamma_api}/events?limit=20&closed=false")
        if resp.status_code != 200:
            print(f"Failed to fetch events: {resp.status_code}")
            return
            
        events = resp.json()
        if not events:
            print("No events found")
            return
            
        for event in events:
            markets = event.get('markets', [])
            if not markets:
                continue

            market = markets[0]
            clob_token_ids = json.loads(market.get('clobTokenIds', '[]'))
            if not clob_token_ids:
                continue
            
            clob_token_id = clob_token_ids[0]
            print(f"Checking market: {market.get('question', 'Unknown')}")
            
            # Get history with fidelity=1
            resp = await client.get(
                f"{clob_api}/prices-history", 
                params={"market": clob_token_id, "fidelity": 1, "interval": "max"} # Try max interval
            )
            
            if resp.status_code != 200:
                print(f"Failed to fetch history: {resp.status_code}")
                continue
                
            history = resp.json().get('history', [])
            print(f"Got {len(history)} points")
            
            if len(history) > 1:
                t0 = history[0]['t']
                t1 = history[1]['t']
                print(f"T0: {t0}")
                print(f"T1: {t1}")
                print(f"Delta: {t1 - t0} seconds")
                
                # Check average delta
                deltas = []
                for i in range(1, min(50, len(history))):
                    deltas.append(history[i]['t'] - history[i-1]['t'])
                avg_delta = sum(deltas)/len(deltas)
                print(f"Average delta of first {len(deltas)} points: {avg_delta} seconds")
                
                # Check min/max delta
                print(f"Min delta: {min(deltas)}")
                print(f"Max delta: {max(deltas)}")
                return
            else:
                print("Not enough history, trying next...")

if __name__ == "__main__":
    asyncio.run(check_fidelity())
