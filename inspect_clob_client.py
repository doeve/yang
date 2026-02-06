#!/usr/bin/env python3
"""Inspect py-clob-client structure to find HTTP client attribute."""

import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

load_dotenv()

private_key = os.getenv("ETH_PRIVATE_KEY")

print("Creating ClobClient...")
client = ClobClient(
    host="https://clob.polymarket.com",
    key=private_key,
    chain_id=137,
    signature_type=0,
)

print("\n📋 ClobClient attributes:")
for attr in dir(client):
    if not attr.startswith('__'):
        try:
            value = getattr(client, attr)
            value_type = type(value).__name__
            print(f"  {attr}: {value_type}")
        except:
            print(f"  {attr}: <unable to access>")

print("\n🔍 Looking for HTTP client-like attributes:")
for attr in dir(client):
    if any(keyword in attr.lower() for keyword in ['client', 'http', 'session', 'request']):
        try:
            value = getattr(client, attr)
            print(f"  {attr}: {type(value)}")
            if hasattr(value, '__dict__'):
                print(f"    -> {value.__dict__.keys()}")
        except:
            pass
