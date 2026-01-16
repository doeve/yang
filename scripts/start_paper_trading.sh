#!/bin/bash
# Start overnight paper trading test
# This starts both the ML API and the paper trading script

set -e

echo "🚀 Starting Overnight ML Paper Trading Test"
echo "============================================"

# Check if aiohttp and websockets are installed
python3 -c "import aiohttp, websockets" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install aiohttp websockets
}

# Kill any existing API server
pkill -f "uvicorn src.api.prediction" 2>/dev/null || true

# Start API server in background
echo "🔧 Starting ML API server on port 8001..."
uvicorn src.api.prediction:app --host 0.0.0.0 --port 8001 &
API_PID=$!

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 5

# Check if API is running
curl -s http://localhost:8001/health > /dev/null || {
    echo "❌ API failed to start"
    exit 1
}

echo "✅ API is ready"
echo ""
echo "📊 Starting paper trading..."
echo "   Press Ctrl+C to stop"
echo ""

# Run paper trading
python3 scripts/paper_trade_overnight.py

# Cleanup
kill $API_PID 2>/dev/null || true
