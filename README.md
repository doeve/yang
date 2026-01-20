# Polymarket ML Trader

A production-ready machine learning system for training autonomous trading agents on crypto-based Polymarket prediction markets with 15-minute resolution.

## Features

- **Simulation & Replay Engine**: 1-second granularity historical data replay with strict temporal ordering (no lookahead bias)
- **ML Training**: PPO-based reinforcement learning with LSTM/Transformer temporal feature extractors
- **Closed-Loop Training**: Automatic evaluation and model improvement based on profit, drawdown, risk, and consistency
- **Web Interface**: Real-time training monitoring, visualization, and configuration
- **Production Ready**: Model export to TorchScript/ONNX, live trading adapter interface

## Quick Start

### Installation

```bash
# Clone the repository
cd /home/dave/projects/yang

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Install frontend dependencies
cd web && npm install && cd ..
```

### Collect Data

```bash
# Collect historical data from Polymarket
yang collect --days 30 --output-dir ./data
```

### Train a Model

```bash
# Train with default settings
yang train --timesteps 100000 --name my_model

# Train with custom settings
yang train \
  --timesteps 500000 \
  --n-envs 8 \
  --extractor transformer \
  --learning-rate 0.0001 \
  --name transformer_model
```

### Evaluate

```bash
# Evaluate on test data
yang evaluate ./logs/my_model/final_model --episodes 20
```

### Export for Production

```bash
# Export to TorchScript
yang export ./logs/my_model/final_model ./exported/model.pt --format torchscript

# Export to ONNX
yang export ./logs/my_model/final_model ./exported/model.onnx --format onnx
```

### Web Interface

```bash
# Start the API server
yang serve --port 8000

# In another terminal, start the frontend
cd web && npm run dev
```

Visit http://localhost:3000 to access the web interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │  Training  │  │ Simulation │  │   Models   │  │  Charts   │  │
│  │   Config   │  │   Control  │  │  Manager   │  │           │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │   REST     │  │  WebSocket │  │ Background │  │   Data    │  │
│  │    API     │  │   Stream   │  │   Tasks    │  │  Storage  │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Infrastructure                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │Orchestrator│  │ Callbacks  │  │ Evaluator  │  │   Hyper   │  │
│  │            │  │            │  │            │  │  Search   │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML Models                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │    PPO     │  │    LSTM    │  │Transformer │  │  Reward   │  │
│  │   Agent    │  │  Features  │  │  Features  │  │Calculator │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Engine                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │   Replay   │  │   Market   │  │    Gym     │  │  Feature  │  │
│  │   Engine   │  │  Simulator │  │Environment │  │Preprocess │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
yang/
├── pyproject.toml          # Python dependencies
├── README.md               # This file
├── src/
│   ├── data/               # Data collection & preprocessing
│   │   ├── collector.py    # Polymarket API client
│   │   ├── storage.py      # SQLite + Parquet storage
│   │   └── preprocessor.py # Feature engineering
│   │
│   ├── simulation/         # Replay & simulation
│   │   ├── replay_engine.py    # Historical replay
│   │   ├── market_simulator.py # Order execution
│   │   └── environment.py      # Gymnasium env
│   │
│   ├── models/             # ML models
│   │   ├── features.py     # LSTM/Transformer extractors
│   │   ├── policy.py       # Custom SB3 policy
│   │   ├── agent.py        # Agent wrapper
│   │   └── rewards.py      # Reward functions
│   │
│   ├── training/           # Training infrastructure
│   │   ├── trainer.py      # Orchestrator
│   │   ├── callbacks.py    # Training callbacks
│   │   └── evaluation.py   # Backtesting
│   │
│   ├── api/                # Web API
│   │   └── main.py         # FastAPI application
│   │
│   ├── live/               # Live trading
│   │   ├── adapter.py      # Trading adapters
│   │   └── export.py       # Model export
│   │
│   └── cli.py              # Command-line interface
│
├── web/                    # React frontend
│   ├── src/
│   │   ├── App.tsx         # Main application
│   │   └── index.css       # Styles
│   └── package.json
│
├── data/                   # Data storage
│   ├── raw/                # Raw API responses
│   ├── processed/          # Processed parquet
│   └── models/             # Trained models
│
└── tests/                  # Test suite
```

## Configuration

### Environment Variables

```bash
# Polymarket API key (optional, for authenticated endpoints)
export POLYMARKET_API_KEY=your_key_here
```

### Training Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 100,000 | Total training steps |
| `n_envs` | 4 | Parallel environments |
| `learning_rate` | 3e-4 | PPO learning rate |
| `extractor_type` | lstm | Feature extractor (lstm/transformer/hybrid) |
| `hidden_dim` | 128 | Hidden layer dimension |
| `features_dim` | 64 | Output feature dimension |
| `sequence_length` | 60 | Price history length |

## Design Decisions

### Why PPO over SAC?

1. **Stability**: PPO's clipped objective prevents catastrophic policy updates
2. **On-policy**: Better suits sequential market data
3. **Simplicity**: Easier to debug and tune

### Why LSTM by Default?

1. **Sample Efficiency**: More efficient for short sequences (60-step lookback)
2. **Speed**: Faster training and inference
3. **Proven**: Well-tested for sequential data

### Realism Constraints

All simulations enforce:
- **No lookahead**: Prices revealed sequentially
- **Execution latency**: 1-second minimum
- **Transaction costs**: 0.1% slippage + fees
- **Position limits**: Max 25% of portfolio per market

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/markets` | List markets |
| POST | `/api/training/start` | Start training |
| GET | `/api/training/{id}/status` | Get status |
| POST | `/api/training/{id}/stop` | Stop training |
| GET | `/api/models` | List models |
| POST | `/api/simulation/run` | Run backtest |

### WebSocket

Connect to `/ws` for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Update:", data);
};
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint
ruff check src/

# Type check
mypy src/
```

## License

MIT

python scripts/paper_trade_enhanced.py --backtest \
    --model logs/my_enhanced_model/enhanced_model.zip \
    --num-candles 100 \
    --days-back-min 30 \
    --days-back-max 365

