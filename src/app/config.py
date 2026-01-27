"""
Configuration management for Yang TUI.
"""
import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict

# Defaults
DEFAULT_CONFIG_PATH = Path("yang_config.json")

@dataclass
class RiskSettings:
    max_position_size_usd: float = 100.0
    max_daily_loss_pct: float = 5.0
    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.20
    max_open_positions: int = 3

@dataclass
class AppConfig:
    # Trading
    trading_mode: str = "paper"  # paper, real
    model_path: str = "logs/market_predictor_v1"
    
    # Data / Execution
    data_source: str = "polymarket"  # polymarket, onchain
    execution_strategy: str = "polymarket"  # polymarket, onchain
    
    # Network
    proxy_url: str = "socks5://127.0.0.1:1080"
    rpc_url: str = "http://localhost:8545"
    
    # Keys (usually from env, but can be overridden here if safe)
    polymarket_api_key: Optional[str] = None
    polymarket_secret: Optional[str] = None
    polymarket_passphrase: Optional[str] = None
    private_key: Optional[str] = None
    
    # Risk
    risk: RiskSettings = field(default_factory=RiskSettings)
    
    def save(self, path: Path = DEFAULT_CONFIG_PATH):
        """Save config to JSON."""
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AppConfig":
        """Load from JSON, falling back to defaults/env."""
        if not path.exists():
            return cls.from_env()
            
        with open(path) as f:
            data = json.load(f)
            
        # Handle nested dataclass
        if "risk" in data:
            data["risk"] = RiskSettings(**data["risk"])
            
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load basic defaults + env vars."""
        return cls(
            proxy_url=os.getenv("SOCKS5_PROXY", "socks5://127.0.0.1:1080"),
            rpc_url=os.getenv("POLYGON_RPC", "http://localhost:8545"),
            polymarket_api_key=os.getenv("POLYMARKET_API_KEY"),
            polymarket_secret=os.getenv("POLYMARKET_SECRET"),
            polymarket_passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
            private_key=os.getenv("ETH_PRIVATE_KEY"),
        )
