"""
Configuration management for live trading.

Loads settings from:
1. config.yaml (default values)
2. Environment variables (.env)
3. CLI arguments (highest priority)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from dotenv import load_dotenv


@dataclass
class RiskConfig:
    """Risk control settings."""
    max_daily_loss_pct: float = 5.0
    max_position_size_usdc: float = 100.0
    min_balance_usdc: float = 10.0
    stop_loss_pct: float = 8.0


@dataclass
class ExecutionConfig:
    """Execution settings."""
    use_clob: bool = False  # Use CLOB API for all operations (vs onchain split/merge)
    order_timeout_seconds: int = 30
    poll_interval_seconds: int = 5
    use_public_rpc_for_redeem: bool = True
    public_rpc_url: str = "https://polygon-rpc.com"


@dataclass
class ModelConfig:
    """Model settings."""
    path: str = "./logs/market_predictor_v1"
    min_confidence: float = 0.5
    min_expected_return: float = 0.02


@dataclass
class LoggingConfig:
    """Logging settings."""
    dir: str = "./logs/paper_trade_unified"
    enable_ml_logging: bool = True


@dataclass
class TradingConfig:
    """Main trading configuration."""
    trading_mode: str = "paper"  # "paper" or "live"
    
    # Sub-configs
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment-loaded credentials (not from config.yaml)
    polygon_rpc_url: str = ""
    polygon_ws_url: str = ""
    public_rpc_url: str = ""  # For onchain execution
    eth_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_passphrase: str = ""
    socks5_proxy: str = ""
    
    @property
    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self.trading_mode == "live"
    
    def validate(self) -> None:
        """Validate configuration for the current mode."""
        if self.is_live:
            if not self.eth_private_key:
                raise ValueError("ETH_PRIVATE_KEY required for live trading")
            if not self.polymarket_api_key:
                raise ValueError("POLYMARKET_API_KEY required for live trading")
            if not self.polymarket_api_secret:
                raise ValueError("POLYMARKET_API_SECRET required for live trading")
            if not self.polymarket_passphrase:
                raise ValueError("POLYMARKET_PASSPHRASE required for live trading")


# Singleton config instance
_config: Optional[TradingConfig] = None


def load_config(
    config_path: str = "config.yaml",
    cli_args: Optional[Dict[str, Any]] = None,
    env_path: str = ".env"
) -> TradingConfig:
    """
    Load configuration from file, environment, and CLI args.
    
    Priority (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Config file
    4. Defaults
    """
    global _config
    
    # Load environment variables
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)
    
    # Start with defaults
    config = TradingConfig()
    
    # Load from YAML file
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            yaml_data = yaml.safe_load(f) or {}
        
        # Apply YAML values
        if "trading_mode" in yaml_data:
            config.trading_mode = yaml_data["trading_mode"]
        
        if "risk" in yaml_data:
            risk_data = yaml_data["risk"]
            config.risk = RiskConfig(
                max_daily_loss_pct=risk_data.get("max_daily_loss_pct", 5.0),
                max_position_size_usdc=risk_data.get("max_position_size_usdc", 100.0),
                min_balance_usdc=risk_data.get("min_balance_usdc", 10.0),
                stop_loss_pct=risk_data.get("stop_loss_pct", 8.0),
            )
        
        if "execution" in yaml_data:
            exec_data = yaml_data["execution"]
            config.execution = ExecutionConfig(
                use_clob=exec_data.get("use_clob", False),
                order_timeout_seconds=exec_data.get("order_timeout_seconds", 30),
                poll_interval_seconds=exec_data.get("poll_interval_seconds", 5),
                use_public_rpc_for_redeem=exec_data.get("use_public_rpc_for_redeem", True),
                public_rpc_url=exec_data.get("public_rpc_url", "https://polygon-rpc.com"),
            )
        
        if "model" in yaml_data:
            model_data = yaml_data["model"]
            config.model = ModelConfig(
                path=model_data.get("path", "./logs/market_predictor_v1"),
                min_confidence=model_data.get("min_confidence", 0.5),
                min_expected_return=model_data.get("min_expected_return", 0.02),
            )
        
        if "logging" in yaml_data:
            log_data = yaml_data["logging"]
            config.logging = LoggingConfig(
                dir=log_data.get("dir", "./logs/paper_trade_unified"),
                enable_ml_logging=log_data.get("enable_ml_logging", True),
            )
    
    # Load from environment
    config.polygon_rpc_url = os.getenv("POLYGON_RPC_URL", os.getenv("POLYGON_RPC", "http://localhost:8545"))
    config.polygon_ws_url = os.getenv("POLYGON_WS_URL", "ws://localhost:8546")
    config.public_rpc_url = os.getenv("PUBLIC_RPC_URL", config.execution.public_rpc_url)  # From env or config.yaml
    config.eth_private_key = os.getenv("ETH_PRIVATE_KEY", "")
    config.polymarket_api_key = os.getenv("POLYMARKET_API_KEY", "")
    config.polymarket_api_secret = os.getenv("POLYMARKET_API_SECRET", os.getenv("POLYMARKET_SECRET", ""))
    config.polymarket_passphrase = os.getenv("POLYMARKET_PASSPHRASE", "")
    config.socks5_proxy = os.getenv("SOCKS5_PROXY", "socks5://127.0.0.1:1080")
    
    # Apply CLI overrides
    if cli_args:
        if cli_args.get("live"):
            config.trading_mode = "live"
        if cli_args.get("clob"):
            config.execution.use_clob = True
        if cli_args.get("model"):
            config.model.path = cli_args["model"]
        if cli_args.get("min_confidence") is not None:
            config.model.min_confidence = cli_args["min_confidence"]
        if cli_args.get("min_return") is not None:
            config.model.min_expected_return = cli_args["min_return"]
        if cli_args.get("log_dir"):
            config.logging.dir = cli_args["log_dir"]
        if cli_args.get("no_ml_log"):
            config.logging.enable_ml_logging = False
        if cli_args.get("balance"):
            # This is handled by UnifiedPaperTradeConfig, but we track it here too
            pass
    
    _config = config
    return config


def get_config() -> TradingConfig:
    """Get the loaded configuration (singleton)."""
    if _config is None:
        return load_config()
    return _config
