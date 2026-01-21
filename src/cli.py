"""
Command-line interface for the ML trading system.

Provides commands for:
- Data collection
- Training
- Evaluation
- Model export
- Live trading
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="yang",
    help="ML Trading System for Polymarket Prediction Markets",
    add_completion=False,
)

console = Console()


@app.command()
def collect(
    output_dir: str = typer.Option("./data", help="Output directory for data"),
    days: int = typer.Option(7, help="Days of historical data to collect"),
    symbols: str = typer.Option("BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated crypto symbols"),
):
    """Collect historical crypto data from Binance (no geo-restrictions)."""
    import asyncio
    from .data.binance_collector import BinanceCollector
    from pathlib import Path
    
    symbol_list = [s.strip() for s in symbols.split(",")]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    async def _collect():
        async with BinanceCollector() as collector:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for i, symbol in enumerate(symbol_list):
                    task = progress.add_task(f"Fetching {symbol}...", total=None)
                    
                    try:
                        df = await collector.fetch_historical_data(symbol, days, "1m")
                        if not df.empty:
                            normalized = collector.normalize_to_prediction_market_format(df, symbol)
                            output_file = output_path / f"{symbol.lower()}_prices.parquet"
                            normalized.to_parquet(output_file)
                            progress.update(task, description=f"✓ {symbol}: {len(normalized):,} records")
                        else:
                            progress.update(task, description=f"✗ {symbol}: No data")
                    except Exception as e:
                        progress.update(task, description=f"✗ {symbol}: {e}")
    
    console.print(f"[blue]Collecting {days} days of data for: {', '.join(symbol_list)}[/blue]")
    asyncio.run(_collect())
    console.print("[green]Data collection complete![/green]")


@app.command()
def train(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs", help="Output directory for models"),
    name: str = typer.Option("training_run", help="Training run name"),
    timesteps: int = typer.Option(100_000, help="Total training timesteps"),
    n_envs: int = typer.Option(4, help="Number of parallel environments"),
    extractor: str = typer.Option("lstm", help="Feature extractor type (lstm/transformer/hybrid)"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
):
    """Train a new model."""
    from .data.storage import DataStorage
    from .models.agent import AgentConfig
    from .training.trainer import TrainingOrchestrator, TrainingConfig
    
    storage = DataStorage(data_dir)
    
    agent_config = AgentConfig(
        extractor_type=extractor,
        learning_rate=learning_rate,
    )
    
    training_config = TrainingConfig(
        total_timesteps=timesteps,
        n_envs=n_envs,
        agent_config=agent_config,
        experiment_name=name,
        log_dir=output_dir,
    )
    
    console.print(f"[blue]Starting training: {name}[/blue]")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print(f"  Extractor: {extractor}")
    console.print()
    
    orchestrator = TrainingOrchestrator(storage, training_config)
    result = orchestrator.train()
    
    console.print()
    console.print("[green]Training complete![/green]")
    console.print(f"  Model saved to: {result['model_path']}")
    console.print(f"  Best reward: {result['best_reward']:.2f}")


@app.command("train-candle")
def train_candle(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs", help="Output directory for models"),
    name: str = typer.Option("candle_prediction", help="Training run name"),
    timesteps: int = typer.Option(500_000, help="Total training timesteps"),
    n_envs: int = typer.Option(24, help="Number of parallel environments"),
    candle_minutes: int = typer.Option(15, help="Candle duration in minutes"),
):
    """Train a binary candle direction prediction model.
    
    This trains a model to predict whether a 15-minute candle will close
    higher or lower than it opened - matching the d3v betting flow exactly.
    """
    import pandas as pd
    from .training.candle_trainer import CandleTrainer, CandleTrainingConfig
    
    # Load price data
    data_path = Path(data_dir) / "btcusdt_100ms.parquet"
    if not data_path.exists():
        # Try 1-second data
        data_path = Path(data_dir) / "btcusdt_1s_30days.parquet"
    
    if not data_path.exists():
        console.print("[red]No price data found. Run 'collect' first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading data from {data_path}[/blue]")
    price_data = pd.read_parquet(data_path)
    
    # Rename columns if needed
    if "close" in price_data.columns and "price" not in price_data.columns:
        price_data = price_data.rename(columns={"close": "price"})
    
    config = CandleTrainingConfig(
        total_timesteps=timesteps,
        n_envs=n_envs,
        candle_minutes=candle_minutes,
        log_dir=output_dir,
        experiment_name=name,
    )
    
    console.print(f"[blue]Starting candle prediction training: {name}[/blue]")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print(f"  Candle duration: {candle_minutes} minutes")
    console.print(f"  Data points: {len(price_data):,}")
    console.print()
    
    trainer = CandleTrainer(price_data, config)
    result = trainer.train()
    
    console.print()
    console.print("[green]Training complete![/green]")
    console.print(f"  Model saved to: {result['model_path']}")
    console.print(f"  Final accuracy: {result['final_accuracy']:.1%}")
    console.print(f"  Avg bet timing: {result['avg_bet_timing']:.1%} into candle")
    console.print(f"  Total predictions: {result['total_predictions']:,}")


@app.command("train-advanced")
def train_advanced(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs", help="Output directory"),
    name: str = typer.Option("advanced_candle", help="Training run name"),
    timesteps: int = typer.Option(500_000, help="Total timesteps"),
    n_envs: int = typer.Option(8, help="Number of environments"),
    initial_balance: float = typer.Option(1000.0, help="Starting balance"),
):
    """Train with position sizing (learns BOTH direction AND bet size).
    
    Uses SAC algorithm with continuous action space.
    The model learns Kelly-like position sizing based on confidence.
    """
    import pandas as pd
    from .training.advanced_trainer import AdvancedTrainer, AdvancedTrainingConfig
    
    # Load data
    data_path = Path(data_dir) / "btcusdt_100ms.parquet"
    if not data_path.exists():
        data_path = Path(data_dir) / "btcusdt_1s_30days.parquet"
    
    if not data_path.exists():
        console.print("[red]No price data found.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading data from {data_path}[/blue]")
    price_data = pd.read_parquet(data_path)
    
    if "close" in price_data.columns and "price" not in price_data.columns:
        price_data = price_data.rename(columns={"close": "price"})
    
    config = AdvancedTrainingConfig(
        total_timesteps=timesteps,
        n_envs=n_envs,
        initial_balance=initial_balance,
        log_dir=output_dir,
        experiment_name=name,
    )
    
    console.print(f"[blue]Starting advanced training: {name}[/blue]")
    console.print(f"  Algorithm: SAC (continuous actions)")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Initial balance: ${initial_balance:,.0f}")
    console.print()
    
    trainer = AdvancedTrainer(price_data, config)
    result = trainer.train()
    
    console.print()
    console.print("[green]Training complete![/green]")
    console.print(f"  Model saved to: {result['model_path']}")
    console.print(f"  Final accuracy: {result['final_accuracy']:.1%}")
    console.print(f"  Avg final balance: ${result['avg_final_balance']:,.2f}")
    console.print(f"  Avg position size: {result['avg_position_size']:.1%}")
    console.print(f"  Total trades: {result['total_trades']:,}")


@app.command("collect-forex")
def collect_forex(
    data_dir: str = typer.Option("./data", help="Data directory"),
    interval: str = typer.Option("1h", help="Data interval (1m, 5m, 1h)"),
):
    """Collect forex data (DXY, EUR/USD) for correlation with BTC."""
    import asyncio
    from .data.forex_collector import ForexCollector
    
    console.print("[blue]Collecting forex data...[/blue]")
    
    collector = ForexCollector(data_dir)
    results = asyncio.run(collector.collect_all(interval))
    
    console.print()
    for symbol, df in results.items():
        console.print(f"[green]✓[/green] {symbol}: {len(df)} rows")
        console.print(f"    Range: {df['timestamp'].min()} to {df['timestamp'].max()}")


@app.command("train-multi")
def train_multi(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs", help="Output directory"),
    name: str = typer.Option("multi_asset", help="Training run name"),
    timesteps: int = typer.Option(2_000_000, help="Total timesteps"),
    n_envs: int = typer.Option(8, help="Number of environments"),
    recurrent: bool = typer.Option(True, help="Use RecurrentSAC (LSTM)"),
):
    """Train with BTC + forex correlation (DXY, EUR/USD).
    
    Run 'collect-forex' first to download forex data.
    Uses asymmetric reward for proper position sizing.
    """
    import pandas as pd
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
    from .simulation.multi_asset_env import MultiAssetEnv, MultiAssetConfig, make_multi_asset_vec_env
    from .training.advanced_trainer import BalanceCallback
    from stable_baselines3.common.callbacks import CallbackList
    
    # Try importing RecurrentPPO
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None
        if recurrent:
            console.print("[yellow]sb3-contrib not found. Falling back to standard SAC.[/yellow]")
            recurrent = False
    
    data_path = Path(data_dir)
    
    # Load BTC data
    btc_path = data_path / "btcusdt_100ms.parquet"
    if not btc_path.exists():
        btc_path = data_path / "btcusdt_1s_30days.parquet"
    
    if not btc_path.exists():
        console.print("[red]No BTC data found. Run data collection first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading BTC data from {btc_path}[/blue]")
    btc_data = pd.read_parquet(btc_path)
    if "close" in btc_data.columns and "price" not in btc_data.columns:
        btc_data = btc_data.rename(columns={"close": "price"})
    
    # Load forex data if available
    dxy_data = None
    eurusd_data = None
    
    dxy_path = data_path / "dxy_1h.parquet"
    if dxy_path.exists():
        console.print(f"[blue]Loading DXY data from {dxy_path}[/blue]")
        dxy_data = pd.read_parquet(dxy_path)
    
    eurusd_path = data_path / "eurusd_1h.parquet"
    if eurusd_path.exists():
        console.print(f"[blue]Loading EUR/USD data from {eurusd_path}[/blue]")
        eurusd_data = pd.read_parquet(eurusd_path)
    
    if dxy_data is None and eurusd_data is None:
        console.print("[yellow]No forex data found. Training with BTC only.[/yellow]")
        console.print("[yellow]Run 'collect-forex' to add correlation data.[/yellow]")
    
    # Create environment
    config = MultiAssetConfig()
    
    console.print()
    console.print(f"[blue]Starting multi-asset training: {name}[/blue]")
    console.print(f"  Assets: BTC" + (" + DXY" if dxy_data is not None else "") + (" + EUR/USD" if eurusd_data is not None else ""))
    console.print(f"  Model: {'RecurrentPPO (LSTM)' if recurrent else 'SAC (MLP)'}")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Reward: Asymmetric (squared penalty for wrong bets)")
    console.print()
    
    train_env = make_multi_asset_vec_env(btc_data, dxy_data, eurusd_data, n_envs, config)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    log_dir = Path(output_dir) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    from src.training.advanced_trainer import BalanceCallback
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    
    balance_callback = BalanceCallback(eval_freq=25_000 // n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(Path(log_dir) / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,
    )
    
    callbacks = CallbackList([balance_callback, checkpoint_callback])
    
    if recurrent and RecurrentPPO:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={"shared_lstm": True, "enable_critic_lstm": False},
            verbose=1,
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            verbose=1,
        )
    
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    model_path = log_dir / "multi_asset_model"
    model.save(str(model_path))
    train_env.save(str(log_dir / "vec_normalize.pkl"))
    
    console.print()
    console.print("[green]Training complete![/green]")
    console.print(f"  Model saved to: {model_path}")
    if balance_callback.total_trades > 0:
        console.print(f"  Final accuracy: {balance_callback.correct_trades / balance_callback.total_trades:.1%}")
        console.print(f"  Total trades: {balance_callback.total_trades:,}")


@app.command("train-enhanced")
def train_enhanced(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs", help="Output directory"),
    name: str = typer.Option("enhanced_multi_asset", help="Training run name"),
    timesteps: int = typer.Option(2_000_000, help="Total timesteps"),
    n_envs: int = typer.Option(8, help="Number of environments"),
    recurrent: bool = typer.Option(True, help="Use RecurrentPPO (LSTM)"),
):
    """Train with enhanced features (order flow, multi-timeframe, skip action).
    
    This uses the EnhancedMultiAssetEnv which includes:
    - Order flow features (volume delta, trade imbalance, large trades, VWAP)
    - Multi-timeframe features (1h/4h price and volume momentum)
    - Skip action (model can choose not to bet on uncertain candles)
    - Timing rewards (bonus for waiting until late in candle)
    - Risk-adjusted rewards (Kelly-inspired position sizing)
    - Increased history window (500 vs 300)
    """
    import pandas as pd
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
    from .simulation.enhanced_multi_asset_env import (
        EnhancedMultiAssetEnv,
        EnhancedMultiAssetConfig,
        make_enhanced_multi_asset_vec_env,
    )
    from .training.advanced_trainer import BalanceCallback
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None
        if recurrent:
            console.print("[yellow]sb3-contrib not found. Falling back to SAC.[/yellow]")
            recurrent = False
    
    data_path = Path(data_dir)
    
    btc_path = data_path / "btcusdt_100ms.parquet"
    if not btc_path.exists():
        btc_path = data_path / "btcusdt_1s_30days.parquet"
    
    if not btc_path.exists():
        console.print("[red]No BTC data found.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading BTC data from {btc_path}[/blue]")
    btc_data = pd.read_parquet(btc_path)
    if "close" in btc_data.columns and "price" not in btc_data.columns:
        btc_data = btc_data.rename(columns={"close": "price"})
    
    dxy_data = None
    eurusd_data = None
    
    if (data_path / "dxy_1h.parquet").exists():
        console.print("[blue]Loading DXY data[/blue]")
        dxy_data = pd.read_parquet(data_path / "dxy_1h.parquet")
    
    if (data_path / "eurusd_1h.parquet").exists():
        console.print("[blue]Loading EUR/USD data[/blue]")
        eurusd_data = pd.read_parquet(data_path / "eurusd_1h.parquet")
    
    config = EnhancedMultiAssetConfig(
        include_order_flow=True,
        include_multi_timeframe=True,
        enable_skip=True,
        use_risk_adjusted_rewards=True,
    )
    
    console.print()
    console.print(f"[blue]Starting enhanced training: {name}[/blue]")
    console.print(f"  Model: {'RecurrentPPO (LSTM)' if recurrent else 'SAC (MLP)'}")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Features: Order Flow + Multi-TF + Skip + Risk-Adjusted")
    console.print()
    
    log_dir = Path(output_dir) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    train_env = make_enhanced_multi_asset_vec_env(btc_data, dxy_data, eurusd_data, n_envs, config)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    console.print(f"[dim]Observation space: {train_env.observation_space.shape}[/dim]")
    
    balance_callback = BalanceCallback(eval_freq=25_000 // n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,
    )
    callbacks = CallbackList([balance_callback, checkpoint_callback])
    
    if recurrent and RecurrentPPO:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=5e-5,           # Reduced from 3e-4 for LSTM stability
            n_steps=2048,
            batch_size=64,                # Smaller batches for stable gradients
            n_epochs=4,                   # Fewer epochs per update (was 10 default)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,               # Tighter clipping for stable updates
            target_kl=0.01,               # Early stop if KL divergence too high
            ent_coef=0.01,                # Encourage exploration
            max_grad_norm=0.5,
            policy_kwargs={
                "shared_lstm": False,          # Separate LSTMs for actor/critic
                "enable_critic_lstm": True,    # Better value estimation for complex rewards
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
            },
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
        )
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    
    model_path = log_dir / "enhanced_model"
    model.save(str(model_path))
    train_env.save(str(log_dir / "vec_normalize.pkl"))
    
    console.print()
    console.print("[green]Training complete![/green]")
    console.print()
    console.print("[bold]Model Location:[/bold]")
    console.print(f"  Model file: {model_path.resolve()}.zip")
    console.print(f"  VecNormalize: {(log_dir / 'vec_normalize.pkl').resolve()}")
    console.print()
    if balance_callback.total_trades > 0:
        console.print(f"  Final Accuracy: {balance_callback.correct_trades / balance_callback.total_trades:.1%}")


@app.command("train-probability")
def train_probability(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs/probability_model", help="Output directory"),
    epochs: int = typer.Option(100, help="Training epochs"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    model_type: str = typer.Option("lstm", help="Model type (lstm/gru)"),
):
    """Train Layer 1 probability model (LSTM/GRU).
    
    Trains a calibrated probability model for predicting P(candle closes up).
    Uses Brier score loss with temperature scaling calibration.
    """
    import pandas as pd
    from .training.probability_trainer import train_probability_model
    
    data_path = Path(data_dir)
    
    # Load BTC data
    btc_path = data_path / "btcusdt_100ms.parquet"
    if not btc_path.exists():
        btc_path = data_path / "btcusdt_1s_30days.parquet"
    
    if not btc_path.exists():
        console.print("[red]No BTC data found. Run 'collect' first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading BTC data from {btc_path}[/blue]")
    btc_data = pd.read_parquet(btc_path)
    if "close" in btc_data.columns and "price" not in btc_data.columns:
        btc_data = btc_data.rename(columns={"close": "price"})
    
    # Load forex data if available
    dxy_data = None
    eurusd_data = None
    
    if (data_path / "dxy_1h.parquet").exists():
        console.print("[blue]Loading DXY data[/blue]")
        dxy_data = pd.read_parquet(data_path / "dxy_1h.parquet")
    
    if (data_path / "eurusd_1h.parquet").exists():
        console.print("[blue]Loading EUR/USD data[/blue]")
        eurusd_data = pd.read_parquet(data_path / "eurusd_1h.parquet")
    
    console.print()
    console.print(f"[blue]Training probability model ({model_type.upper()})[/blue]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Output: {output_dir}")
    console.print()
    
    model, metrics = train_probability_model(
        btc_data=btc_data,
        dxy_data=dxy_data,
        eurusd_data=eurusd_data,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        model_type=model_type,
    )
    
    console.print()
    console.print("[bold green]Model trained and calibrated![/bold green]")
    console.print(f"  Brier Score: {metrics['brier_score']:.4f}")
    console.print(f"  Accuracy: {metrics['accuracy']:.1%}")
    console.print(f"  ECE: {metrics['ece']:.4f}")


@app.command("train-deep-lob")
def train_deep_lob(
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs/deep_lob_model", help="Output directory"),
    epochs: int = typer.Option(200, help="Training epochs"),
    alpha: float = typer.Option(0.001, help="Threshold for Up/Down classification (0.001 = 0.1%)"),
):
    """Train DeepLOB 3-class model (research-based architecture).
    
    Uses Order Flow Imbalance, CNN-LSTM with attention, and focal loss.
    Predicts: Down (0), Hold (1), Up (2) with threshold filtering.
    Evaluates on F1 score instead of accuracy.
    """
    import pandas as pd
    from .training.deep_lob_trainer import train_deep_lob_model
    
    data_path = Path(data_dir)
    
    # Load BTC data (prefer 100ms with buy_pressure)
    btc_path = data_path / "btcusdt_100ms.parquet"
    if not btc_path.exists():
        console.print("[red]No btcusdt_100ms.parquet found. This model requires buy_pressure.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading BTC data from {btc_path}[/blue]")
    btc_data = pd.read_parquet(btc_path)
    
    # Load trades data if available
    trades_data = None
    trades_path = data_path / "btcusdt_aggtrades.parquet"
    if trades_path.exists():
        console.print("[blue]Loading aggregated trades for TFI[/blue]")
        trades_data = pd.read_parquet(trades_path)
    
    console.print()
    console.print(f"[blue]Training DeepLOB 3-class model[/blue]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Alpha threshold: {alpha} ({alpha*100:.1f}%)")
    console.print(f"  Output: {output_dir}")
    console.print()
    
    model, results = train_deep_lob_model(
        btc_data=btc_data,
        trades_data=trades_data,
        output_dir=output_dir,
        epochs=epochs,
        alpha_threshold=alpha,
    )
    
    console.print()
    console.print("[bold green]DeepLOB model trained![/bold green]")
    console.print(f"  F1 Macro: {results['f1_macro']:.3f}")
    console.print(f"  Accuracy: {results['accuracy']:.1%}")


@app.command("fetch-balanced-data")
def fetch_balanced_data_cmd(
    n_up: int = typer.Option(500, help="Number of Up intervals"),
    n_down: int = typer.Option(500, help="Number of Down intervals"),
    n_hold: int = typer.Option(250, help="Number of Hold intervals"),
    days_back: int = typer.Option(30, help="Days of history to sample from"),
    output_dir: str = typer.Option("./data/balanced", help="Output directory"),
):
    """Fetch balanced training data from Binance.
    
    Randomly samples historical intervals ensuring equal representation
    of Up, Down, and Hold outcomes. This prevents training bias from
    trending markets.
    """
    import asyncio
    from .data.balanced_data_generator import fetch_balanced_data
    
    console.print("[bold blue]Fetching Balanced Training Data[/bold blue]")
    console.print(f"  Target: {n_up} Up, {n_down} Down, {n_hold} Hold")
    console.print(f"  Days back: {days_back}")
    console.print(f"  Output: {output_dir}")
    console.print()
    
    output_file = asyncio.run(fetch_balanced_data(
        n_up=n_up,
        n_down=n_down,
        n_hold=n_hold,
        days_back=days_back,
        output_dir=output_dir,
    ))
    
    console.print()
    console.print(f"[bold green]✓ Balanced data saved to {output_file}[/bold green]")


@app.command("train-deep-lob-balanced")
def train_deep_lob_balanced(
    data_dir: str = typer.Option("./data/balanced", help="Directory with balanced data"),
    output_dir: str = typer.Option("./logs/deep_lob_balanced", help="Output directory"),
    epochs: int = typer.Option(150, help="Training epochs"),
    alpha: float = typer.Option(0.0001, help="Alpha threshold for Up/Down"),
):
    """Train DeepLOB on balanced data (unbiased Up/Down).
    
    Uses data from fetch-balanced-data command which has equal
    representation of Up, Down, and Hold outcomes.
    """
    import pandas as pd
    from .training.deep_lob_trainer import train_deep_lob_balanced
    
    data_path = Path(data_dir)
    balanced_file = data_path / "balanced_intervals.parquet"
    
    if not balanced_file.exists():
        console.print("[red]No balanced_intervals.parquet found.[/red]")
        console.print("Run 'fetch-balanced-data' first to generate balanced training data.")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading balanced data from {balanced_file}[/blue]")
    balanced_data = pd.read_parquet(balanced_file)
    
    # Load metadata
    import json
    metadata_file = data_path / "balanced_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        console.print(f"  Intervals: {metadata['n_intervals']}")
        console.print(f"  Up: {metadata['n_up']}, Down: {metadata['n_down']}, Hold: {metadata['n_hold']}")
    
    console.print()
    console.print(f"[blue]Training DeepLOB on balanced data[/blue]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Output: {output_dir}")
    console.print()
    
    model, results = train_deep_lob_balanced(
        balanced_data=balanced_data,
        output_dir=output_dir,
        epochs=epochs,
        alpha_threshold=alpha,
    )
    
    console.print()
    console.print("[bold green]DeepLOB balanced model trained![/bold green]")
    console.print(f"  F1 Macro: {results['f1_macro']:.3f}")
    console.print(f"  Accuracy: {results['accuracy']:.1%}")


@app.command("train-execution")
def train_execution(
    probability_model: str = typer.Argument(..., help="Path to trained probability model"),
    data_dir: str = typer.Option("./data", help="Data directory"),
    output_dir: str = typer.Option("./logs/execution_model", help="Output directory"),
    timesteps: int = typer.Option(500_000, help="Total training timesteps"),
    n_envs: int = typer.Option(4, help="Number of environments"),
):
    """Train Layer 2 execution policy (SAC).
    
    Uses the trained probability model to generate states and trains
    a SAC agent to make optimal execution decisions.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback
    from .simulation.polymarket_execution_env import (
        PolymarketExecutionEnv,
        PolymarketExecutionConfig,
        make_polymarket_vec_env,
    )
    
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[blue]Training execution policy (SAC)[/blue]")
    console.print(f"  Probability model: {probability_model}")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print()
    
    config = PolymarketExecutionConfig()
    train_env = make_polymarket_vec_env(n_envs, config)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac",
        save_vecnormalize=True,
    )
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
    )
    
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    model_path = log_dir / "execution_model"
    model.save(str(model_path))
    train_env.save(str(log_dir / "vec_normalize.pkl"))
    
    console.print()
    console.print("[green]Execution policy trained![/green]")
    console.print(f"  Model: {model_path}.zip")


@app.command("train-execution-deeplob")
def train_execution_deeplob(
    output_dir: str = typer.Option("./logs/sac_deeplob", help="Output directory"),
    timesteps: int = typer.Option(300_000, help="Total training timesteps"),
    n_envs: int = typer.Option(8, help="Number of environments"),
):
    """Train SAC execution policy for DeepLOB 3-class predictions.
    
    Uses DeepLOBExecutionEnv which receives 3-class probabilities (Down/Hold/Up)
    and learns optimal trade execution decisions.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from .simulation.deep_lob_execution_env import (
        DeepLOBExecutionEnv,
        DeepLOBExecutionConfig,
        make_deep_lob_vec_env,
    )
    
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[blue]Training DeepLOB execution policy (SAC)[/blue]")
    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Environments: {n_envs}")
    console.print()
    
    config = DeepLOBExecutionConfig()
    train_env = make_deep_lob_vec_env(n_envs, config)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Eval env
    eval_env = make_deep_lob_vec_env(1, config)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000 // n_envs,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac",
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=10_000 // n_envs,
        deterministic=True,
    )
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
    )
    
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    model_path = log_dir / "execution_model"
    model.save(str(model_path))
    train_env.save(str(log_dir / "vec_normalize.pkl"))
    
    console.print()
    console.print("[green]DeepLOB execution policy trained![/green]")
    console.print(f"  Model: {model_path}.zip")
    console.print(f"  Best model: {log_dir / 'best_model.zip'}")


@app.command("paper-trade-deeplob")
def paper_trade_deeplob(
    deep_lob_model: str = typer.Option("./logs/deep_lob_v2", help="DeepLOB model path"),
    sac_model: str = typer.Option("./logs/sac_deeplob", help="SAC model path"),
    num_candles: int = typer.Option(30, help="Number of candles to simulate"),
    initial_balance: float = typer.Option(1000.0, help="Initial balance"),
    aggressive: bool = typer.Option(True, help="Use aggressive mode (trade on Up/Down diff)"),
):
    """Paper trade DeepLOB + SAC with random Binance data.
    
    Fetches random intervals of historical data from Binance and simulates
    trading decisions using the trained DeepLOB + SAC two-layer system.
    
    Use --aggressive to force more trades based on Up vs Down probability
    difference instead of waiting for predicted class.
    """
    import asyncio
    from .paper_trade_deeplob import run_paper_trade
    
    asyncio.run(run_paper_trade(
        num_candles=num_candles,
        deep_lob_path=deep_lob_model,
        sac_path=sac_model,
        initial_balance=initial_balance,
        aggressive=aggressive,
    ))


@app.command("paper-trade-polymarket")
def paper_trade_polymarket(
    probability_model: str = typer.Argument(..., help="Path to probability model"),
    execution_model: str = typer.Option(None, help="Path to execution model (optional)"),
    data_dir: str = typer.Option("./data", help="Data directory"),
    num_candles: int = typer.Option(100, help="Number of candles to simulate"),
    initial_balance: float = typer.Option(1000.0, help="Initial balance"),
):
    """Paper trade with the two-layer Polymarket system.
    
    Uses LSTM probability model + SAC execution policy (or rule-based fallback)
    to simulate trading on historical data.
    """
    import pandas as pd
    from .inference.two_layer_inference import create_bot
    
    data_path = Path(data_dir)
    
    # Load data
    btc_path = data_path / "btcusdt_100ms.parquet"
    if not btc_path.exists():
        btc_path = data_path / "btcusdt_1s_30days.parquet"
    
    if not btc_path.exists():
        console.print("[red]No BTC data found.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Loading data...[/blue]")
    btc_data = pd.read_parquet(btc_path)
    if "close" in btc_data.columns and "price" not in btc_data.columns:
        btc_data = btc_data.rename(columns={"close": "price"})
    
    dxy_data = None
    if (data_path / "dxy_1h.parquet").exists():
        dxy_data = pd.read_parquet(data_path / "dxy_1h.parquet")
    
    eurusd_data = None
    if (data_path / "eurusd_1h.parquet").exists():
        eurusd_data = pd.read_parquet(data_path / "eurusd_1h.parquet")
    
    # Create bot
    console.print(f"[blue]Loading models...[/blue]")
    bot = create_bot(probability_model, execution_model)
    
    # Create candles
    candle_minutes = 15
    candle_seconds = candle_minutes * 60
    
    if not pd.api.types.is_datetime64_any_dtype(btc_data["timestamp"]):
        btc_data["timestamp"] = pd.to_datetime(btc_data["timestamp"], utc=True)
    
    btc_data["candle_idx"] = (btc_data["timestamp"].astype("int64") // 10**9 // candle_seconds).astype(int)
    
    candles = []
    for candle_id, group in btc_data.groupby("candle_idx"):
        if len(group) < 10:
            continue
        candles.append({
            "start_idx": group.index[0],
            "end_idx": group.index[-1],
            "open": group["price"].iloc[0],
            "close": group["price"].iloc[-1],
            "return": (group["price"].iloc[-1] - group["price"].iloc[0]) / group["price"].iloc[0],
        })
    
    if len(candles) < num_candles + 100:
        console.print(f"[yellow]Not enough candles. Have {len(candles)}, need {num_candles + 100}[/yellow]")
        num_candles = len(candles) - 100
    
    # Simulate trading
    console.print()
    console.print(f"[blue]Simulating {num_candles} candles...[/blue]")
    
    balance = initial_balance
    trades = []
    holds = []
    
    from rich.progress import Progress
    
    with Progress() as progress:
        task = progress.add_task("Trading", total=num_candles)
        
        for i in range(100, 100 + num_candles):
            candle = candles[i]
            
            # Get data up to candle midpoint
            mid_idx = (candle["start_idx"] + candle["end_idx"]) // 2
            feature_data = btc_data.iloc[:mid_idx]
            
            # Simulate market price (with some noise from model)
            market_yes = 0.5 + candle["return"] * 10 + 0.02 * (2 * (i % 2) - 1)
            market_yes = max(0.1, min(0.9, market_yes))
            
            candle_end = btc_data["timestamp"].iloc[candle["end_idx"]]
            
            # Get decision
            decision = bot.step(
                btc_data=feature_data.tail(2000),
                market_yes_price=market_yes,
                dxy_data=dxy_data,
                eurusd_data=eurusd_data,
                candle_end_timestamp=candle_end,
            )
            
            # Calculate outcome
            outcome = candle["close"] > candle["open"]
            
            if decision["action"] == "hold":
                holds.append(decision)
            else:
                # Execute trade
                size = decision["size"] * balance
                
                if decision["action"] == "buy_yes":
                    pnl = size * (1.0 if outcome else -1.0)
                else:  # buy_no
                    pnl = size * (1.0 if not outcome else -1.0)
                
                # Apply fees
                pnl -= size * 0.001  # 0.1% fee
                
                balance += pnl
                
                bot.record_trade_outcome(decision, outcome, pnl)
                trades.append({**decision, "pnl": pnl, "outcome": outcome})
            
            progress.update(task, advance=1)
    
    # Report results
    console.print()
    console.print("[bold]Results:[/bold]")
    console.print(f"  Initial Balance: ${initial_balance:,.2f}")
    console.print(f"  Final Balance: ${balance:,.2f}")
    console.print(f"  PnL: ${balance - initial_balance:,.2f} ({(balance/initial_balance - 1)*100:.1f}%)")
    console.print()
    console.print(f"  Trades: {len(trades)}")
    console.print(f"  Holds: {len(holds)}")
    console.print(f"  Hold Rate: {len(holds) / num_candles:.1%}")
    
    if trades:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        console.print(f"  Win Rate: {wins / len(trades):.1%}")
        console.print(f"  Avg Edge: {sum(t['edge'] for t in trades) / len(trades):.3f}")

@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    data_dir: str = typer.Option("./data", help="Data directory"),
    market_id: Optional[str] = typer.Option(None, help="Specific market to evaluate"),
    episodes: int = typer.Option(10, help="Number of evaluation episodes"),
):
    """Evaluate a trained model."""
    from .data.storage import DataStorage
    from .models.agent import TradingAgent
    from .training.evaluation import Evaluator, generate_report
    from .simulation.environment import PolymarketEnv, EnvConfig
    
    storage = DataStorage(data_dir)
    agent = TradingAgent.load(model_path)
    
    # Get market data
    if market_id is None:
        markets = storage.get_available_markets_with_data()
        if not markets:
            console.print("[red]No market data available[/red]")
            raise typer.Exit(1)
        market_id = markets[0]
    
    df = storage.load_price_data(market_id)
    if df is None:
        console.print(f"[red]No data for market {market_id}[/red]")
        raise typer.Exit(1)
    
    # Run evaluation
    console.print(f"[blue]Evaluating on {market_id}[/blue]")
    
    evaluator = Evaluator()
    result = evaluator.backtest(agent, df, market_id=market_id)
    
    # Print report
    report = generate_report(result)
    console.print(report)


@app.command()
def export(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    output_path: str = typer.Argument(..., help="Output path"),
    format: str = typer.Option("torchscript", help="Export format (checkpoint/torchscript/onnx)"),
):
    """Export a trained model for production."""
    from .models.agent import TradingAgent
    from .live.export import ModelExporter
    
    agent = TradingAgent.load(model_path)
    exporter = ModelExporter()
    
    console.print(f"[blue]Exporting model to {format}...[/blue]")
    
    if format == "checkpoint":
        path = exporter.export_checkpoint(agent, output_path)
    elif format == "torchscript":
        path = exporter.export_torchscript(agent, output_path)
    elif format == "onnx":
        path = exporter.export_onnx(agent, output_path)
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Exported to: {path}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the web server."""
    import uvicorn
    
    console.print(f"[blue]Starting server on {host}:{port}[/blue]")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def paper_trade(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    market_id: str = typer.Option("btc-100k", help="Market to trade"),
    duration_hours: float = typer.Option(1.0, help="Trading duration in hours"),
    initial_balance: float = typer.Option(10000.0, help="Initial paper balance"),
):
    """Run paper trading with a trained model."""
    import asyncio
    from .models.agent import TradingAgent
    from .live.adapter import PaperTradingAdapter, LiveTradingBot
    
    agent = TradingAgent.load(model_path)
    adapter = PaperTradingAdapter(initial_balance=initial_balance)
    bot = LiveTradingBot(agent, adapter)
    
    console.print(f"[blue]Starting paper trading[/blue]")
    console.print(f"  Market: {market_id}")
    console.print(f"  Duration: {duration_hours} hours")
    console.print(f"  Balance: ${initial_balance:,.2f}")
    console.print()
    
    max_iterations = int(duration_hours * 3600)
    
    asyncio.run(bot.run(
        market_id=market_id,
        interval_seconds=1.0,
        max_iterations=max_iterations,
    ))
    
    console.print()
    console.print(f"[green]Paper trading complete![/green]")
    console.print(f"  Final balance: ${adapter.balance:,.2f}")
    console.print(f"  PnL: ${adapter.balance - initial_balance:,.2f}")


@app.command()
def list_models(
    data_dir: str = typer.Option("./data", help="Data directory"),
):
    """List available trained models."""
    from .data.storage import DataStorage
    
    storage = DataStorage(data_dir)
    models = storage.list_models()
    
    if not models:
        console.print("[yellow]No models found[/yellow]")
        return
    
    table = Table(title="Trained Models")
    table.add_column("Name", style="cyan")
    table.add_column("Versions", style="green")
    table.add_column("Latest", style="yellow")
    
    for model in models:
        versions = model["versions"]
        latest = versions[0] if versions else {}
        
        table.add_row(
            model["name"],
            str(len(versions)),
            latest.get("modified_at", "N/A")[:10] if latest else "N/A",
        )
    
    console.print(table)


@app.command()
def list_runs(
    data_dir: str = typer.Option("./data", help="Data directory"),
    limit: int = typer.Option(10, help="Number of runs to show"),
):
    """List training runs."""
    from .data.storage import DataStorage
    
    storage = DataStorage(data_dir)
    runs = storage.get_training_runs(limit=limit)
    
    if not runs:
        console.print("[yellow]No training runs found[/yellow]")
        return
    
    table = Table(title="Training Runs")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Steps", style="yellow")
    table.add_column("Best Reward", style="magenta")
    
    for run in runs:
        status_style = {
            "running": "blue",
            "completed": "green",
            "failed": "red",
        }.get(run["status"], "white")
        
        table.add_row(
            str(run["id"]),
            run["name"],
            f"[{status_style}]{run['status']}[/{status_style}]",
            f"{run['total_steps']:,}",
            f"{run['best_reward']:.2f}" if run['best_reward'] else "N/A",
        )
    
    console.print(table)


@app.command("paper-trade-live")
def paper_trade_live_cmd(
    deep_lob_model: str = typer.Option("./logs/deep_lob_balanced", help="Path to DeepLOB model"),
    balance: float = typer.Option(1000.0, help="Initial balance in USD"),
    aggressive: bool = typer.Option(True, help="Use aggressive trading mode"),
):
    """Live paper trading with real-time Binance data.
    
    Simulates Polymarket 15-minute crypto markets using live BTC data.
    Each 15-minute candle is a separate market where we bet on Up/Down.
    
    Press Ctrl+C to stop.
    """
    import asyncio
    from .paper_trade_live import run_live_paper_trading
    
    asyncio.run(run_live_paper_trading(
        deep_lob_model=deep_lob_model,
        initial_balance=balance,
        aggressive=aggressive,
    ))


if __name__ == "__main__":
    app()

