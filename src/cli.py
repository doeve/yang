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


if __name__ == "__main__":
    app()
