#!/usr/bin/env python3
"""
Evaluate Multi-Asset Model with Full PnL Simulation.

Features:
- Realistic PnL tracking with compounding
- Risk management metrics (drawdown, Sharpe, Sortino)
- Position sizing analysis
- Kelly criterion comparison
- Equity curve visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.simulation.multi_asset_env import MultiAssetEnv, MultiAssetConfig

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

console = Console()


@dataclass
class Trade:
    """Single trade record."""
    candle_idx: int
    direction: str
    size: float
    entry_price: float
    candle_return: float
    pnl_pct: float
    balance_after: float
    is_correct: bool


@dataclass
class RiskMetrics:
    """Risk and performance metrics."""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_position_size: float
    kelly_optimal: float
    risk_of_ruin: float


class PnLSimulator:
    """
    Simulates realistic PnL with proper risk management.
    
    Uses asymmetric rewards matching the training environment:
    - Wins: Linear return based on position size
    - Losses: Quadratic penalty (punishes overconfidence)
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_position_pct: float = 0.5,
        use_kelly_sizing: bool = False,
        kelly_fraction: float = 0.25,  # Fractional Kelly for safety
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position_pct = max_position_pct
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction = kelly_fraction
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_balance]
        self.peak_balance = initial_balance
        self.drawdowns: List[float] = []
        
    def execute_trade(
        self,
        direction: str,
        model_size: float,
        candle_return: float,
        entry_price: float,
        candle_idx: int,
    ) -> float:
        """
        Execute a trade and return the reward.
        
        Args:
            direction: 'UP' or 'DOWN'
            model_size: Position size from model (0-1)
            candle_return: Actual candle return (decimal)
            entry_price: Price at bet entry
            candle_idx: Candle index for tracking
        
        Returns:
            reward: PnL percentage
        """
        actual_direction = "UP" if candle_return > 0 else "DOWN"
        is_correct = direction == actual_direction
        
        # Apply position sizing
        if self.use_kelly_sizing:
            # Use Kelly criterion with historical win rate
            position_size = self._compute_kelly_size(model_size)
        else:
            position_size = model_size * self.max_position_pct
        
        position_size = min(position_size, self.max_position_pct)
        
        # Calculate PnL using asymmetric reward structure
        move_pct = abs(candle_return) * 100
        
        if is_correct:
            # Win: Linear reward
            pnl_pct = move_pct * position_size
        else:
            # Loss: Quadratic penalty
            pnl_pct = -move_pct * (position_size ** 2) * 2.0
        
        # Update balance
        self.balance *= (1 + pnl_pct / 100)
        
        # Track equity and drawdown
        self.equity_curve.append(self.balance)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.drawdowns.append(current_drawdown)
        
        # Record trade
        trade = Trade(
            candle_idx=candle_idx,
            direction=direction,
            size=position_size,
            entry_price=entry_price,
            candle_return=candle_return,
            pnl_pct=pnl_pct,
            balance_after=self.balance,
            is_correct=is_correct,
        )
        self.trades.append(trade)
        
        return pnl_pct
    
    def _compute_kelly_size(self, model_confidence: float) -> float:
        """Compute Kelly-optimal position size based on historical performance."""
        if len(self.trades) < 10:
            return model_confidence * 0.1  # Conservative until we have data
        
        wins = [t for t in self.trades if t.is_correct]
        losses = [t for t in self.trades if not t.is_correct]
        
        if not losses:
            return 0.25  # Cap at 25% if no losses yet
        
        win_rate = len(wins) / len(self.trades)
        avg_win = np.mean([abs(t.pnl_pct) for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 1
        
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-p
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly = 0.1
        
        # Apply fractional Kelly and scale by model confidence
        kelly = max(0, kelly) * self.kelly_fraction * model_confidence
        
        return min(kelly, 0.25)  # Never bet more than 25%
    
    def compute_metrics(self) -> RiskMetrics:
        """Compute comprehensive risk and performance metrics."""
        if not self.trades:
            return RiskMetrics(
                total_return=0, annualized_return=0, max_drawdown=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
                largest_win=0, largest_loss=0, consecutive_wins=0,
                consecutive_losses=0, avg_position_size=0, kelly_optimal=0,
                risk_of_ruin=0,
            )
        
        wins = [t for t in self.trades if t.is_correct]
        losses = [t for t in self.trades if not t.is_correct]
        
        # Returns
        total_return = (self.balance / self.initial_balance - 1) * 100
        
        # Assume ~4 candles per hour, ~24 hours per day
        candles_per_year = 4 * 24 * 365
        n_candles = len(self.trades)
        if n_candles > 0:
            annualized_return = ((1 + total_return/100) ** (candles_per_year / n_candles) - 1) * 100
        else:
            annualized_return = 0
        
        # Drawdown
        max_drawdown = max(self.drawdowns) * 100 if self.drawdowns else 0
        
        # Sharpe Ratio (annualized)
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(candles_per_year / n_candles * len(returns))
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (downside deviation only)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns and np.std(downside_returns) > 0:
            sortino_ratio = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(candles_per_year / n_candles * len(returns))
        else:
            sortino_ratio = sharpe_ratio
        
        # Calmar Ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win/Loss stats
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        win_pnls = [t.pnl_pct for t in wins]
        loss_pnls = [abs(t.pnl_pct) for t in losses]
        
        total_wins = sum(win_pnls) if win_pnls else 0
        total_losses = sum(loss_pnls) if loss_pnls else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        largest_win = max(win_pnls) if win_pnls else 0
        largest_loss = max(loss_pnls) if loss_pnls else 0
        
        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(True)
        consecutive_losses = self._max_consecutive(False)
        
        # Position sizing
        avg_position_size = np.mean([t.size for t in self.trades])
        
        # Kelly optimal (theoretical)
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_optimal = max(0, (b * win_rate - (1 - win_rate)) / b)
        else:
            kelly_optimal = 0.25
        
        # Risk of Ruin (simplified)
        # P(ruin) ≈ ((1-edge)/(1+edge))^(bankroll/bet_size)
        edge = win_rate - 0.5
        if edge > 0 and avg_position_size > 0:
            units = 1 / avg_position_size
            risk_of_ruin = ((1 - edge) / (1 + edge)) ** units
        else:
            risk_of_ruin = 1.0 if edge <= 0 else 0.0
        
        return RiskMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            avg_position_size=avg_position_size,
            kelly_optimal=kelly_optimal,
            risk_of_ruin=risk_of_ruin,
        )
    
    def _max_consecutive(self, wins: bool) -> int:
        """Find maximum consecutive wins or losses."""
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.is_correct == wins:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def print_equity_curve_ascii(self, width: int = 60, height: int = 15):
        """Print ASCII equity curve."""
        if len(self.equity_curve) < 2:
            return
        
        # Normalize values
        min_eq = min(self.equity_curve)
        max_eq = max(self.equity_curve)
        range_eq = max_eq - min_eq if max_eq != min_eq else 1
        
        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        for i, eq in enumerate(self.equity_curve):
            x = int(i / len(self.equity_curve) * (width - 1))
            y = int((eq - min_eq) / range_eq * (height - 1))
            y = height - 1 - y  # Flip y-axis
            if 0 <= x < width and 0 <= y < height:
                canvas[y][x] = '█'
        
        # Print
        console.print("\n[bold cyan]Equity Curve[/bold cyan]")
        console.print(f"${max_eq:,.0f} ┤")
        for row in canvas:
            console.print("      │" + ''.join(row))
        console.print(f"${min_eq:,.0f} ┼" + "─" * width)


def evaluate_multi_model(
    model_path: str = "logs/recurrent_multi_asset_500000/multi_asset_model",
    data_dir: str = "data",
    n_episodes: int = 500,
    initial_balance: float = 10000.0,
    use_kelly: bool = False,
):
    """
    Evaluate model with full PnL simulation and risk metrics.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        old_path = Path("logs/multi_asset/multi_asset_model")
        if old_path.exists():
            model_path = old_path
    
    console.print(f"[blue]Loading model from {model_path}...[/blue]")
    
    # Load model
    model = None
    is_recurrent = False
    
    if RecurrentPPO:
        try:
            model = RecurrentPPO.load(model_path)
            console.print("[green]✓ Loaded RecurrentPPO (LSTM)[/green]")
            is_recurrent = True
        except:
            pass
    
    if model is None:
        try:
            model = SAC.load(model_path)
            console.print("[green]✓ Loaded SAC (MLP)[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            return
    
    # Load data
    console.print("[blue]Loading data...[/blue]")
    
    btc_path = Path(data_dir) / "btcusdt_100ms.parquet"
    if not btc_path.exists():
        btc_path = Path(data_dir) / "btcusdt_1s_30days.parquet"
    
    btc = pd.read_parquet(btc_path)
    if "close" in btc.columns and "price" not in btc.columns:
        btc = btc.rename(columns={"close": "price"})
    
    dxy = None
    if (Path(data_dir) / "dxy_1h.parquet").exists():
        dxy = pd.read_parquet(Path(data_dir) / "dxy_1h.parquet")
    
    eurusd = None
    if (Path(data_dir) / "eurusd_1h.parquet").exists():
        eurusd = pd.read_parquet(Path(data_dir) / "eurusd_1h.parquet")
    
    # Create environment
    config = MultiAssetConfig(random_start=True)
    env = DummyVecEnv([lambda: MultiAssetEnv(btc, dxy, eurusd, config=config)])
    
    vec_path = model_path.parent / "vec_normalize.pkl"
    if vec_path.exists():
        console.print("[green]✓ Loading normalization stats[/green]")
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False
    
    # Initialize PnL simulator
    simulator = PnLSimulator(
        initial_balance=initial_balance,
        use_kelly_sizing=use_kelly,
    )
    
    console.print(f"\n[bold]Running {n_episodes} episodes with ${initial_balance:,.0f} capital...[/bold]\n")
    
    # Track stats
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    up_bets = 0
    down_bets = 0
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_starts[0] = True
        done = False
        
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            episode_starts[0] = False
            obs, _, done, info = env.step(action)
            
            done = done[0]
            inf = info[0]
            
            if done and inf.get("position_taken"):
                direction = inf["position_direction"]
                size = inf["position_size"]
                candle_return = inf["candle_return"]
                candle_idx = inf.get("candle_idx", episode)
                
                if direction == "UP":
                    up_bets += 1
                else:
                    down_bets += 1
                
                pnl = simulator.execute_trade(
                    direction=direction,
                    model_size=size,
                    candle_return=candle_return,
                    entry_price=0,  # Not used for PnL calc
                    candle_idx=candle_idx,
                )
                
                # Print first 20 trades
                trade = simulator.trades[-1]
                if len(simulator.trades) <= 20:
                    result = "[green]✓[/green]" if trade.is_correct else "[red]✗[/red]"
                    console.print(
                        f"Trade {len(simulator.trades):3d}: {direction:4s} "
                        f"(Size: {trade.size:.1%}) → "
                        f"{'UP  ' if candle_return > 0 else 'DOWN'} ({candle_return*100:+.2f}%) | "
                        f"PnL: {pnl:+.3f}% | Bal: ${trade.balance_after:,.0f} {result}"
                    )
    
    # Compute and display metrics
    metrics = simulator.compute_metrics()
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]📊 EVALUATION RESULTS[/bold cyan]")
    console.print("=" * 70)
    
    # Performance table
    perf_table = Table(title="Performance Metrics", show_header=True, header_style="bold green")
    perf_table.add_column("Metric", style="dim")
    perf_table.add_column("Value", justify="right")
    
    pnl_color = "green" if metrics.total_return >= 0 else "red"
    perf_table.add_row("Initial Balance", f"${initial_balance:,.0f}")
    perf_table.add_row("Final Balance", f"${simulator.balance:,.0f}")
    perf_table.add_row("Total Return", f"[{pnl_color}]{metrics.total_return:+.2f}%[/{pnl_color}]")
    perf_table.add_row("Annualized Return", f"{metrics.annualized_return:+.1f}%")
    perf_table.add_row("Max Drawdown", f"[red]{metrics.max_drawdown:.2f}%[/red]")
    
    console.print(perf_table)
    
    # Risk table
    risk_table = Table(title="Risk Metrics", show_header=True, header_style="bold yellow")
    risk_table.add_column("Metric", style="dim")
    risk_table.add_column("Value", justify="right")
    
    sharpe_color = "green" if metrics.sharpe_ratio > 1 else "yellow" if metrics.sharpe_ratio > 0 else "red"
    risk_table.add_row("Sharpe Ratio", f"[{sharpe_color}]{metrics.sharpe_ratio:.2f}[/{sharpe_color}]")
    risk_table.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    risk_table.add_row("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
    risk_table.add_row("Risk of Ruin", f"{metrics.risk_of_ruin:.1%}")
    
    console.print(risk_table)
    
    # Trading table
    trade_table = Table(title="Trading Statistics", show_header=True, header_style="bold magenta")
    trade_table.add_column("Metric", style="dim")
    trade_table.add_column("Value", justify="right")
    
    trade_table.add_row("Total Trades", f"{len(simulator.trades)}")
    trade_table.add_row("Win Rate", f"{metrics.win_rate:.1%}")
    trade_table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
    trade_table.add_row("UP / DOWN Bets", f"{up_bets} / {down_bets}")
    trade_table.add_row("Avg Win", f"[green]+{metrics.avg_win:.3f}%[/green]")
    trade_table.add_row("Avg Loss", f"[red]-{metrics.avg_loss:.3f}%[/red]")
    trade_table.add_row("Largest Win", f"[green]+{metrics.largest_win:.3f}%[/green]")
    trade_table.add_row("Largest Loss", f"[red]-{metrics.largest_loss:.3f}%[/red]")
    trade_table.add_row("Max Consecutive Wins", f"{metrics.consecutive_wins}")
    trade_table.add_row("Max Consecutive Losses", f"{metrics.consecutive_losses}")
    
    console.print(trade_table)
    
    # Position sizing table
    size_table = Table(title="Position Sizing", show_header=True, header_style="bold blue")
    size_table.add_column("Metric", style="dim")
    size_table.add_column("Value", justify="right")
    
    size_table.add_row("Avg Position Size", f"{metrics.avg_position_size:.1%}")
    size_table.add_row("Kelly Optimal", f"{metrics.kelly_optimal:.1%}")
    size_table.add_row("Using Kelly", f"{'Yes' if use_kelly else 'No'}")
    
    console.print(size_table)
    
    # Equity curve
    simulator.print_equity_curve_ascii()
    
    console.print("\n[dim]Use --kelly flag to enable Kelly criterion position sizing[/dim]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multi-asset model with PnL simulation")
    parser.add_argument("--model", "-m", default="logs/recurrent_multi_asset/multi_asset_model")
    parser.add_argument("--data", "-d", default="data")
    parser.add_argument("--episodes", "-n", type=int, default=500)
    parser.add_argument("--balance", "-b", type=float, default=10000.0)
    parser.add_argument("--kelly", action="store_true", help="Use Kelly criterion sizing")
    
    args = parser.parse_args()
    
    evaluate_multi_model(
        model_path=args.model,
        data_dir=args.data,
        n_episodes=args.episodes,
        initial_balance=args.balance,
        use_kelly=args.kelly,
    )
