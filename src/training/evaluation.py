"""
Evaluation and backtesting for trained models.

Provides comprehensive evaluation tools:
- Backtesting on historical data
- Performance metrics calculation
- Trade analysis
- Visualization
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from ..models.agent import TradingAgent
from ..simulation.environment import PolymarketEnv, EnvConfig
from ..simulation.market_simulator import Portfolio

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: datetime
    side: str  # "buy" or "sell"
    size: float
    price: float
    fees: float
    pnl: float = 0.0  # Realized PnL if closing
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "fees": self.fees,
            "pnl": self.pnl,
        }


@dataclass
class BacktestResult:
    """Results from backtesting a model."""
    
    # Basic metrics
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Time series
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    drawdown_curve: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    
    # Trades
    trades: list[Trade] = field(default_factory=list)
    
    # Metadata
    market_id: str = ""
    start_time: str = ""
    end_time: str = ""
    initial_capital: float = 10000.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "market_id": self.market_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class Evaluator:
    """
    Evaluates trained trading models.
    
    Provides:
    - Episode-based evaluation
    - Full backtesting with trade tracking
    - Performance metrics calculation
    - Comparison between models
    
    Example:
        evaluator = Evaluator()
        result = evaluator.backtest(agent, price_data)
        print(f"Sharpe: {result.sharpe_ratio}")
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def evaluate(
        self,
        agent: TradingAgent,
        env: PolymarketEnv | Any,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate agent on environment for multiple episodes.
        
        Args:
            agent: Trained trading agent
            env: Evaluation environment
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
            
        Returns:
            Dict with mean and std of reward
        """
        from stable_baselines3.common.vec_env import VecEnv
        
        rewards = []
        is_vec_env = isinstance(env, VecEnv)
        
        for _ in range(n_episodes):
            # VecEnv.reset() returns just obs, Gym returns (obs, info)
            reset_result = env.reset()
            if is_vec_env:
                obs = reset_result
            else:
                obs, _ = reset_result
            
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = agent.predict(obs, deterministic=deterministic)
                
                # VecEnv.step() returns (obs, reward, done, info)
                # Gym.step() returns (obs, reward, terminated, truncated, info)
                if is_vec_env:
                    obs, reward, dones, infos = env.step(action)
                    episode_reward += float(reward[0]) if hasattr(reward, '__len__') else float(reward)
                    done = dones[0] if hasattr(dones, '__len__') else dones
                else:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += float(reward)
                    done = terminated or truncated
            
            rewards.append(episode_reward)
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }
    
    def backtest(
        self,
        agent: TradingAgent,
        price_data: pd.DataFrame,
        market_id: str = "backtest",
        resolution_at: datetime | None = None,
        created_at: datetime | None = None,
        outcome: bool | None = None,
        initial_capital: float = 10000.0,
        deterministic: bool = True,
    ) -> BacktestResult:
        """
        Run full backtest on historical data.
        
        Args:
            agent: Trained trading agent
            price_data: DataFrame with timestamp, price, volume
            market_id: Market identifier
            resolution_at: Market resolution time
            created_at: Market creation time
            outcome: Market outcome
            initial_capital: Starting capital
            deterministic: Use deterministic policy
            
        Returns:
            BacktestResult with full analysis
        """
        # Create environment
        env_config = EnvConfig(random_start=False)
        env = PolymarketEnv(
            price_data,
            config=env_config,
            market_id=market_id,
            resolution_at=resolution_at,
            created_at=created_at,
            outcome=outcome,
        )
        
        # Run episode
        obs, info = env.reset()
        
        equity_curve = [initial_capital]
        returns = []
        trades: list[Trade] = []
        timestamps = [info.get("timestamp", "")]
        
        prev_position = 0.0
        prev_equity = initial_capital
        
        done = False
        step = 0
        
        while not done:
            # Get action
            action, _ = agent.predict(obs, deterministic=deterministic)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Track equity
            current_equity = initial_capital + info.get("total_pnl", 0)
            equity_curve.append(current_equity)
            
            # Calculate return
            if prev_equity > 0:
                ret = (current_equity - prev_equity) / prev_equity
            else:
                ret = 0.0
            returns.append(ret)
            
            # Track trades
            current_position = info.get("position", 0)
            if current_position != prev_position:
                side = "buy" if current_position > prev_position else "sell"
                trade = Trade(
                    timestamp=datetime.fromisoformat(info.get("timestamp", datetime.now().isoformat())),
                    side=side,
                    size=abs(current_position - prev_position),
                    price=info.get("price", 0.5),
                    fees=0.0,  # Would need to track from simulator
                    pnl=info.get("realized_pnl", 0) if side == "sell" else 0,
                )
                trades.append(trade)
            
            # Update trackers
            prev_position = current_position
            prev_equity = current_equity
            timestamps.append(info.get("timestamp", ""))
        
        # Calculate metrics
        result = self._calculate_metrics(
            equity_curve,
            returns,
            trades,
            timestamps,
            initial_capital,
            market_id,
        )
        
        return result
    
    def _calculate_metrics(
        self,
        equity_curve: list[float],
        returns: list[float],
        trades: list[Trade],
        timestamps: list[str],
        initial_capital: float,
        market_id: str,
    ) -> BacktestResult:
        """Calculate all performance metrics."""
        result = BacktestResult(
            market_id=market_id,
            initial_capital=initial_capital,
            equity_curve=equity_curve,
            returns=returns,
            timestamps=timestamps,
            trades=trades,
            start_time=timestamps[0] if timestamps else "",
            end_time=timestamps[-1] if timestamps else "",
        )
        
        if not equity_curve or len(equity_curve) < 2:
            return result
        
        equity = np.array(equity_curve)
        rets = np.array(returns)
        
        # Basic metrics
        result.total_pnl = equity[-1] - initial_capital
        result.total_return = (equity[-1] - initial_capital) / initial_capital
        
        # Sharpe ratio (annualized, assuming 1-second steps)
        if len(rets) > 1 and rets.std() > 0:
            # Convert to annualized
            periods_per_year = 365 * 24 * 3600  # Seconds in a year
            excess_returns = rets - self.risk_free_rate / periods_per_year
            result.sharpe_ratio = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
            )
        
        # Sortino ratio (penalize only downside)
        downside_returns = rets[rets < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            periods_per_year = 365 * 24 * 3600
            result.sortino_ratio = (
                rets.mean() / downside_returns.std() * np.sqrt(periods_per_year)
            )
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        result.max_drawdown = float(drawdown.max())
        result.drawdown_curve = drawdown.tolist()
        
        # Calmar ratio
        if result.max_drawdown > 0:
            result.calmar_ratio = result.total_return / result.max_drawdown
        
        # Trade statistics
        result.total_trades = len(trades)
        
        if trades:
            winning = [t for t in trades if t.pnl > 0]
            losing = [t for t in trades if t.pnl < 0]
            
            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            
            if result.total_trades > 0:
                result.win_rate = result.winning_trades / result.total_trades
            
            if winning:
                result.avg_win = np.mean([t.pnl for t in winning])
            if losing:
                result.avg_loss = np.mean([abs(t.pnl) for t in losing])
            
            total_profit = sum(t.pnl for t in winning)
            total_loss = abs(sum(t.pnl for t in losing))
            
            if total_loss > 0:
                result.profit_factor = total_profit / total_loss
        
        return result
    
    def compare_models(
        self,
        agents: dict[str, TradingAgent],
        price_data: pd.DataFrame,
        **backtest_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same data.
        
        Args:
            agents: Dict mapping model name to agent
            price_data: Price data for backtesting
            **backtest_kwargs: Additional backtest arguments
            
        Returns:
            DataFrame comparing all models
        """
        results = []
        
        for name, agent in agents.items():
            result = self.backtest(agent, price_data, **backtest_kwargs)
            
            results.append({
                "model": name,
                **result.to_dict(),
            })
        
        return pd.DataFrame(results)
    
    def monte_carlo_analysis(
        self,
        agent: TradingAgent,
        price_data: pd.DataFrame,
        n_simulations: int = 100,
        **backtest_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo analysis with random starting points.
        
        Args:
            agent: Trading agent
            price_data: Price data
            n_simulations: Number of simulations
            **backtest_kwargs: Additional backtest arguments
            
        Returns:
            Statistics from Monte Carlo analysis
        """
        sharpes = []
        returns = []
        drawdowns = []
        
        for i in range(n_simulations):
            # Random starting point
            max_start = len(price_data) - 1000
            if max_start > 100:
                start_idx = np.random.randint(100, max_start)
                data_slice = price_data.iloc[start_idx:start_idx + 1000].copy()
            else:
                data_slice = price_data.copy()
            
            result = self.backtest(agent, data_slice, **backtest_kwargs)
            
            sharpes.append(result.sharpe_ratio)
            returns.append(result.total_return)
            drawdowns.append(result.max_drawdown)
        
        return {
            "sharpe": {
                "mean": float(np.mean(sharpes)),
                "std": float(np.std(sharpes)),
                "median": float(np.median(sharpes)),
                "percentile_5": float(np.percentile(sharpes, 5)),
                "percentile_95": float(np.percentile(sharpes, 95)),
            },
            "return": {
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "median": float(np.median(returns)),
                "percentile_5": float(np.percentile(returns, 5)),
                "percentile_95": float(np.percentile(returns, 95)),
            },
            "max_drawdown": {
                "mean": float(np.mean(drawdowns)),
                "std": float(np.std(drawdowns)),
                "median": float(np.median(drawdowns)),
                "percentile_5": float(np.percentile(drawdowns, 5)),
                "percentile_95": float(np.percentile(drawdowns, 95)),
            },
            "n_simulations": n_simulations,
        }


def generate_report(result: BacktestResult) -> str:
    """
    Generate a text report from backtest results.
    
    Args:
        result: BacktestResult to report on
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        "",
        f"Market: {result.market_id}",
        f"Period: {result.start_time} to {result.end_time}",
        f"Initial Capital: ${result.initial_capital:,.2f}",
        "",
        "PERFORMANCE METRICS",
        "-" * 40,
        f"Total PnL:        ${result.total_pnl:,.2f}",
        f"Total Return:     {result.total_return * 100:.2f}%",
        f"Sharpe Ratio:     {result.sharpe_ratio:.2f}",
        f"Sortino Ratio:    {result.sortino_ratio:.2f}",
        f"Max Drawdown:     {result.max_drawdown * 100:.2f}%",
        f"Calmar Ratio:     {result.calmar_ratio:.2f}",
        "",
        "TRADE STATISTICS",
        "-" * 40,
        f"Total Trades:     {result.total_trades}",
        f"Winning Trades:   {result.winning_trades}",
        f"Losing Trades:    {result.losing_trades}",
        f"Win Rate:         {result.win_rate * 100:.1f}%",
        f"Profit Factor:    {result.profit_factor:.2f}",
        f"Avg Win:          ${result.avg_win:.2f}",
        f"Avg Loss:         ${result.avg_loss:.2f}",
        "",
        "=" * 60,
    ]
    
    return "\n".join(lines)
