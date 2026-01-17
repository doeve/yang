#!/usr/bin/env python3
"""
Evaluate Enhanced Multi-Asset Model.

Tests the enhanced model with order flow, multi-timeframe,
skip action, and risk-adjusted rewards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.simulation.enhanced_multi_asset_env import (
    EnhancedMultiAssetEnv,
    EnhancedMultiAssetConfig,
)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

console = Console()


def evaluate_enhanced_model(
    model_path: str = "logs/my_enhanced_model/enhanced_model",
    data_dir: str = "data",
    n_episodes: int = 100,
):
    model_path = Path(model_path)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        return
    
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
    config = EnhancedMultiAssetConfig(
        random_start=True,
        include_order_flow=True,
        include_multi_timeframe=True,
        enable_skip=True,
    )
    
    env = DummyVecEnv([lambda: EnhancedMultiAssetEnv(btc, dxy, eurusd, config=config)])
    
    # Load normalization
    vec_path = model_path.parent / "vec_normalize.pkl"
    if vec_path.exists():
        console.print("[green]✓ Loading normalization stats[/green]")
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False
    
    console.print(f"\n[bold]Evaluating for {n_episodes} episodes...[/bold]\n")
    
    # Tracking
    results = {
        "correct": 0,
        "wrong": 0,
        "skipped": 0,
        "total_reward": 0.0,
        "position_sizes": [],
        "entry_times": [],
        "up_bets": 0,
        "down_bets": 0,
    }
    
    # LSTM state
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_starts[0] = True
        done = False
        episode_reward = 0.0
        
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            episode_starts[0] = False
            obs, reward, done, info = env.step(action)
            
            reward = reward[0]
            done = done[0]
            inf = info[0]
            episode_reward += reward
            
            if done:
                results["total_reward"] += episode_reward
                
                # Check action taken
                skip_prob = float(action[0][2]) if len(action[0]) > 2 else 0.0
                
                if skip_prob > 0.5:
                    results["skipped"] += 1
                elif inf.get("position_taken"):
                    direction = inf["position_direction"]
                    size = inf["position_size"]
                    candle_ret = inf["candle_return"]
                    entry_time = inf.get("entry_time_pct", 0)
                    
                    results["position_sizes"].append(size)
                    results["entry_times"].append(entry_time)
                    
                    if direction == "UP":
                        results["up_bets"] += 1
                    else:
                        results["down_bets"] += 1
                    
                    actual = "UP" if candle_ret > 0 else "DOWN"
                    if direction == actual:
                        results["correct"] += 1
                    else:
                        results["wrong"] += 1
                    
                    # Print first 20 trades
                    if results["correct"] + results["wrong"] <= 20:
                        result = "[green]✓[/green]" if direction == actual else "[red]✗[/red]"
                        console.print(
                            f"Trade {results['correct'] + results['wrong']}: "
                            f"{direction} (Size: {size:.1%}, Entry: {entry_time:.0%}) → "
                            f"{actual} ({candle_ret*100:+.2f}%) | Reward: {episode_reward:.3f} {result}"
                        )
    
    # Summary
    total_bets = results["correct"] + results["wrong"]
    
    console.print("\n" + "=" * 50)
    console.print("[bold]EVALUATION RESULTS[/bold]")
    console.print("=" * 50)
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", style="green")
    
    if total_bets > 0:
        accuracy = results["correct"] / total_bets
        table.add_row("Accuracy", f"{accuracy:.1%}")
        table.add_row("Correct / Wrong", f"{results['correct']} / {results['wrong']}")
    
    table.add_row("Skipped Candles", f"{results['skipped']}")
    table.add_row("UP / DOWN Bets", f"{results['up_bets']} / {results['down_bets']}")
    
    if results["position_sizes"]:
        table.add_row("Avg Position Size", f"{np.mean(results['position_sizes']):.1%}")
    
    if results["entry_times"]:
        table.add_row("Avg Entry Time", f"{np.mean(results['entry_times']):.0%} into candle")
    
    table.add_row("Total Reward", f"{results['total_reward']:.2f}")
    table.add_row("Avg Reward/Episode", f"{results['total_reward'] / n_episodes:.3f}")
    
    console.print(table)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="logs/enhanced_multi_asset/enhanced_model")
    parser.add_argument("--data", "-d", default="data")
    parser.add_argument("--episodes", "-n", type=int, default=100)
    
    args = parser.parse_args()
    
    evaluate_enhanced_model(
        model_path=args.model,
        data_dir=args.data,
        n_episodes=args.episodes,
    )
