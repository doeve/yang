#!/usr/bin/env python3
"""
Generate DeepLOB Predictions Dataset.

Creates a dataset of real DeepLOB predictions aligned to Polymarket candle outcomes.
This is used for unbiased SAC training with real model predictions.
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

# Import DeepLOB components
from src.inference.deep_lob_inference import DeepLOBTwoLayerBot
from src.features.deep_lob_features import DeepLOBFeatureBuilder

console = Console()


def fetch_btc_data_for_candle(
    candle_timestamp: int,
    btc_data: pd.DataFrame,
    lookback_seconds: int = 180,  # 3 minutes of data
) -> pd.DataFrame:
    """
    Get BTC 1s data before a candle end time.
    
    Args:
        candle_timestamp: Unix timestamp of candle start
        btc_data: Full BTC 1s dataframe
        lookback_seconds: Seconds of history to use
        
    Returns:
        DataFrame with BTC data for feature engineering
    """
    # Candle ends 15 minutes after start
    candle_end = candle_timestamp + 900  # +15 min
    candle_end_dt = pd.Timestamp(candle_end, unit='s', tz='UTC')
    
    # Make btc timestamps tz-aware for comparison
    if btc_data['timestamp'].dt.tz is None:
        btc_data = btc_data.copy()
        btc_data['timestamp'] = btc_data['timestamp'].dt.tz_localize('UTC')
    
    # Get data from lookback period
    start_time = candle_end_dt - timedelta(seconds=lookback_seconds)
    mask = (btc_data['timestamp'] >= start_time) & (btc_data['timestamp'] <= candle_end_dt)
    
    return btc_data[mask].copy()


def generate_predictions_dataset(
    polymarket_path: str = "./data/polymarket/btc_15min_candles.parquet",
    btc_path: str = "./data/btcusdt_1s_30days.parquet",
    deep_lob_model: str = "./logs/deep_lob_balanced",
    output_path: str = "./data/predictions_dataset.parquet",
    min_btc_rows: int = 60,  # Minimum BTC rows needed
):
    """Generate DeepLOB predictions for all historical candles."""
    
    console.print("[bold blue]Generating DeepLOB Predictions Dataset[/bold blue]")
    console.print()
    
    # Load data
    console.print("[dim]Loading data...[/dim]")
    poly = pd.read_parquet(polymarket_path)
    btc = pd.read_parquet(btc_path)
    
    console.print(f"  Polymarket candles: {len(poly)}")
    console.print(f"  BTC 1s rows: {len(btc)}")
    
    # Filter for closed candles
    poly = poly[poly['closed'] == True].reset_index(drop=True)
    console.print(f"  Closed candles: {len(poly)}")
    
    # Initialize DeepLOB bot
    console.print("[dim]Loading DeepLOB model...[/dim]")
    bot = DeepLOBTwoLayerBot()
    bot.load_models(deep_lob_model)
    
    feature_builder = DeepLOBFeatureBuilder()
    
    # Generate predictions
    predictions = []
    skipped = 0
    
    console.print("[bold]Generating predictions...[/bold]")
    
    for idx in track(range(len(poly)), description="Processing candles"):
        candle = poly.iloc[idx]
        
        # Get BTC data for this candle
        btc_slice = fetch_btc_data_for_candle(
            candle['timestamp'],
            btc,
            lookback_seconds=180,
        )
        
        if len(btc_slice) < min_btc_rows:
            skipped += 1
            continue
        
        try:
            # Build features
            # Ensure btc_slice has required columns
            if 'price' not in btc_slice.columns:
                btc_slice['price'] = btc_slice['close']
            if 'buy_pressure' not in btc_slice.columns:
                btc_slice['buy_pressure'] = 0.5  # Default
            
            # Get DeepLOB predictions
            probs = bot.predict_class_probabilities(btc_slice)
            
            predictions.append({
                'timestamp': candle['timestamp'],
                'datetime': candle['datetime'],
                'prob_down': probs['down'],
                'prob_hold': probs['hold'],
                'prob_up': probs['up'],
                'predicted_class': probs['predicted_class'],
                'up_won': candle['up_won'],
                'volume': candle['volume'],
            })
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed on candle {idx}: {e}[/yellow]")
            skipped += 1
            continue
    
    console.print()
    console.print(f"Generated {len(predictions)} predictions, skipped {skipped}")
    
    if len(predictions) == 0:
        console.print("[red]No predictions generated! Check data overlap.[/red]")
        return
    
    # Save to parquet
    df = pd.DataFrame(predictions)
    df.to_parquet(output_path, index=False)
    
    console.print(f"[green]✓ Saved to {output_path}[/green]")
    console.print()
    
    # Stats
    console.print("[bold]Dataset Statistics:[/bold]")
    console.print(f"  Total predictions: {len(df)}")
    console.print(f"  Up won: {df['up_won'].sum()} ({df['up_won'].mean()*100:.1f}%)")
    console.print(f"  Down won: {(~df['up_won']).sum()} ({(~df['up_won']).mean()*100:.1f}%)")
    
    # Check prediction accuracy
    correct = ((df['predicted_class'] == 2) & df['up_won']) | \
              ((df['predicted_class'] == 0) & ~df['up_won'])
    console.print(f"  DeepLOB accuracy: {correct.mean()*100:.1f}%")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DeepLOB predictions dataset")
    parser.add_argument("--polymarket", default="./data/polymarket/btc_15min_candles.parquet")
    parser.add_argument("--btc", default="./data/btcusdt_1s_30days.parquet")
    parser.add_argument("--model", default="./logs/deep_lob_balanced")
    parser.add_argument("--output", default="./data/predictions_dataset.parquet")
    
    args = parser.parse_args()
    
    generate_predictions_dataset(
        polymarket_path=args.polymarket,
        btc_path=args.btc,
        deep_lob_model=args.model,
        output_path=args.output,
    )
