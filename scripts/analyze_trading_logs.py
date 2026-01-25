#!/usr/bin/env python3
"""
Analyze trading logs from backtest and live paper trading.
Useful for feature engineering insights.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def analyze_log(entries: list, name: str):
    """Analyze a single log file."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Separate by type
    metadata = [e for e in entries if e.get('type') == 'metadata']
    ticks = [e for e in entries if e.get('type') == 'tick']
    trades = [e for e in entries if e.get('type') == 'trade']
    summary = [e for e in entries if e.get('type') == 'summary']

    print(f"\nEntries: {len(entries)} total")
    print(f"  - Metadata: {len(metadata)}")
    print(f"  - Ticks: {len(ticks)}")
    print(f"  - Trades: {len(trades)}")
    print(f"  - Summary: {len(summary)}")

    if metadata:
        m = metadata[0]
        print(f"\nConfig:")
        for k, v in m.get('config', {}).items():
            print(f"  {k}: {v}")

    if not ticks:
        print("\nNo tick data to analyze.")
        return

    # Action distribution
    actions = [t['model_output']['action_name'] for t in ticks]
    action_counts = defaultdict(int)
    for a in actions:
        action_counts[a] += 1

    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(actions) * 100
        print(f"  {action:12} {count:6} ({pct:5.1f}%)")

    # Confidence stats
    confidences = [t['model_output']['confidence'] for t in ticks]
    print(f"\nConfidence:")
    print(f"  min: {np.min(confidences):.3f}")
    print(f"  max: {np.max(confidences):.3f}")
    print(f"  mean: {np.mean(confidences):.3f}")
    print(f"  std: {np.std(confidences):.3f}")

    # Expected return stats
    exp_returns = [t['model_output']['expected_return'] for t in ticks]
    print(f"\nExpected Return:")
    print(f"  min: {np.min(exp_returns):.3f}")
    print(f"  max: {np.max(exp_returns):.3f}")
    print(f"  mean: {np.mean(exp_returns):.3f}")
    print(f"  std: {np.std(exp_returns):.3f}")

    # Q-value stats
    q_values = np.array([t['model_output']['q_values'] for t in ticks])
    print(f"\nQ-Values (5 actions):")
    for i, name in enumerate(['WAIT', 'BUY_YES', 'BUY_NO', 'EXIT', 'HOLD']):
        q_col = q_values[:, i]
        print(f"  {name:8} mean={np.mean(q_col):8.2f}  std={np.std(q_col):6.2f}  range=[{np.min(q_col):.1f}, {np.max(q_col):.1f}]")

    # Time remaining distribution when actions taken
    buy_ticks = [t for t in ticks if t['model_output']['action_name'] in ['BUY_YES', 'BUY_NO']]
    if buy_ticks:
        buy_times = [t['time_remaining'] for t in buy_ticks]
        print(f"\nTime Remaining at BUY signals ({len(buy_ticks)} signals):")
        print(f"  min: {np.min(buy_times):.3f}")
        print(f"  max: {np.max(buy_times):.3f}")
        print(f"  mean: {np.mean(buy_times):.3f}")

        # Confidence at buy signals
        buy_conf = [t['model_output']['confidence'] for t in buy_ticks]
        print(f"\nConfidence at BUY signals:")
        print(f"  min: {np.min(buy_conf):.3f}")
        print(f"  max: {np.max(buy_conf):.3f}")
        print(f"  mean: {np.mean(buy_conf):.3f}")

        # Expected return at buy signals
        buy_er = [t['model_output']['expected_return'] for t in buy_ticks]
        print(f"\nExpected Return at BUY signals:")
        print(f"  min: {np.min(buy_er):.3f}")
        print(f"  max: {np.max(buy_er):.3f}")
        print(f"  mean: {np.mean(buy_er):.3f}")

    # Price analysis
    yes_prices = [t['market']['yes_price'] for t in ticks]
    print(f"\nYES Price:")
    print(f"  min: {np.min(yes_prices):.3f}")
    print(f"  max: {np.max(yes_prices):.3f}")
    print(f"  mean: {np.mean(yes_prices):.3f}")

    # Trade analysis
    if trades:
        print(f"\n--- Trade Analysis ({len(trades)} trades) ---")

        entry_trades = [t for t in trades if t.get('action') == 'entry']
        exit_trades = [t for t in trades if t.get('action') in ['exit', 'settlement']]

        print(f"  Entries: {len(entry_trades)}")
        print(f"  Exits: {len(exit_trades)}")

        if exit_trades:
            pnls = [t.get('pnl', 0) for t in exit_trades]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p <= 0)
            print(f"\n  Win/Loss: {wins}/{losses} ({wins/(wins+losses)*100:.1f}% win rate)" if (wins+losses) > 0 else "")
            print(f"  Total PnL: ${sum(pnls):.2f}")
            print(f"  Avg PnL: ${np.mean(pnls):.2f}")

            # By reason
            by_reason = defaultdict(list)
            for t in exit_trades:
                by_reason[t.get('reason', 'unknown')].append(t.get('pnl', 0))

            print(f"\n  By Exit Reason:")
            for reason, pnl_list in by_reason.items():
                total = sum(pnl_list)
                wins = sum(1 for p in pnl_list if p > 0)
                print(f"    {reason}: {len(pnl_list)} trades, ${total:.2f} PnL, {wins}/{len(pnl_list)} wins")

    # Position state analysis
    position_ticks = [t for t in ticks if t['position']['side'] is not None]
    if position_ticks:
        print(f"\nPosition Analysis:")
        print(f"  Ticks with position: {len(position_ticks)} ({len(position_ticks)/len(ticks)*100:.1f}%)")

        yes_pos = [t for t in position_ticks if t['position']['side'] == 'yes']
        no_pos = [t for t in position_ticks if t['position']['side'] == 'no']
        print(f"  YES positions: {len(yes_pos)}")
        print(f"  NO positions: {len(no_pos)}")

    # Summary
    if summary:
        s = summary[0]
        print(f"\nFinal Summary:")
        print(f"  Total Ticks: {s.get('total_ticks', 'N/A')}")
        print(f"  Final Balance: ${s.get('final_balance', 'N/A'):.2f}" if isinstance(s.get('final_balance'), (int, float)) else f"  Final Balance: {s.get('final_balance', 'N/A')}")
        print(f"  Total PnL: ${s.get('total_pnl', 'N/A'):.2f}" if isinstance(s.get('total_pnl'), (int, float)) else f"  Total PnL: {s.get('total_pnl', 'N/A')}")
        print(f"  Wins/Losses: {s.get('wins', 0)}/{s.get('losses', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trading logs")
    parser.add_argument("--backtest", type=str, help="Path to backtest log")
    parser.add_argument("--live", type=str, help="Path to live paper trade log")
    parser.add_argument("--all", action="store_true", help="Analyze all logs in default directories")
    args = parser.parse_args()

    if args.all:
        # Find all logs
        backtest_dir = Path("./logs/backtest_unified")
        live_dir = Path("./logs/paper_trade_unified")

        if backtest_dir.exists():
            for log_file in sorted(backtest_dir.glob("*.jsonl")):
                entries = load_jsonl(log_file)
                analyze_log(entries, f"BACKTEST: {log_file.name}")

        if live_dir.exists():
            for log_file in sorted(live_dir.glob("*.jsonl")):
                entries = load_jsonl(log_file)
                analyze_log(entries, f"LIVE: {log_file.name}")

    else:
        if args.backtest:
            entries = load_jsonl(Path(args.backtest))
            analyze_log(entries, f"BACKTEST: {args.backtest}")

        if args.live:
            entries = load_jsonl(Path(args.live))
            analyze_log(entries, f"LIVE: {args.live}")

        if not args.backtest and not args.live:
            print("Usage: python scripts/analyze_trading_logs.py --all")
            print("   or: python scripts/analyze_trading_logs.py --backtest <path> --live <path>")


if __name__ == "__main__":
    main()
