#!/usr/bin/env python3
"""
Feedback Extractor for Paper Trading Logs.

Analyzes trading logs to identify:
1. Missed Opportunities - When market moved favorably but bot did nothing
2. Bad Decisions - Poor entries, late exits, or negative expectancy trades

Outputs a compact dataset for feature engineering refinement.

Usage:
    python scripts/extract_feedback.py logs/paper_trade_unified/market_predictor_v1_20260125_191601.jsonl
"""

import json
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
from collections import defaultdict
import csv


# =============================================================================
# Definitions
# =============================================================================

"""
MISSED OPPORTUNITY Definition:
A tick where the bot chose WAIT (action=0) but in hindsight there was clear profit:
- YES opportunity: yes_price was low and later settled at 1.0 (BTC went up)
- NO opportunity: no_price was low and later settled at 1.0 (BTC went down)

We detect this by looking at:
1. WAIT action during a candle
2. Final outcome of that candle (BTC up or down)
3. Price at WAIT decision vs potential profit at settlement

Thresholds:
- Minimum missed profit: 15% potential return if entered at that price
- Minimum time remaining: 20% (don't flag near-settlement WAITs)


BAD DECISION Definition:
A trade that resulted in loss AND had warning signals:
1. Entry at poor price (price already moved significantly toward expected direction)
2. Late entry (entered in last 30% of candle with low confidence)
3. Premature exit (exited too early while price was improving)
4. Negative expectancy trade (expected_return was negative but still entered)

We detect this by examining:
- Entry tick vs last tick of that candle
- Price trajectory after entry
- Model confidence and expected_return at entry
"""


@dataclass
class CandleTracker:
    """Track data for a single candle to detect outcomes."""
    candle_ts: int
    btc_open: Optional[float] = None
    btc_close: Optional[float] = None
    outcome: Optional[str] = None  # "up" or "down"
    
    # Track best prices seen during candle (for missed opportunity detection)
    min_yes_price: float = 1.0
    max_yes_price: float = 0.0
    min_no_price: float = 1.0
    max_no_price: float = 0.0
    
    # All ticks for this candle
    ticks: List[Dict] = field(default_factory=list)
    
    # Trade events
    entries: List[Dict] = field(default_factory=list)
    exits: List[Dict] = field(default_factory=list)


@dataclass
class MissedOpportunity:
    """A missed trading opportunity."""
    candle_ts: int
    tick: int
    timestamp: str
    time_remaining: float
    
    # What was missed
    opportunity_type: str  # "YES" or "NO"
    price_at_decision: float
    settlement_payout: float  # 1.0 if correct side
    potential_return_pct: float
    
    # Model state at decision
    action_taken: str  # Should be "WAIT"
    confidence: float
    expected_return: float
    q_values: List[float]
    
    # Market context
    yes_price: float
    no_price: float
    btc_price: float
    btc_open: float
    btc_change_pct: float
    
    # Features for training
    position_state: List[float]


@dataclass 
class BadDecision:
    """A trade that resulted in poor outcome with warning signs."""
    candle_ts: int
    tick: int
    timestamp: str
    
    # Decision details
    decision_type: str  # "poor_entry", "late_entry", "premature_exit", "negative_expectancy"
    side: str  # "yes" or "no"
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    
    # Warning signs at entry
    confidence: float
    expected_return: float
    time_remaining_at_entry: float
    
    # Why it was bad
    reason: str
    
    # Price trajectory
    best_price_after_entry: float
    worst_price_after_entry: float
    price_at_settlement: float
    
    # Features for training
    q_values: List[float]
    position_state: List[float]


@dataclass
class FeedbackSummary:
    """Summary of feedback extraction."""
    total_ticks: int
    total_candles: int
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    
    missed_opportunities_count: int
    missed_opportunities_potential_profit: float
    
    bad_decisions_count: int
    bad_decisions_loss: float
    
    # Categorized
    poor_entries: int
    late_entries: int
    premature_exits: int
    negative_expectancy_entries: int


# =============================================================================
# Log Parser (Streaming)
# =============================================================================

def stream_jsonl(filepath: str) -> Iterator[Dict]:
    """Stream JSONL file line by line without loading into memory."""
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def group_ticks_by_candle(filepath: str) -> Iterator[CandleTracker]:
    """Group ticks by candle and yield complete candles."""
    current_candle: Optional[CandleTracker] = None
    
    for entry in stream_jsonl(filepath):
        if entry.get("type") != "tick":
            continue
            
        candle_ts = entry.get("candle_ts")
        if candle_ts is None:
            continue
        
        # New candle started
        if current_candle is None or current_candle.candle_ts != candle_ts:
            # Yield previous candle if complete
            if current_candle is not None and len(current_candle.ticks) > 0:
                # Determine outcome from BTC movement
                if current_candle.btc_open and current_candle.btc_close:
                    current_candle.outcome = "up" if current_candle.btc_close > current_candle.btc_open else "down"
                yield current_candle
            
            # Start new candle
            current_candle = CandleTracker(candle_ts=candle_ts)
        
        # Update candle data
        market = entry.get("market", {})
        yes_price = market.get("yes_price", 0.5)
        no_price = market.get("no_price", 0.5)
        btc_price = market.get("btc_price")
        btc_open = market.get("btc_open")
        
        if btc_open and current_candle.btc_open is None:
            current_candle.btc_open = btc_open
        if btc_price:
            current_candle.btc_close = btc_price
            
        current_candle.min_yes_price = min(current_candle.min_yes_price, yes_price)
        current_candle.max_yes_price = max(current_candle.max_yes_price, yes_price)
        current_candle.min_no_price = min(current_candle.min_no_price, no_price)
        current_candle.max_no_price = max(current_candle.max_no_price, no_price)
        
        # Track position changes for entry/exit detection
        position = entry.get("position", {})
        model_output = entry.get("model_output", {})
        
        current_candle.ticks.append({
            "tick": entry.get("tick"),
            "timestamp": entry.get("timestamp"),
            "time_remaining": entry.get("time_remaining", 0),
            "yes_price": yes_price,
            "no_price": no_price,
            "btc_price": btc_price,
            "btc_open": btc_open,
            "action": model_output.get("action", 0),
            "action_name": model_output.get("action_name", "WAIT"),
            "confidence": model_output.get("confidence", 0),
            "expected_return": model_output.get("expected_return", 0),
            "q_values": model_output.get("q_values", []),
            "position_side": position.get("side"),
            "position_size": position.get("size", 0),
            "entry_price": position.get("entry_price", 0),
            "position_state": entry.get("position_state", []),
        })
    
    # Yield last candle
    if current_candle is not None and len(current_candle.ticks) > 0:
        if current_candle.btc_open and current_candle.btc_close:
            current_candle.outcome = "up" if current_candle.btc_close > current_candle.btc_open else "down"
        yield current_candle


# =============================================================================
# Opportunity Detection
# =============================================================================

def detect_missed_opportunities(
    candle: CandleTracker,
    min_potential_return: float = 0.15,
    min_time_remaining: float = 0.20,
) -> List[MissedOpportunity]:
    """
    Detect the best missed trading opportunity in a candle.
    
    Returns at most ONE opportunity per candle - the one with highest potential return
    where the bot chose WAIT.
    """
    if candle.outcome is None:
        return []
    
    winning_side = "yes" if candle.outcome == "up" else "no"
    
    best_opportunity = None
    best_potential_return = 0.0
    
    for tick in candle.ticks:
        # Only look at WAIT decisions
        if tick["action_name"] != "WAIT":
            continue
            
        # Skip near-settlement ticks
        if tick["time_remaining"] < min_time_remaining:
            continue
            
        # Calculate potential return for the winning side
        if winning_side == "yes":
            entry_price = tick["yes_price"]
        else:
            entry_price = tick["no_price"]
        
        potential_return = (1.0 - entry_price) / entry_price if entry_price > 0 else 0
        
        # Track the best opportunity (lowest entry price = highest return)
        if potential_return >= min_potential_return and potential_return > best_potential_return:
            btc_change = 0
            if tick["btc_open"] and tick["btc_price"]:
                btc_change = (tick["btc_price"] - tick["btc_open"]) / tick["btc_open"] * 100
            
            best_opportunity = MissedOpportunity(
                candle_ts=candle.candle_ts,
                tick=tick["tick"],
                timestamp=tick["timestamp"],
                time_remaining=tick["time_remaining"],
                opportunity_type=winning_side.upper(),
                price_at_decision=entry_price,
                settlement_payout=1.0,
                potential_return_pct=potential_return * 100,
                action_taken="WAIT",
                confidence=tick["confidence"],
                expected_return=tick["expected_return"],
                q_values=tick["q_values"],
                yes_price=tick["yes_price"],
                no_price=tick["no_price"],
                btc_price=tick["btc_price"] or 0,
                btc_open=tick["btc_open"] or 0,
                btc_change_pct=btc_change,
                position_state=tick["position_state"],
            )
            best_potential_return = potential_return
    
    return [best_opportunity] if best_opportunity else []


def detect_bad_decisions(
    candle: CandleTracker,
    late_entry_threshold: float = 0.30,
) -> List[BadDecision]:
    """Detect bad trading decisions in a candle."""
    decisions = []
    
    if candle.outcome is None:
        return decisions
    
    # Find trades by detecting position state changes
    prev_position = None
    entry_tick = None
    
    for i, tick in enumerate(candle.ticks):
        current_side = tick["position_side"]
        
        # Detect entry
        if prev_position is None and current_side is not None:
            entry_tick = tick
        
        # Detect exit
        if prev_position is not None and current_side is None and entry_tick is not None:
            # Calculate trade outcome
            side = prev_position
            entry_price = entry_tick["entry_price"]
            
            # Determine settlement price
            if side == "yes":
                settlement_payout = 1.0 if candle.outcome == "up" else 0.0
            else:
                settlement_payout = 1.0 if candle.outcome == "down" else 0.0
            
            # Find best/worst prices after entry
            subsequent_ticks = candle.ticks[candle.ticks.index(entry_tick):]
            if side == "yes":
                prices = [t["yes_price"] for t in subsequent_ticks]
            else:
                prices = [t["no_price"] for t in subsequent_ticks]
            
            best_price = max(prices) if prices else entry_price
            worst_price = min(prices) if prices else entry_price
            
            # Calculate PnL (simplified)
            exit_price = tick[f"{side}_price"]
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Determine if this was a bad decision and why
            bad_decision = None
            reason = None
            
            # 1. Negative expectancy entry
            if entry_tick["expected_return"] < 0:
                bad_decision = "negative_expectancy"
                reason = f"Entered with negative expected return: {entry_tick['expected_return']:.2%}"
            
            # 2. Late entry with low confidence
            elif entry_tick["time_remaining"] < late_entry_threshold and entry_tick["confidence"] < 0.5:
                bad_decision = "late_entry"
                reason = f"Late entry at {entry_tick['time_remaining']:.1%} remaining with {entry_tick['confidence']:.1%} confidence"
            
            # 3. Poor entry (price already moved significantly)
            elif side == "yes" and entry_tick["yes_price"] > 0.70:
                bad_decision = "poor_entry"
                reason = f"Bought YES at high price {entry_tick['yes_price']:.3f}"
            elif side == "no" and entry_tick["no_price"] > 0.70:
                bad_decision = "poor_entry"
                reason = f"Bought NO at high price {entry_tick['no_price']:.3f}"
            
            # 4. Premature exit (exited while price was improving and would have won)
            elif pnl_pct < 0 and settlement_payout == 1.0:
                bad_decision = "premature_exit"
                reason = f"Exited at {exit_price:.3f} but would have won at settlement"
            
            if bad_decision:
                decisions.append(BadDecision(
                    candle_ts=candle.candle_ts,
                    tick=entry_tick["tick"],
                    timestamp=entry_tick["timestamp"],
                    decision_type=bad_decision,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl_pct * 100,  # Store as percentage
                    pnl_pct=pnl_pct,
                    confidence=entry_tick["confidence"],
                    expected_return=entry_tick["expected_return"],
                    time_remaining_at_entry=entry_tick["time_remaining"],
                    reason=reason,
                    best_price_after_entry=best_price,
                    worst_price_after_entry=worst_price,
                    price_at_settlement=settlement_payout,
                    q_values=entry_tick["q_values"],
                    position_state=entry_tick["position_state"],
                ))
            
            entry_tick = None
        
        prev_position = current_side
    
    return decisions


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_log(
    filepath: str,
    min_missed_return: float = 0.15,
    min_time_remaining: float = 0.20,
) -> tuple[List[MissedOpportunity], List[BadDecision], FeedbackSummary]:
    """Analyze log file and extract feedback signals."""
    
    missed_opportunities = []
    bad_decisions = []
    
    total_ticks = 0
    total_candles = 0
    final_account = None
    
    print(f"Analyzing {filepath}...")
    
    for candle in group_ticks_by_candle(filepath):
        total_candles += 1
        total_ticks += len(candle.ticks)
        
        # Detect missed opportunities
        missed = detect_missed_opportunities(candle, min_missed_return, min_time_remaining)
        missed_opportunities.extend(missed)
        
        # Detect bad decisions
        bad = detect_bad_decisions(candle)
        bad_decisions.extend(bad)
        
        if total_candles % 50 == 0:
            print(f"  Processed {total_candles} candles, {total_ticks} ticks...")
    
    # Get final stats by reading last few lines
    final_wins = 0
    final_losses = 0
    final_pnl = 0.0
    
    for entry in stream_jsonl(filepath):
        if entry.get("type") == "tick":
            account = entry.get("account", {})
            final_wins = account.get("wins", 0)
            final_losses = account.get("losses", 0)
            final_pnl = account.get("total_pnl", 0)
    
    # Categorize bad decisions
    poor_entries = sum(1 for d in bad_decisions if d.decision_type == "poor_entry")
    late_entries = sum(1 for d in bad_decisions if d.decision_type == "late_entry")
    premature_exits = sum(1 for d in bad_decisions if d.decision_type == "premature_exit")
    negative_expectancy = sum(1 for d in bad_decisions if d.decision_type == "negative_expectancy")
    
    summary = FeedbackSummary(
        total_ticks=total_ticks,
        total_candles=total_candles,
        total_trades=final_wins + final_losses,
        wins=final_wins,
        losses=final_losses,
        total_pnl=final_pnl,
        missed_opportunities_count=len(missed_opportunities),
        missed_opportunities_potential_profit=sum(m.potential_return_pct for m in missed_opportunities),
        bad_decisions_count=len(bad_decisions),
        bad_decisions_loss=sum(d.pnl for d in bad_decisions if d.pnl < 0),
        poor_entries=poor_entries,
        late_entries=late_entries,
        premature_exits=premature_exits,
        negative_expectancy_entries=negative_expectancy,
    )
    
    return missed_opportunities, bad_decisions, summary


def save_results(
    missed: List[MissedOpportunity],
    bad: List[BadDecision],
    summary: FeedbackSummary,
    output_dir: str,
):
    """Save results to JSON and CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_file = output_path / f"feedback_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    # Save missed opportunities
    if missed:
        missed_file = output_path / f"missed_opportunities_{timestamp}.json"
        with open(missed_file, 'w') as f:
            json.dump([asdict(m) for m in missed], f, indent=2)
        print(f"Missed opportunities ({len(missed)}) saved to: {missed_file}")
        
        # Also save as CSV for easy analysis
        missed_csv = output_path / f"missed_opportunities_{timestamp}.csv"
        with open(missed_csv, 'w', newline='') as f:
            if missed:
                writer = csv.DictWriter(f, fieldnames=[
                    "candle_ts", "tick", "time_remaining", "opportunity_type",
                    "price_at_decision", "potential_return_pct", "confidence",
                    "expected_return", "btc_change_pct"
                ])
                writer.writeheader()
                for m in missed:
                    writer.writerow({
                        "candle_ts": m.candle_ts,
                        "tick": m.tick,
                        "time_remaining": f"{m.time_remaining:.2%}",
                        "opportunity_type": m.opportunity_type,
                        "price_at_decision": f"{m.price_at_decision:.3f}",
                        "potential_return_pct": f"{m.potential_return_pct:.1f}%",
                        "confidence": f"{m.confidence:.2%}",
                        "expected_return": f"{m.expected_return:.2%}",
                        "btc_change_pct": f"{m.btc_change_pct:.3f}%",
                    })
        print(f"Missed opportunities CSV: {missed_csv}")
    
    # Save bad decisions
    if bad:
        bad_file = output_path / f"bad_decisions_{timestamp}.json"
        with open(bad_file, 'w') as f:
            json.dump([asdict(b) for b in bad], f, indent=2)
        print(f"Bad decisions ({len(bad)}) saved to: {bad_file}")
        
        # Also save as CSV
        bad_csv = output_path / f"bad_decisions_{timestamp}.csv"
        with open(bad_csv, 'w', newline='') as f:
            if bad:
                writer = csv.DictWriter(f, fieldnames=[
                    "candle_ts", "tick", "decision_type", "side", "reason",
                    "entry_price", "exit_price", "pnl", "confidence",
                    "expected_return", "time_remaining_at_entry"
                ])
                writer.writeheader()
                for b in bad:
                    writer.writerow({
                        "candle_ts": b.candle_ts,
                        "tick": b.tick,
                        "decision_type": b.decision_type,
                        "side": b.side,
                        "reason": b.reason,
                        "entry_price": f"{b.entry_price:.3f}",
                        "exit_price": f"{b.exit_price:.3f}",
                        "pnl": f"{b.pnl:.1f}%",
                        "confidence": f"{b.confidence:.2%}",
                        "expected_return": f"{b.expected_return:.2%}",
                        "time_remaining_at_entry": f"{b.time_remaining_at_entry:.2%}",
                    })
        print(f"Bad decisions CSV: {bad_csv}")


def print_summary(summary: FeedbackSummary):
    """Print analysis summary."""
    print("\n" + "="*60)
    print("FEEDBACK EXTRACTION SUMMARY")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total ticks: {summary.total_ticks:,}")
    print(f"   Total candles: {summary.total_candles}")
    print(f"   Total trades: {summary.total_trades} ({summary.wins}W / {summary.losses}L)")
    print(f"   Total PnL: ${summary.total_pnl:,.2f}")
    
    print(f"\n🎯 Missed Opportunities:")
    print(f"   Count: {summary.missed_opportunities_count}")
    print(f"   Total forgone potential: {summary.missed_opportunities_potential_profit:.1f}%")
    
    print(f"\n⚠️  Bad Decisions:")
    print(f"   Count: {summary.bad_decisions_count}")
    print(f"   - Poor entries: {summary.poor_entries}")
    print(f"   - Late entries: {summary.late_entries}")
    print(f"   - Premature exits: {summary.premature_exits}")
    print(f"   - Negative expectancy: {summary.negative_expectancy_entries}")
    print(f"   Total loss from bad decisions: {summary.bad_decisions_loss:.1f}%")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract feedback signals from paper trading logs for feature engineering"
    )
    parser.add_argument(
        "logfile",
        type=str,
        help="Path to JSONL log file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/feedback",
        help="Output directory for results"
    )
    parser.add_argument(
        "--min-missed-return",
        type=float,
        default=0.15,
        help="Minimum potential return to flag as missed opportunity (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--min-time-remaining",
        type=float,
        default=0.20,
        help="Minimum time remaining to consider for missed opportunities (default: 0.20 = 20%%)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.logfile).exists():
        print(f"Error: Log file not found: {args.logfile}")
        return
    
    missed, bad, summary = analyze_log(
        args.logfile,
        min_missed_return=args.min_missed_return,
        min_time_remaining=args.min_time_remaining,
    )
    
    print_summary(summary)
    save_results(missed, bad, summary, args.output_dir)
    
    print(f"\n✅ Done! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
