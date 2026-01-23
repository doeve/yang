"""
Log Analyzer for Paper Trade Sessions.

Analyzes JSONL logs from paper trading to identify inefficiencies such as:
- High confidence trades that lose money
- Position churning (rapid entry/exit)
- Poor edge detection accuracy
- Suboptimal trade timing
- SAC action analysis
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class Trade:
    """Represents a completed trade (entry + exit)."""
    entry_tick: int
    exit_tick: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    invested: float
    edge_at_entry: float
    edge_at_exit: float
    p_yes_at_entry: float
    confidence_at_entry: float
    time_remaining_at_entry: float
    time_remaining_at_exit: float
    btc_price_entry: float
    btc_price_exit: float
    hold_duration_ticks: int = field(init=False)

    def __post_init__(self):
        self.hold_duration_ticks = self.exit_tick - self.entry_tick


@dataclass
class LogAnalysis:
    """Complete analysis of a paper trade log."""
    log_path: str
    model_name: str
    config: dict

    # Basic stats
    total_ticks: int
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    total_pnl: float
    final_balance: float
    initial_balance: float

    # Trade details
    trades: list[Trade]

    # Tick-level data for analysis
    tick_data: list[dict]

    # Computed inefficiencies
    inefficiencies: list[dict] = field(default_factory=list)


def parse_log(log_path: Path) -> LogAnalysis:
    """Parse a paper trade log file."""
    entries = []
    with open(log_path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Extract metadata
    metadata = entries[0]
    config = metadata.get("config", {})
    model_name = metadata.get("model_name", "unknown")

    # Separate ticks and trades
    ticks = [e for e in entries if e.get("type") == "tick"]
    trade_events = [e for e in entries if e.get("type") == "trade"]

    # Pair up entries and exits into complete trades
    trades = []
    pending_entry = None

    for event in trade_events:
        if event["action"] == "entry":
            pending_entry = event
        elif event["action"] == "exit" and pending_entry:
            # Find the entry tick data
            entry_tick_data = next(
                (t for t in ticks if t["tick"] == pending_entry["tick"]),
                None
            )

            trades.append(Trade(
                entry_tick=pending_entry["tick"],
                exit_tick=event["tick"],
                side=pending_entry["side"],
                entry_price=pending_entry["price"],
                exit_price=event["exit_price"],
                pnl=event["pnl"],
                invested=event["invested"],
                edge_at_entry=pending_entry["edge"],
                edge_at_exit=event["edge_at_exit"],
                p_yes_at_entry=pending_entry["p_yes"],
                confidence_at_entry=pending_entry["confidence"],
                time_remaining_at_entry=pending_entry["time_remaining"],
                time_remaining_at_exit=event["time_remaining"],
                btc_price_entry=pending_entry["btc_price"],
                btc_price_exit=entry_tick_data.get("market", {}).get("btc_price", 0) if entry_tick_data else 0,
            ))
            pending_entry = None

    # Calculate stats
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    breakeven = sum(1 for t in trades if t.pnl == 0)
    total_pnl = sum(t.pnl for t in trades)

    final_tick = ticks[-1] if ticks else {}
    final_balance = final_tick.get("account", {}).get("balance", config.get("initial_balance", 1000))
    initial_balance = config.get("initial_balance", 1000)

    return LogAnalysis(
        log_path=str(log_path),
        model_name=model_name,
        config=config,
        total_ticks=len(ticks),
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        total_pnl=total_pnl,
        final_balance=final_balance,
        initial_balance=initial_balance,
        trades=trades,
        tick_data=ticks,
    )


def analyze_inefficiencies(analysis: LogAnalysis) -> list[dict]:
    """Identify trading inefficiencies from the log analysis."""
    inefficiencies = []

    # 1. High confidence losses
    high_conf_losses = [
        t for t in analysis.trades
        if t.pnl < 0 and t.confidence_at_entry > 0.7
    ]
    if high_conf_losses:
        total_loss = sum(t.pnl for t in high_conf_losses)
        inefficiencies.append({
            "type": "HIGH_CONFIDENCE_LOSSES",
            "severity": "critical",
            "description": f"{len(high_conf_losses)} trades with confidence >70% resulted in losses",
            "impact": f"${abs(total_loss):.2f} lost",
            "details": [
                {
                    "tick": t.entry_tick,
                    "confidence": f"{t.confidence_at_entry:.1%}",
                    "edge": f"{t.edge_at_entry:.3f}",
                    "pnl": f"${t.pnl:.2f}",
                    "p_yes": f"{t.p_yes_at_entry:.1%}",
                }
                for t in high_conf_losses
            ],
            "recommendation": "Edge detector may be overconfident. Consider recalibrating or adding ensemble methods.",
        })

    # 2. Position churning (entry/exit within 5 ticks)
    churned = [t for t in analysis.trades if t.hold_duration_ticks <= 5]
    if churned:
        avg_pnl = sum(t.pnl for t in churned) / len(churned)
        inefficiencies.append({
            "type": "POSITION_CHURNING",
            "severity": "high" if avg_pnl < 0 else "medium",
            "description": f"{len(churned)} trades held for <=5 ticks (rapid exit)",
            "impact": f"Avg PnL: ${avg_pnl:.2f} per churned trade",
            "details": [
                {
                    "entry_tick": t.entry_tick,
                    "hold_ticks": t.hold_duration_ticks,
                    "pnl": f"${t.pnl:.2f}",
                    "edge_change": f"{t.edge_at_entry:.3f} -> {t.edge_at_exit:.3f}",
                }
                for t in churned[:10]  # Show first 10
            ],
            "recommendation": "Consider adding minimum hold time or exit signal smoothing.",
        })

    # 3. Large single-trade losses
    big_losses = [t for t in analysis.trades if t.pnl < -50]
    if big_losses:
        inefficiencies.append({
            "type": "CATASTROPHIC_LOSSES",
            "severity": "critical",
            "description": f"{len(big_losses)} trades lost more than $50 each",
            "impact": f"${abs(sum(t.pnl for t in big_losses)):.2f} total",
            "details": [
                {
                    "tick": t.entry_tick,
                    "pnl": f"${t.pnl:.2f}",
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "time_remaining": f"{t.time_remaining_at_entry:.1%}",
                    "edge": f"{t.edge_at_entry:.3f}",
                }
                for t in big_losses
            ],
            "recommendation": "Add stop-loss or reduce position size when time_remaining is low.",
        })

    # 4. Edge reversal (edge positive at entry, negative at exit or vice versa)
    edge_reversals = [
        t for t in analysis.trades
        if (t.edge_at_entry > 0 and t.edge_at_exit < 0) or (t.edge_at_entry < 0 and t.edge_at_exit > 0)
    ]
    if edge_reversals:
        inefficiencies.append({
            "type": "EDGE_REVERSAL",
            "severity": "medium",
            "description": f"{len(edge_reversals)} trades where edge reversed sign during hold",
            "impact": f"Avg PnL: ${sum(t.pnl for t in edge_reversals) / len(edge_reversals):.2f}",
            "details": [
                {
                    "tick": t.entry_tick,
                    "edge_entry": f"{t.edge_at_entry:.3f}",
                    "edge_exit": f"{t.edge_at_exit:.3f}",
                    "pnl": f"${t.pnl:.2f}",
                }
                for t in edge_reversals[:10]
            ],
            "recommendation": "Edge signal may be unstable. Consider using EMA smoothing on edge.",
        })

    # 5. Late entries (entering when time_remaining < 20%)
    late_entries = [t for t in analysis.trades if t.time_remaining_at_entry < 0.2]
    if late_entries:
        avg_pnl = sum(t.pnl for t in late_entries) / len(late_entries)
        inefficiencies.append({
            "type": "LATE_ENTRIES",
            "severity": "medium" if avg_pnl >= 0 else "high",
            "description": f"{len(late_entries)} trades entered with <20% time remaining",
            "impact": f"Avg PnL: ${avg_pnl:.2f}",
            "details": [
                {
                    "tick": t.entry_tick,
                    "time_remaining": f"{t.time_remaining_at_entry:.1%}",
                    "pnl": f"${t.pnl:.2f}",
                    "edge": f"{t.edge_at_entry:.3f}",
                }
                for t in late_entries[:10]
            ],
            "recommendation": "Late entries have less time to recover. Consider time-based position sizing.",
        })

    # 6. SAC action analysis - check if SAC is ignored
    sac_ignored_count = 0
    for tick in analysis.tick_data:
        sac = tick.get("sac_action", {})
        position = tick.get("position", {})
        hold_prob = sac.get("hold_prob", 0)
        has_position = position.get("side") is not None

        # If hold_prob > 0.7 but we have a position, SAC was ignored
        if hold_prob > 0.7 and has_position:
            sac_ignored_count += 1

    if sac_ignored_count > 0:
        inefficiencies.append({
            "type": "SAC_IGNORED",
            "severity": "medium",
            "description": f"SAC recommended hold (>70% prob) but position was open in {sac_ignored_count} ticks",
            "impact": "SAC risk management signals not being used",
            "recommendation": "Consider incorporating SAC hold_prob into exit decisions.",
        })

    # 7. Win rate analysis
    if analysis.total_trades > 0:
        win_rate = analysis.wins / analysis.total_trades
        if win_rate < 0.4:
            inefficiencies.append({
                "type": "LOW_WIN_RATE",
                "severity": "critical",
                "description": f"Win rate is only {win_rate:.1%} ({analysis.wins}/{analysis.total_trades})",
                "impact": f"Net PnL: ${analysis.total_pnl:.2f}",
                "recommendation": "Edge detector accuracy may be poor. Check p_yes calibration against outcomes.",
            })

    # 8. Confidence vs Edge mismatch
    conf_edge_mismatch = [
        t for t in analysis.trades
        if t.confidence_at_entry > 0.8 and abs(t.edge_at_entry) < 0.1
    ]
    if conf_edge_mismatch:
        inefficiencies.append({
            "type": "CONFIDENCE_EDGE_MISMATCH",
            "severity": "medium",
            "description": f"{len(conf_edge_mismatch)} trades with high confidence (>80%) but low edge (<10%)",
            "impact": "Model may be confident about wrong predictions",
            "details": [
                {
                    "tick": t.entry_tick,
                    "confidence": f"{t.confidence_at_entry:.1%}",
                    "edge": f"{t.edge_at_entry:.3f}",
                    "pnl": f"${t.pnl:.2f}",
                }
                for t in conf_edge_mismatch[:5]
            ],
            "recommendation": "Confidence and edge should correlate. Check feature engineering.",
        })

    # 9. Price movement analysis
    if analysis.tick_data:
        yes_prices = [t.get("market", {}).get("yes_price", 0.5) for t in analysis.tick_data]
        if len(yes_prices) > 10:
            price_volatility = max(yes_prices) - min(yes_prices)
            if price_volatility > 0.5:
                inefficiencies.append({
                    "type": "HIGH_PRICE_VOLATILITY",
                    "severity": "info",
                    "description": f"YES price ranged from {min(yes_prices):.3f} to {max(yes_prices):.3f}",
                    "impact": f"Price swing: {price_volatility:.3f}",
                    "recommendation": "Consider volatility-adjusted position sizing.",
                })

    # 10. Unrealized opportunities (high edge but no trade)
    missed_opportunities = []
    for i, tick in enumerate(analysis.tick_data):
        edge = tick.get("edge_detector", {}).get("edge", 0)
        conf = tick.get("edge_detector", {}).get("confidence", 0)
        position = tick.get("position", {})
        has_position = position.get("side") is not None

        if abs(edge) > 0.15 and conf > 0.7 and not has_position:
            missed_opportunities.append({
                "tick": tick.get("tick"),
                "edge": edge,
                "confidence": conf,
            })

    if len(missed_opportunities) > 5:
        inefficiencies.append({
            "type": "MISSED_OPPORTUNITIES",
            "severity": "medium",
            "description": f"{len(missed_opportunities)} ticks with edge >15% and confidence >70% had no position",
            "impact": "Potential profits missed",
            "details": missed_opportunities[:10],
            "recommendation": "Check min_edge_to_trade and min_confidence thresholds.",
        })

    analysis.inefficiencies = inefficiencies
    return inefficiencies


def print_analysis(analysis: LogAnalysis, console: Console):
    """Print formatted analysis to console."""
    console.print()
    console.print(Panel(f"[bold]Log Analysis: {Path(analysis.log_path).name}[/bold]"))

    # Summary table
    summary = Table(title="Session Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")

    summary.add_row("Model", analysis.model_name)
    summary.add_row("Total Ticks", str(analysis.total_ticks))
    summary.add_row("Total Trades", str(analysis.total_trades))
    summary.add_row("Wins / Losses / Breakeven", f"{analysis.wins} / {analysis.losses} / {analysis.breakeven}")
    if analysis.total_trades > 0:
        summary.add_row("Win Rate", f"{analysis.wins / analysis.total_trades:.1%}")
    summary.add_row("Initial Balance", f"${analysis.initial_balance:,.2f}")
    summary.add_row("Final Balance", f"${analysis.final_balance:,.2f}")
    pnl_color = "green" if analysis.total_pnl >= 0 else "red"
    summary.add_row("Total PnL", f"[{pnl_color}]${analysis.total_pnl:,.2f}[/{pnl_color}]")
    summary.add_row("Return", f"[{pnl_color}]{(analysis.final_balance / analysis.initial_balance - 1) * 100:.1f}%[/{pnl_color}]")

    console.print(summary)
    console.print()

    # Config
    if analysis.config:
        config_table = Table(title="Configuration", show_header=False)
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value", style="white")
        for k, v in analysis.config.items():
            config_table.add_row(k, str(v))
        console.print(config_table)
        console.print()

    # Inefficiencies
    if analysis.inefficiencies:
        console.print("[bold red]Identified Inefficiencies[/bold red]")
        console.print()

        for i, issue in enumerate(analysis.inefficiencies, 1):
            severity_colors = {
                "critical": "red",
                "high": "yellow",
                "medium": "blue",
                "info": "dim",
            }
            color = severity_colors.get(issue["severity"], "white")

            console.print(f"[bold {color}]{i}. {issue['type']}[/bold {color}] [{issue['severity'].upper()}]")
            console.print(f"   {issue['description']}")
            if "impact" in issue:
                console.print(f"   [dim]Impact:[/dim] {issue['impact']}")

            if "details" in issue and issue["details"]:
                details_table = Table(show_header=True, box=None, padding=(0, 2))
                first_detail = issue["details"][0]
                for key in first_detail.keys():
                    details_table.add_column(key, style="dim")
                for detail in issue["details"][:5]:  # Show max 5
                    details_table.add_row(*[str(v) for v in detail.values()])
                console.print(details_table)

            if "recommendation" in issue:
                console.print(f"   [green]Recommendation:[/green] {issue['recommendation']}")
            console.print()
    else:
        console.print("[green]No significant inefficiencies detected.[/green]")

    # Trade details
    if analysis.trades:
        console.print("[bold]Trade Details[/bold]")
        trade_table = Table(show_header=True)
        trade_table.add_column("Entry Tick", style="cyan")
        trade_table.add_column("Side", style="blue")
        trade_table.add_column("Hold", style="dim")
        trade_table.add_column("Entry $", style="white")
        trade_table.add_column("Exit $", style="white")
        trade_table.add_column("PnL", style="white")
        trade_table.add_column("Edge", style="dim")
        trade_table.add_column("Conf", style="dim")

        for t in analysis.trades:
            pnl_color = "green" if t.pnl > 0 else ("red" if t.pnl < 0 else "white")
            trade_table.add_row(
                str(t.entry_tick),
                t.side.upper(),
                str(t.hold_duration_ticks),
                f"{t.entry_price:.3f}",
                f"{t.exit_price:.3f}",
                f"[{pnl_color}]${t.pnl:.2f}[/{pnl_color}]",
                f"{t.edge_at_entry:.3f}",
                f"{t.confidence_at_entry:.1%}",
            )

        console.print(trade_table)


def analyze_log(log_path: str, verbose: bool = True) -> LogAnalysis:
    """Main entry point to analyze a log file."""
    console = Console()
    path = Path(log_path)

    if not path.exists():
        console.print(f"[red]Log file not found: {log_path}[/red]")
        raise FileNotFoundError(log_path)

    analysis = parse_log(path)
    analyze_inefficiencies(analysis)

    if verbose:
        print_analysis(analysis, console)

    return analysis


def analyze_directory(log_dir: str = "logs/paper_trade", pattern: str = "*.jsonl") -> list[LogAnalysis]:
    """Analyze all log files in a directory."""
    console = Console()
    path = Path(log_dir)

    if not path.exists():
        console.print(f"[red]Directory not found: {log_dir}[/red]")
        return []

    log_files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        console.print(f"[yellow]No log files found matching {pattern} in {log_dir}[/yellow]")
        return []

    analyses = []
    for log_file in log_files:
        console.print(f"\n[bold]{'='*60}[/bold]")
        analysis = analyze_log(str(log_file), verbose=True)
        analyses.append(analysis)

    return analyses


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        analyze_log(sys.argv[1])
    else:
        analyze_directory()
