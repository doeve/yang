#!/usr/bin/env python3
"""
Analyze trading logs from backtest and live paper trading.
Useful for feature engineering insights and model quality assessment.

Key analyses:
1. Missed opportunities - signals filtered out that would have been profitable
2. Bad trades - losing trades and why they failed
3. Model calibration - confidence vs actual outcomes
4. Feature correlations - which features predict success
5. Equity curve / drawdown analysis
6. Streak analysis (consecutive wins/losses)
7. Q-value health check
8. Predicted vs actual return calibration
9. Time-in-candle effect on outcomes
10. BTC momentum correlation
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class MissedOpportunity:
    """A signal that was filtered but would have been profitable."""
    tick: int
    action: str
    confidence: float
    expected_return: float
    time_remaining: float
    entry_price: float
    filter_reason: str
    potential_profit: Optional[float] = None
    actual_outcome: Optional[str] = None


@dataclass
class BadTrade:
    """A trade that resulted in a loss."""
    entry_tick: int
    exit_tick: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    confidence_at_entry: float
    expected_return_at_entry: float
    time_remaining_at_entry: float
    exit_reason: str
    ticks_held: int


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def find_missed_opportunities(
    ticks: List[Dict],
    trades: List[Dict],
    config: Dict,
) -> List[MissedOpportunity]:
    """
    Find signals that were filtered out but would have been profitable.
    """
    missed = []
    min_conf = config.get('min_confidence', 0.3)
    min_er = config.get('min_expected_return', 0.02)
    min_time = config.get('min_time_remaining', 0.05)

    # Track when we have positions (can't enter)
    in_position = False
    position_side = None

    # Build trade entry/exit timeline
    entry_ticks = {t.get('tick'): t for t in trades if t.get('action') == 'entry'}
    exit_ticks = set(t.get('tick') for t in trades if t.get('action') in ['exit', 'settlement'])

    for i, tick in enumerate(ticks):
        tick_num = tick.get('tick', i)

        # Update position status
        if tick_num in entry_ticks:
            in_position = True
            position_side = entry_ticks[tick_num].get('side')
        if tick_num in exit_ticks:
            in_position = False
            position_side = None

        # Skip if in position
        if in_position:
            continue

        mo = tick.get('model_output', {})
        action = mo.get('action_name', '')
        conf = mo.get('confidence', 0)
        er = mo.get('expected_return', 0)
        tr = tick.get('time_remaining', 1.0)

        # Check for BUY signals that were filtered
        if action in ['BUY_YES', 'BUY_NO']:
            filter_reason = None

            if conf < min_conf:
                filter_reason = f"confidence {conf:.3f} < {min_conf}"
            elif er < min_er:
                filter_reason = f"expected_return {er:.3f} < {min_er}"
            elif tr < min_time:
                filter_reason = f"time_remaining {tr:.3f} < {min_time}"

            if filter_reason:
                entry_price = tick['market']['yes_price'] if action == 'BUY_YES' else tick['market']['no_price']

                # Calculate potential profit by looking at future prices
                potential_profit = None
                future_ticks = ticks[i+1:min(i+50, len(ticks))]  # Look ahead 50 ticks

                if future_ticks:
                    if action == 'BUY_YES':
                        future_prices = [t['market']['yes_price'] for t in future_ticks]
                    else:
                        future_prices = [t['market']['no_price'] for t in future_ticks]

                    max_price = max(future_prices)
                    potential_profit = (max_price - entry_price) / entry_price

                missed.append(MissedOpportunity(
                    tick=tick_num,
                    action=action,
                    confidence=conf,
                    expected_return=er,
                    time_remaining=tr,
                    entry_price=entry_price,
                    filter_reason=filter_reason,
                    potential_profit=potential_profit,
                ))

    return missed


def find_bad_trades(
    ticks: List[Dict],
    trades: List[Dict],
) -> List[BadTrade]:
    """
    Find trades that resulted in losses and analyze why.
    """
    bad_trades = []

    # Build tick lookup
    tick_lookup = {t.get('tick', i): t for i, t in enumerate(ticks)}

    # Pair entries with exits
    entries = [t for t in trades if t.get('action') == 'entry']
    exits = [t for t in trades if t.get('action') in ['exit', 'settlement']]

    for i, entry in enumerate(entries):
        # Find corresponding exit
        exit_trade = exits[i] if i < len(exits) else None
        if not exit_trade:
            continue

        pnl = exit_trade.get('pnl', 0)
        if pnl >= 0:
            continue  # Skip winning trades

        entry_tick_num = entry.get('tick', 0)
        exit_tick_num = exit_trade.get('tick', 0)

        # Get tick data at entry
        entry_tick = tick_lookup.get(entry_tick_num, {})
        entry_mo = entry_tick.get('model_output', {})

        bad_trades.append(BadTrade(
            entry_tick=entry_tick_num,
            exit_tick=exit_tick_num,
            side=entry.get('side', ''),
            entry_price=entry.get('price', 0),
            exit_price=exit_trade.get('exit_price', 0),
            pnl=pnl,
            pnl_pct=exit_trade.get('pnl_pct', 0),
            confidence_at_entry=entry.get('confidence', entry_mo.get('confidence', 0)),
            expected_return_at_entry=entry.get('expected_return', entry_mo.get('expected_return', 0)),
            time_remaining_at_entry=entry.get('time_remaining', entry_tick.get('time_remaining', 0)),
            exit_reason=exit_trade.get('reason', 'unknown'),
            ticks_held=exit_tick_num - entry_tick_num,
        ))

    return bad_trades


def analyze_model_calibration(ticks: List[Dict], trades: List[Dict]) -> Dict:
    """
    Analyze how well model confidence correlates with actual outcomes.
    """
    # Group signals by confidence buckets
    buckets = defaultdict(lambda: {'total': 0, 'acted': 0, 'profitable': 0})

    # Build trade outcome lookup
    entry_outcomes = {}
    entries = [t for t in trades if t.get('action') == 'entry']
    exits = [t for t in trades if t.get('action') in ['exit', 'settlement']]

    for i, entry in enumerate(entries):
        if i < len(exits):
            entry_outcomes[entry.get('tick')] = exits[i].get('pnl', 0) > 0

    for tick in ticks:
        mo = tick.get('model_output', {})
        action = mo.get('action_name', '')
        conf = mo.get('confidence', 0)
        tick_num = tick.get('tick')

        if action in ['BUY_YES', 'BUY_NO']:
            bucket = int(conf * 10) / 10  # 0.0, 0.1, 0.2, etc.
            buckets[bucket]['total'] += 1

            if tick_num in entry_outcomes:
                buckets[bucket]['acted'] += 1
                if entry_outcomes[tick_num]:
                    buckets[bucket]['profitable'] += 1

    return dict(buckets)


def analyze_feature_importance(ticks: List[Dict]) -> Dict:
    """
    Analyze which market conditions correlate with model signals.
    """
    analysis = {
        'price_at_buy_yes': [],
        'price_at_buy_no': [],
        'price_at_wait': [],
        'btc_change_at_buy_yes': [],
        'btc_change_at_buy_no': [],
        'time_remaining_distribution': defaultdict(int),
    }

    for tick in ticks:
        mo = tick.get('model_output', {})
        action = mo.get('action_name', '')
        market = tick.get('market', {})

        yes_price = market.get('yes_price', 0.5)
        btc_price = market.get('btc_price')
        btc_open = market.get('btc_open')
        tr = tick.get('time_remaining', 1.0)

        btc_change = None
        if btc_price and btc_open and btc_open > 0:
            btc_change = (btc_price - btc_open) / btc_open

        if action == 'BUY_YES':
            analysis['price_at_buy_yes'].append(yes_price)
            if btc_change is not None:
                analysis['btc_change_at_buy_yes'].append(btc_change)
        elif action == 'BUY_NO':
            analysis['price_at_buy_no'].append(yes_price)
            if btc_change is not None:
                analysis['btc_change_at_buy_no'].append(btc_change)
        elif action == 'WAIT':
            analysis['price_at_wait'].append(yes_price)

        # Time remaining buckets
        tr_bucket = round(tr, 1)
        analysis['time_remaining_distribution'][tr_bucket] += 1

    return analysis


def analyze_equity_and_drawdown(ticks: List[Dict], trades: List[Dict]) -> Dict:
    """
    Analyze equity curve, drawdowns, and streaks.
    """
    result = {
        'equity_curve': [],
        'max_drawdown': 0.0,
        'max_drawdown_duration': 0,
        'current_drawdown': 0.0,
        'win_streaks': [],
        'loss_streaks': [],
        'longest_win_streak': 0,
        'longest_loss_streak': 0,
        'profit_factor': 0.0,
        'sharpe_like': 0.0,
    }

    # Extract balance over time
    balances = []
    for tick in ticks:
        bal = tick.get('account', {}).get('balance', 1000.0)
        balances.append(bal)

    if not balances:
        return result

    result['equity_curve'] = balances

    # Calculate drawdown
    peak = balances[0]
    max_dd = 0.0
    dd_start = 0
    max_dd_duration = 0

    for i, bal in enumerate(balances):
        if bal > peak:
            peak = bal
            dd_start = i
        dd = (peak - bal) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - dd_start

    result['max_drawdown'] = max_dd
    result['max_drawdown_duration'] = max_dd_duration
    result['current_drawdown'] = (peak - balances[-1]) / peak if peak > 0 else 0

    # Analyze streaks
    exits = [t for t in trades if t.get('action') in ['exit', 'settlement']]
    if exits:
        pnls = [t.get('pnl', 0) for t in exits]

        # Calculate profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe-like ratio
        if len(pnls) > 1:
            result['sharpe_like'] = np.mean(pnls) / (np.std(pnls) + 1e-8)

        # Streak analysis
        current_streak = 0
        streak_type = None
        win_streaks = []
        loss_streaks = []

        for pnl in pnls:
            if pnl > 0:
                if streak_type == 'win':
                    current_streak += 1
                else:
                    if streak_type == 'loss' and current_streak > 0:
                        loss_streaks.append(current_streak)
                    streak_type = 'win'
                    current_streak = 1
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    if streak_type == 'win' and current_streak > 0:
                        win_streaks.append(current_streak)
                    streak_type = 'loss'
                    current_streak = 1

        # Capture final streak
        if streak_type == 'win':
            win_streaks.append(current_streak)
        elif streak_type == 'loss':
            loss_streaks.append(current_streak)

        result['win_streaks'] = win_streaks
        result['loss_streaks'] = loss_streaks
        result['longest_win_streak'] = max(win_streaks) if win_streaks else 0
        result['longest_loss_streak'] = max(loss_streaks) if loss_streaks else 0

    return result


def analyze_q_values(ticks: List[Dict]) -> Dict:
    """
    Analyze Q-value health - scale, spread, and patterns.
    """
    result = {
        'q_value_stats': {},
        'q_value_spread': [],  # Difference between best and second-best
        'action_q_dominance': defaultdict(int),  # How often each action has highest Q
        'q_scale_issue': False,
        'q_values_by_action': defaultdict(list),
    }

    for tick in ticks:
        mo = tick.get('model_output', {})
        q_values = mo.get('q_values', [])
        action = mo.get('action', 0)
        action_name = mo.get('action_name', '')

        if not q_values or len(q_values) < 2:
            continue

        q_arr = np.array(q_values)

        # Track which action has max Q
        max_q_action = np.argmax(q_arr)
        action_names = ['WAIT', 'BUY_YES', 'BUY_NO', 'EXIT', 'HOLD']
        if max_q_action < len(action_names):
            result['action_q_dominance'][action_names[max_q_action]] += 1

        # Q-value spread (decision margin)
        sorted_q = np.sort(q_arr)[::-1]
        spread = sorted_q[0] - sorted_q[1]
        result['q_value_spread'].append(spread)

        # Store Q-values by action taken
        result['q_values_by_action'][action_name].append(q_arr.tolist())

    # Compute stats
    if result['q_value_spread']:
        spreads = result['q_value_spread']
        result['q_value_stats'] = {
            'mean_spread': np.mean(spreads),
            'min_spread': np.min(spreads),
            'max_spread': np.max(spreads),
            'std_spread': np.std(spreads),
        }

        # Check for Q-scale issues
        all_q = []
        for qlist in result['q_values_by_action'].values():
            for q in qlist:
                all_q.extend(q)
        if all_q:
            mean_q = np.mean(all_q)
            if mean_q < -50:
                result['q_scale_issue'] = True
                result['q_scale_warning'] = f"Q-values are very negative (mean={mean_q:.1f}). This suggests reward scaling issues or training instability."

    return result


def analyze_predicted_vs_actual(ticks: List[Dict], trades: List[Dict]) -> Dict:
    """
    Compare predicted expected_return vs actual returns.
    """
    result = {
        'predictions': [],  # (predicted_er, actual_return, action)
        'mean_predicted': 0.0,
        'mean_actual': 0.0,
        'correlation': 0.0,
        'overconfident': False,
        'by_action': defaultdict(lambda: {'predicted': [], 'actual': []}),
    }

    # Build trade outcome lookup
    tick_lookup = {t.get('tick', i): t for i, t in enumerate(ticks)}
    entries = [t for t in trades if t.get('action') == 'entry']
    exits = [t for t in trades if t.get('action') in ['exit', 'settlement']]

    for i, entry in enumerate(entries):
        if i >= len(exits):
            break

        entry_tick = entry.get('tick')
        exit_trade = exits[i]
        actual_pnl_pct = exit_trade.get('pnl_pct', 0)

        # Get prediction at entry
        tick_data = tick_lookup.get(entry_tick, {})
        mo = tick_data.get('model_output', {})
        predicted_er = mo.get('expected_return', entry.get('expected_return', 0))
        action = entry.get('side', '')

        result['predictions'].append((predicted_er, actual_pnl_pct, action))
        result['by_action'][action]['predicted'].append(predicted_er)
        result['by_action'][action]['actual'].append(actual_pnl_pct)

    if result['predictions']:
        predicted = [p[0] for p in result['predictions']]
        actual = [p[1] for p in result['predictions']]

        result['mean_predicted'] = np.mean(predicted)
        result['mean_actual'] = np.mean(actual)

        if len(predicted) > 2:
            result['correlation'] = np.corrcoef(predicted, actual)[0, 1]

        # Check for overconfidence
        if result['mean_predicted'] > 0 and result['mean_actual'] < 0:
            result['overconfident'] = True
            result['overconfidence_warning'] = f"Model predicts +{result['mean_predicted']:.1%} but achieves {result['mean_actual']:.1%}"

    return result


def analyze_time_effect(ticks: List[Dict], trades: List[Dict]) -> Dict:
    """
    Analyze how time_remaining affects trade outcomes.
    """
    result = {
        'by_time_bucket': defaultdict(lambda: {'entries': 0, 'wins': 0, 'pnl': 0.0}),
        'late_entry_threshold': 0.3,  # <30% time remaining
        'late_entries': 0,
        'late_wins': 0,
        'recommendation': '',
    }

    # Build trade outcome lookup
    entries = [t for t in trades if t.get('action') == 'entry']
    exits = [t for t in trades if t.get('action') in ['exit', 'settlement']]
    tick_lookup = {t.get('tick', i): t for i, t in enumerate(ticks)}

    for i, entry in enumerate(entries):
        if i >= len(exits):
            break

        entry_tick = entry.get('tick')
        exit_trade = exits[i]
        pnl = exit_trade.get('pnl', 0)

        # Get time remaining at entry
        tr = entry.get('time_remaining')
        if tr is None:
            tick_data = tick_lookup.get(entry_tick, {})
            tr = tick_data.get('time_remaining', 1.0)

        # Bucket: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        bucket = int(tr * 5) / 5
        bucket = min(bucket, 0.8)  # Cap at 0.8

        result['by_time_bucket'][bucket]['entries'] += 1
        result['by_time_bucket'][bucket]['pnl'] += pnl
        if pnl > 0:
            result['by_time_bucket'][bucket]['wins'] += 1

        # Track late entries
        if tr < result['late_entry_threshold']:
            result['late_entries'] += 1
            if pnl > 0:
                result['late_wins'] += 1

    # Generate recommendation
    if result['late_entries'] > 0:
        late_wr = result['late_wins'] / result['late_entries']
        if late_wr < 0.4:
            result['recommendation'] = f"Late entries (<{result['late_entry_threshold']:.0%} time) have {late_wr:.0%} win rate. Consider increasing min_time_remaining filter."

    return result


def analyze_btc_momentum(ticks: List[Dict], trades: List[Dict]) -> Dict:
    """
    Analyze correlation between BTC momentum and trade success.
    """
    result = {
        'momentum_correlation': 0.0,
        'contrarian_trades': 0,  # Trades against BTC direction
        'contrarian_wins': 0,
        'momentum_trades': 0,  # Trades with BTC direction
        'momentum_wins': 0,
        'recommendation': '',
    }

    entries = [t for t in trades if t.get('action') == 'entry']
    exits = [t for t in trades if t.get('action') in ['exit', 'settlement']]
    tick_lookup = {t.get('tick', i): t for i, t in enumerate(ticks)}

    for i, entry in enumerate(entries):
        if i >= len(exits):
            break

        entry_tick = entry.get('tick')
        exit_trade = exits[i]
        pnl = exit_trade.get('pnl', 0)
        side = entry.get('side', '')

        # Get BTC change at entry
        tick_data = tick_lookup.get(entry_tick, {})
        market = tick_data.get('market', {})
        btc_price = market.get('btc_price')
        btc_open = market.get('btc_open')

        if btc_price and btc_open and btc_open > 0:
            btc_change = (btc_price - btc_open) / btc_open

            # BUY_YES when BTC up = momentum trade
            # BUY_YES when BTC down = contrarian trade
            is_momentum = (side == 'yes' and btc_change > 0) or (side == 'no' and btc_change < 0)

            if is_momentum:
                result['momentum_trades'] += 1
                if pnl > 0:
                    result['momentum_wins'] += 1
            else:
                result['contrarian_trades'] += 1
                if pnl > 0:
                    result['contrarian_wins'] += 1

    # Calculate win rates
    if result['momentum_trades'] > 0:
        mom_wr = result['momentum_wins'] / result['momentum_trades']
    else:
        mom_wr = 0

    if result['contrarian_trades'] > 0:
        con_wr = result['contrarian_wins'] / result['contrarian_trades']
    else:
        con_wr = 0

    result['momentum_win_rate'] = mom_wr
    result['contrarian_win_rate'] = con_wr

    if result['momentum_trades'] > 3 and result['contrarian_trades'] > 3:
        if con_wr > mom_wr + 0.1:
            result['recommendation'] = f"Contrarian trades ({con_wr:.0%} WR) outperform momentum ({mom_wr:.0%}). Consider adding mean-reversion features."
        elif mom_wr > con_wr + 0.1:
            result['recommendation'] = f"Momentum trades ({mom_wr:.0%} WR) outperform contrarian ({con_wr:.0%}). Model may benefit from trend features."

    return result


def analyze_action_consistency(ticks: List[Dict]) -> Dict:
    """
    Analyze if model action matches max Q-value (should always match for deterministic policy).
    Also detect HOLD-without-position bugs.
    """
    result = {
        'action_mismatch_count': 0,
        'hold_without_position': 0,
        'exit_without_position': 0,
        'action_changes': 0,
        'action_persistence': defaultdict(int),  # How long actions persist
    }

    prev_action = None
    action_run_length = 0

    for tick in ticks:
        mo = tick.get('model_output', {})
        q_values = mo.get('q_values', [])
        action = mo.get('action', 0)
        action_name = mo.get('action_name', '')
        position = tick.get('position', {})
        has_position = position.get('side') is not None

        # Check Q-value consistency
        if q_values and len(q_values) > action:
            max_q_action = np.argmax(q_values)
            if max_q_action != action:
                result['action_mismatch_count'] += 1

        # Check HOLD/EXIT without position
        if action_name == 'HOLD' and not has_position:
            result['hold_without_position'] += 1
        if action_name == 'EXIT' and not has_position:
            result['exit_without_position'] += 1

        # Track action persistence
        if action_name == prev_action:
            action_run_length += 1
        else:
            if prev_action:
                result['action_persistence'][prev_action] = max(
                    result['action_persistence'].get(prev_action, 0),
                    action_run_length
                )
            result['action_changes'] += 1
            action_run_length = 1
            prev_action = action_name

    return result


def analyze_candle_performance(ticks: List[Dict], trades: List[Dict]) -> Dict:
    """
    Analyze performance by candle (market session).
    """
    result = {
        'by_candle': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0, 'ticks': 0}),
        'best_candle': None,
        'worst_candle': None,
    }

    # Count ticks per candle
    for tick in ticks:
        candle_ts = tick.get('candle_ts')
        if candle_ts:
            result['by_candle'][candle_ts]['ticks'] += 1

    # Assign trades to candles
    for trade in trades:
        if trade.get('action') in ['exit', 'settlement']:
            # Try to get candle_ts from trade or infer
            candle_ts = trade.get('candle_ts')
            if not candle_ts:
                continue

            result['by_candle'][candle_ts]['trades'] += 1
            pnl = trade.get('pnl', 0)
            result['by_candle'][candle_ts]['pnl'] += pnl
            if pnl > 0:
                result['by_candle'][candle_ts]['wins'] += 1

    # Find best/worst
    candles_with_trades = [(ts, data) for ts, data in result['by_candle'].items() if data['trades'] > 0]
    if candles_with_trades:
        result['best_candle'] = max(candles_with_trades, key=lambda x: x[1]['pnl'])
        result['worst_candle'] = min(candles_with_trades, key=lambda x: x[1]['pnl'])

    return result


def print_missed_opportunities(missed: List[MissedOpportunity]):
    """Print missed opportunities analysis."""
    if not missed:
        print("\n[Missed Opportunities] None detected")
        return

    print(f"\n{'='*60}")
    print(f"  MISSED OPPORTUNITIES ({len(missed)} signals filtered)")
    print(f"{'='*60}")

    # Group by filter reason
    by_reason = defaultdict(list)
    for m in missed:
        reason_type = m.filter_reason.split()[0]  # 'confidence', 'expected_return', etc.
        by_reason[reason_type].append(m)

    print(f"\nBy Filter Reason:")
    for reason, items in by_reason.items():
        profitable = [m for m in items if m.potential_profit and m.potential_profit > 0.05]
        avg_potential = np.mean([m.potential_profit for m in items if m.potential_profit]) if items else 0
        print(f"  {reason}: {len(items)} filtered, {len(profitable)} would be >5% profit, avg potential: {avg_potential:.1%}")

    # Show top missed profits
    profitable_missed = sorted(
        [m for m in missed if m.potential_profit],
        key=lambda x: x.potential_profit,
        reverse=True
    )[:10]

    if profitable_missed:
        print(f"\nTop 10 Missed Profits:")
        for m in profitable_missed:
            print(f"  Tick {m.tick}: {m.action} @ ${m.entry_price:.3f}, potential +{m.potential_profit:.1%}")
            print(f"    Filter: {m.filter_reason}")

    # Confidence distribution of missed profitable signals
    profitable_confs = [m.confidence for m in missed if m.potential_profit and m.potential_profit > 0.05]
    if profitable_confs:
        print(f"\nConfidence of Missed Profitable Signals:")
        print(f"  min: {np.min(profitable_confs):.3f}")
        print(f"  max: {np.max(profitable_confs):.3f}")
        print(f"  mean: {np.mean(profitable_confs):.3f}")
        print(f"  Suggestion: Consider lowering min_confidence to {np.percentile(profitable_confs, 25):.2f}")


def print_bad_trades(bad_trades: List[BadTrade]):
    """Print bad trades analysis."""
    if not bad_trades:
        print("\n[Bad Trades] None detected")
        return

    print(f"\n{'='*60}")
    print(f"  BAD TRADES ({len(bad_trades)} losing trades)")
    print(f"{'='*60}")

    total_loss = sum(t.pnl for t in bad_trades)
    avg_loss = np.mean([t.pnl for t in bad_trades])
    avg_loss_pct = np.mean([t.pnl_pct for t in bad_trades])

    print(f"\nOverall:")
    print(f"  Total Loss: ${total_loss:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f} ({avg_loss_pct:.1%})")

    # By side
    yes_trades = [t for t in bad_trades if t.side == 'yes']
    no_trades = [t for t in bad_trades if t.side == 'no']

    print(f"\nBy Side:")
    if yes_trades:
        print(f"  YES: {len(yes_trades)} trades, ${sum(t.pnl for t in yes_trades):.2f} loss")
    if no_trades:
        print(f"  NO: {len(no_trades)} trades, ${sum(t.pnl for t in no_trades):.2f} loss")

    # By exit reason
    by_reason = defaultdict(list)
    for t in bad_trades:
        by_reason[t.exit_reason].append(t)

    print(f"\nBy Exit Reason:")
    for reason, trades in by_reason.items():
        print(f"  {reason}: {len(trades)} trades, ${sum(t.pnl for t in trades):.2f} loss")

    # Analyze patterns
    print(f"\nPatterns in Bad Trades:")

    # Confidence at entry
    confs = [t.confidence_at_entry for t in bad_trades]
    print(f"  Confidence at entry: mean={np.mean(confs):.3f}, range=[{np.min(confs):.3f}, {np.max(confs):.3f}]")

    # Time remaining
    trs = [t.time_remaining_at_entry for t in bad_trades]
    print(f"  Time remaining at entry: mean={np.mean(trs):.3f}, range=[{np.min(trs):.3f}, {np.max(trs):.3f}]")

    # Ticks held
    held = [t.ticks_held for t in bad_trades]
    print(f"  Ticks held: mean={np.mean(held):.1f}, range=[{np.min(held)}, {np.max(held)}]")

    # Entry prices
    print(f"\nEntry Price Analysis (Bad Trades):")
    yes_entry = [t.entry_price for t in bad_trades if t.side == 'yes']
    no_entry = [t.entry_price for t in bad_trades if t.side == 'no']
    if yes_entry:
        print(f"  YES entries: mean=${np.mean(yes_entry):.3f} (bought high?)")
    if no_entry:
        print(f"  NO entries: mean=${np.mean(no_entry):.3f}")

    # Show worst trades
    print(f"\nWorst 5 Trades:")
    for t in sorted(bad_trades, key=lambda x: x.pnl)[:5]:
        print(f"  {t.side.upper()} @ ${t.entry_price:.3f} -> ${t.exit_price:.3f}: ${t.pnl:.2f} ({t.pnl_pct:.1%})")
        print(f"    conf={t.confidence_at_entry:.3f}, er={t.expected_return_at_entry:.3f}, tr={t.time_remaining_at_entry:.3f}, reason={t.exit_reason}")


def print_calibration(calibration: Dict):
    """Print model calibration analysis."""
    if not calibration:
        return

    print(f"\n{'='*60}")
    print(f"  MODEL CALIBRATION (Confidence vs Actual Win Rate)")
    print(f"{'='*60}")

    print(f"\n{'Conf Bucket':<12} {'Signals':<10} {'Acted':<10} {'Won':<10} {'Win Rate':<10}")
    print("-" * 52)

    for bucket in sorted(calibration.keys()):
        data = calibration[bucket]
        total = data['total']
        acted = data['acted']
        won = data['profitable']
        win_rate = won / acted if acted > 0 else 0

        print(f"{bucket:.1f}-{bucket+0.1:.1f}      {total:<10} {acted:<10} {won:<10} {win_rate:.1%}")

    # Recommendations
    print(f"\nCalibration Insights:")
    high_conf_buckets = [(b, d) for b, d in calibration.items() if b >= 0.3 and d['acted'] > 0]
    if high_conf_buckets:
        avg_win_rate = np.mean([d['profitable']/d['acted'] for _, d in high_conf_buckets if d['acted'] > 0])
        print(f"  High confidence (>=0.3) win rate: {avg_win_rate:.1%}")


def print_feature_analysis(analysis: Dict):
    """Print feature importance analysis."""
    print(f"\n{'='*60}")
    print(f"  FEATURE ANALYSIS (for engineering)")
    print(f"{'='*60}")

    # Price at different actions
    if analysis['price_at_buy_yes']:
        prices = analysis['price_at_buy_yes']
        print(f"\nYES Price when BUY_YES signaled:")
        print(f"  mean: {np.mean(prices):.3f}, std: {np.std(prices):.3f}")
        print(f"  Model buys YES when price is {'high' if np.mean(prices) > 0.5 else 'low'}")

    if analysis['price_at_buy_no']:
        prices = analysis['price_at_buy_no']
        print(f"\nYES Price when BUY_NO signaled:")
        print(f"  mean: {np.mean(prices):.3f}, std: {np.std(prices):.3f}")
        print(f"  Model buys NO when YES price is {'high' if np.mean(prices) > 0.5 else 'low'}")

    # BTC correlation
    if analysis['btc_change_at_buy_yes']:
        changes = analysis['btc_change_at_buy_yes']
        print(f"\nBTC Change when BUY_YES signaled:")
        print(f"  mean: {np.mean(changes):.4f}, std: {np.std(changes):.4f}")
        print(f"  Model buys YES when BTC is {'up' if np.mean(changes) > 0 else 'down'}")

    if analysis['btc_change_at_buy_no']:
        changes = analysis['btc_change_at_buy_no']
        print(f"\nBTC Change when BUY_NO signaled:")
        print(f"  mean: {np.mean(changes):.4f}, std: {np.std(changes):.4f}")

    # Time distribution
    print(f"\nSignals by Time Remaining:")
    tr_dist = analysis['time_remaining_distribution']
    for tr in sorted(tr_dist.keys(), reverse=True)[:5]:
        print(f"  {tr:.1f}: {tr_dist[tr]} signals")


def print_equity_analysis(analysis: Dict):
    """Print equity curve and drawdown analysis."""
    print(f"\n{'='*60}")
    print(f"  EQUITY & RISK ANALYSIS")
    print(f"{'='*60}")

    ec = analysis.get('equity_curve', [])
    if ec:
        print(f"\nEquity Curve:")
        print(f"  Start: ${ec[0]:.2f}")
        print(f"  End: ${ec[-1]:.2f}")
        print(f"  Min: ${min(ec):.2f}")
        print(f"  Max: ${max(ec):.2f}")

    print(f"\nDrawdown:")
    print(f"  Max Drawdown: {analysis['max_drawdown']:.1%}")
    print(f"  Max DD Duration: {analysis['max_drawdown_duration']} ticks")
    print(f"  Current Drawdown: {analysis['current_drawdown']:.1%}")

    print(f"\nRisk Metrics:")
    pf = analysis.get('profit_factor', 0)
    print(f"  Profit Factor: {pf:.2f}" if pf != float('inf') else "  Profit Factor: ∞ (no losses)")
    print(f"  Sharpe-like: {analysis.get('sharpe_like', 0):.3f}")

    print(f"\nStreaks:")
    print(f"  Longest Win Streak: {analysis['longest_win_streak']}")
    print(f"  Longest Loss Streak: {analysis['longest_loss_streak']}")

    ws = analysis.get('win_streaks', [])
    ls = analysis.get('loss_streaks', [])
    if ws:
        print(f"  Win Streaks: {ws}")
    if ls:
        print(f"  Loss Streaks: {ls}")

    # Warnings
    if analysis['longest_loss_streak'] >= 4:
        print(f"\n  ⚠️  WARNING: {analysis['longest_loss_streak']} consecutive losses detected!")
    if analysis['max_drawdown'] > 0.15:
        print(f"\n  ⚠️  WARNING: Max drawdown {analysis['max_drawdown']:.1%} exceeds 15%!")


def print_q_value_analysis(analysis: Dict):
    """Print Q-value health analysis."""
    print(f"\n{'='*60}")
    print(f"  Q-VALUE HEALTH CHECK")
    print(f"{'='*60}")

    stats = analysis.get('q_value_stats', {})
    if stats:
        print(f"\nQ-Value Spread (decision margin):")
        print(f"  Mean: {stats.get('mean_spread', 0):.3f}")
        print(f"  Min: {stats.get('min_spread', 0):.3f}")
        print(f"  Max: {stats.get('max_spread', 0):.3f}")
        print(f"  Std: {stats.get('std_spread', 0):.3f}")

    print(f"\nAction by Max Q-Value:")
    dominance = analysis.get('action_q_dominance', {})
    total = sum(dominance.values()) or 1
    for action, count in sorted(dominance.items(), key=lambda x: -x[1]):
        print(f"  {action:12}: {count:5} ({count/total*100:.1f}%)")

    if analysis.get('q_scale_issue'):
        print(f"\n  ⚠️  {analysis.get('q_scale_warning', 'Q-value scale issue detected')}")


def print_predicted_vs_actual(analysis: Dict):
    """Print predicted vs actual return analysis."""
    print(f"\n{'='*60}")
    print(f"  PREDICTED vs ACTUAL RETURNS")
    print(f"{'='*60}")

    if not analysis['predictions']:
        print("\n  No trades to analyze.")
        return

    print(f"\nOverall:")
    print(f"  Mean Predicted Return: {analysis['mean_predicted']:+.2%}")
    print(f"  Mean Actual Return: {analysis['mean_actual']:+.2%}")
    print(f"  Correlation: {analysis['correlation']:.3f}")

    print(f"\nBy Side:")
    for action, data in analysis['by_action'].items():
        if data['predicted']:
            pred_mean = np.mean(data['predicted'])
            act_mean = np.mean(data['actual'])
            print(f"  {action.upper():6}: Predicted {pred_mean:+.2%}, Actual {act_mean:+.2%}")

    if analysis.get('overconfident'):
        print(f"\n  ⚠️  {analysis.get('overconfidence_warning', 'Model is overconfident')}")

    # Calibration quality
    if analysis['correlation'] < 0:
        print(f"\n  ⚠️  NEGATIVE CORRELATION: Model predictions are anti-correlated with outcomes!")
        print(f"       This suggests the expected_return output is not calibrated.")
    elif analysis['correlation'] < 0.1:
        print(f"\n  ⚠️  WEAK CORRELATION: Expected return has little predictive value.")


def print_time_effect(analysis: Dict):
    """Print time-in-candle effect analysis."""
    print(f"\n{'='*60}")
    print(f"  TIME-IN-CANDLE EFFECT")
    print(f"{'='*60}")

    print(f"\nWin Rate by Entry Time:")
    print(f"  {'Time Bucket':<15} {'Entries':<10} {'Wins':<10} {'Win Rate':<10} {'PnL':<10}")
    print(f"  {'-'*55}")

    for bucket in sorted(analysis['by_time_bucket'].keys(), reverse=True):
        data = analysis['by_time_bucket'][bucket]
        entries = data['entries']
        wins = data['wins']
        pnl = data['pnl']
        wr = wins / entries if entries > 0 else 0
        label = f"{bucket:.1f}-{bucket+0.2:.1f}"
        print(f"  {label:<15} {entries:<10} {wins:<10} {wr:<10.1%} ${pnl:<+10.2f}")

    if analysis['late_entries'] > 0:
        late_wr = analysis['late_wins'] / analysis['late_entries']
        print(f"\nLate Entries (<{analysis['late_entry_threshold']:.0%} time):")
        print(f"  Count: {analysis['late_entries']}, Win Rate: {late_wr:.1%}")

    if analysis.get('recommendation'):
        print(f"\n  💡 {analysis['recommendation']}")


def print_btc_momentum(analysis: Dict):
    """Print BTC momentum analysis."""
    print(f"\n{'='*60}")
    print(f"  BTC MOMENTUM ANALYSIS")
    print(f"{'='*60}")

    print(f"\nMomentum Trades (with BTC direction):")
    mom = analysis['momentum_trades']
    mom_w = analysis['momentum_wins']
    print(f"  Count: {mom}, Wins: {mom_w}, Win Rate: {analysis.get('momentum_win_rate', 0):.1%}")

    print(f"\nContrarian Trades (against BTC direction):")
    con = analysis['contrarian_trades']
    con_w = analysis['contrarian_wins']
    print(f"  Count: {con}, Wins: {con_w}, Win Rate: {analysis.get('contrarian_win_rate', 0):.1%}")

    if analysis.get('recommendation'):
        print(f"\n  💡 {analysis['recommendation']}")


def print_action_consistency(analysis: Dict):
    """Print action consistency analysis."""
    print(f"\n{'='*60}")
    print(f"  ACTION CONSISTENCY CHECK")
    print(f"{'='*60}")

    if analysis['action_mismatch_count'] > 0:
        print(f"\n  ⚠️  Action-Q mismatch: {analysis['action_mismatch_count']} times")
        print(f"       (Action taken differs from max Q-value)")

    if analysis['hold_without_position'] > 0:
        print(f"\n  ⚠️  HOLD without position: {analysis['hold_without_position']} times")
        print(f"       This is a BUG - HOLD should only output when holding a position.")

    if analysis['exit_without_position'] > 0:
        print(f"\n  ⚠️  EXIT without position: {analysis['exit_without_position']} times")
        print(f"       This is a BUG - EXIT should only output when holding a position.")

    print(f"\n  Action changes: {analysis['action_changes']} times")

    if analysis['hold_without_position'] == 0 and analysis['exit_without_position'] == 0 and analysis['action_mismatch_count'] == 0:
        print(f"\n  ✓ No consistency issues detected.")


def print_candle_performance(analysis: Dict):
    """Print per-candle performance."""
    print(f"\n{'='*60}")
    print(f"  PER-CANDLE PERFORMANCE")
    print(f"{'='*60}")

    candles = [(ts, data) for ts, data in analysis['by_candle'].items() if data['trades'] > 0]
    if not candles:
        print("\n  No candles with trades.")
        return

    print(f"\n  Total candles with trades: {len(candles)}")

    if analysis['best_candle']:
        ts, data = analysis['best_candle']
        print(f"\n  Best Candle: ts={ts}")
        print(f"    Trades: {data['trades']}, Wins: {data['wins']}, PnL: ${data['pnl']:.2f}")

    if analysis['worst_candle']:
        ts, data = analysis['worst_candle']
        print(f"\n  Worst Candle: ts={ts}")
        print(f"    Trades: {data['trades']}, Wins: {data['wins']}, PnL: ${data['pnl']:.2f}")


def print_recommendations_summary(
    equity: Dict,
    q_analysis: Dict,
    pred_actual: Dict,
    consistency: Dict,
    time_effect: Dict,
    btc_momentum: Dict,
    bad_trades: List[BadTrade],
    missed: List[MissedOpportunity],
):
    """Print prioritized recommendations based on all analyses."""
    print(f"\n{'='*60}")
    print(f"  PRIORITIZED IMPROVEMENT ROADMAP")
    print(f"{'='*60}")

    issues = []

    # Critical issues first
    if consistency.get('hold_without_position', 0) > 0:
        issues.append((1, "CRITICAL BUG", "Model outputs HOLD when no position exists. Fix action masking in training."))

    if consistency.get('exit_without_position', 0) > 0:
        issues.append((1, "CRITICAL BUG", "Model outputs EXIT when no position exists. Fix action masking in training."))

    if q_analysis.get('q_scale_issue'):
        issues.append((2, "TRAINING ISSUE", "Q-values are severely negative. Check reward scaling and discount factor."))

    if pred_actual.get('overconfident'):
        issues.append((2, "CALIBRATION", f"Model overconfident: predicts {pred_actual['mean_predicted']:+.1%}, achieves {pred_actual['mean_actual']:+.1%}"))

    if pred_actual.get('correlation', 0) < 0:
        issues.append((2, "CALIBRATION", "Expected return is ANTI-correlated with outcomes. Retrain or remove this output."))

    if equity.get('max_drawdown', 0) > 0.15:
        issues.append((3, "RISK", f"Max drawdown {equity['max_drawdown']:.1%} too high. Add position sizing limits or stop-loss."))

    if equity.get('longest_loss_streak', 0) >= 4:
        issues.append((3, "RISK", f"Loss streak of {equity['longest_loss_streak']}. Consider adding regime detection to pause trading."))

    # Strategy issues
    if bad_trades:
        yes_losses = sum(1 for t in bad_trades if t.side == 'yes')
        no_losses = sum(1 for t in bad_trades if t.side == 'no')
        if no_losses > yes_losses * 2:
            issues.append((4, "BIAS", f"Heavy NO-side losses ({no_losses} vs {yes_losses} YES). Model may have directional bias."))
        elif yes_losses > no_losses * 2:
            issues.append((4, "BIAS", f"Heavy YES-side losses ({yes_losses} vs {no_losses} NO). Model may have directional bias."))

    # Time effect
    late_entries = time_effect.get('late_entries', 0)
    if late_entries > 0:
        late_wr = time_effect.get('late_wins', 0) / late_entries
        if late_wr < 0.35:
            issues.append((4, "TIMING", f"Late entries have {late_wr:.0%} win rate. Increase min_time_remaining threshold."))

    # Momentum
    mom_wr = btc_momentum.get('momentum_win_rate', 0)
    con_wr = btc_momentum.get('contrarian_win_rate', 0)
    if btc_momentum.get('momentum_trades', 0) >= 3 and btc_momentum.get('contrarian_trades', 0) >= 3:
        if abs(mom_wr - con_wr) > 0.15:
            better = "contrarian" if con_wr > mom_wr else "momentum"
            issues.append((5, "FEATURE", f"Add {better} features. Win rate diff: {abs(mom_wr-con_wr):.0%}"))

    # Missed opportunities
    profitable_missed = [m for m in missed if m.potential_profit and m.potential_profit > 0.10]
    if len(profitable_missed) > 10:
        issues.append((5, "THRESHOLD", f"{len(profitable_missed)} signals with >10% potential were filtered. Consider lowering min_confidence."))

    # Sort by priority and print
    issues.sort(key=lambda x: x[0])

    if not issues:
        print("\n  ✓ No critical issues detected. Model appears healthy.")
        return

    print("\n")
    for priority, category, description in issues:
        priority_label = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🔵", 5: "⚪"}
        print(f"  {priority_label.get(priority, '•')} [{category}] {description}")

    print(f"\n{'='*60}")
    print("  SUGGESTED EXPERIMENTS")
    print(f"{'='*60}")

    experiments = []

    if any(i[1] == "TRAINING ISSUE" for i in issues):
        experiments.append("1. Reward normalization: Scale rewards to [-1, 1] range")
        experiments.append("2. Check discount factor gamma - may be too high")

    if any(i[1] == "CALIBRATION" for i in issues):
        experiments.append("3. Re-train with proper return targets (realized PnL, not predicted)")
        experiments.append("4. Add calibration loss term during training")

    if any(i[1] == "BIAS" for i in issues):
        experiments.append("5. Check training data balance between YES/NO outcomes")
        experiments.append("6. Add symmetry augmentation (flip YES<->NO)")

    if any(i[1] in ["TIMING", "FEATURE"] for i in issues):
        experiments.append("7. Add volatility regime features (ATR, price range)")
        experiments.append("8. Add momentum indicators (RSI, price velocity)")
        experiments.append("9. Add time-decay feature for confidence near candle end")

    if any(i[1] == "RISK" for i in issues):
        experiments.append("10. Implement trailing stop-loss in execution logic")
        experiments.append("11. Add drawdown-based position scaling")

    for exp in experiments[:6]:  # Limit to top 6
        print(f"  • {exp}")


def analyze_log(entries: list, name: str, detailed: bool = True):
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

    config = metadata[0].get('config', {}) if metadata else {}
    if config:
        print(f"\nConfig:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    if not ticks:
        print("\nNo tick data to analyze.")
        return

    # Basic action distribution
    actions = [t['model_output']['action_name'] for t in ticks]
    action_counts = defaultdict(int)
    for a in actions:
        action_counts[a] += 1

    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(actions) * 100
        print(f"  {action:12} {count:6} ({pct:5.1f}%)")

    # Confidence and expected return stats
    confidences = [t['model_output']['confidence'] for t in ticks]
    exp_returns = [t['model_output']['expected_return'] for t in ticks]

    print(f"\nConfidence: min={np.min(confidences):.3f}, max={np.max(confidences):.3f}, mean={np.mean(confidences):.3f}")
    print(f"Expected Return: min={np.min(exp_returns):.3f}, max={np.max(exp_returns):.3f}, mean={np.mean(exp_returns):.3f}")

    # Price stats
    yes_prices = [t['market']['yes_price'] for t in ticks]
    print(f"YES Price: min={np.min(yes_prices):.3f}, max={np.max(yes_prices):.3f}, mean={np.mean(yes_prices):.3f}")

    # Trade summary
    if trades:
        entry_trades = [t for t in trades if t.get('action') == 'entry']
        exit_trades = [t for t in trades if t.get('action') in ['exit', 'settlement']]

        if exit_trades:
            pnls = [t.get('pnl', 0) for t in exit_trades]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p <= 0)
            total_pnl = sum(pnls)

            print(f"\nTrade Results: {len(entry_trades)} entries, {len(exit_trades)} exits")
            print(f"  Win/Loss: {wins}/{losses} ({wins/(wins+losses)*100:.1f}% win rate)" if (wins+losses) > 0 else "")
            print(f"  Total PnL: ${total_pnl:.2f}")

    # Detailed analysis
    if detailed:
        # Equity and risk analysis
        equity_analysis = analyze_equity_and_drawdown(ticks, trades)
        print_equity_analysis(equity_analysis)

        # Q-value health check
        q_analysis = analyze_q_values(ticks)
        print_q_value_analysis(q_analysis)

        # Predicted vs actual returns
        pred_actual = analyze_predicted_vs_actual(ticks, trades)
        print_predicted_vs_actual(pred_actual)

        # Action consistency check
        consistency = analyze_action_consistency(ticks)
        print_action_consistency(consistency)

        # Time effect analysis
        time_effect = analyze_time_effect(ticks, trades)
        print_time_effect(time_effect)

        # BTC momentum analysis
        btc_momentum = analyze_btc_momentum(ticks, trades)
        print_btc_momentum(btc_momentum)

        # Bad trades
        bad_trades = find_bad_trades(ticks, trades)
        print_bad_trades(bad_trades)

        # Missed opportunities
        missed = find_missed_opportunities(ticks, trades, config)
        print_missed_opportunities(missed)

        # Model calibration
        calibration = analyze_model_calibration(ticks, trades)
        print_calibration(calibration)

        # Feature analysis
        feature_analysis = analyze_feature_importance(ticks)
        print_feature_analysis(feature_analysis)

        # Per-candle performance
        candle_perf = analyze_candle_performance(ticks, trades)
        print_candle_performance(candle_perf)

        # Final recommendations summary
        print_recommendations_summary(
            equity_analysis, q_analysis, pred_actual, consistency,
            time_effect, btc_momentum, bad_trades, missed
        )

    # Summary
    if summary:
        s = summary[0]
        print(f"\nFinal Summary:")
        print(f"  Total Ticks: {s.get('total_ticks', 'N/A')}")
        fb = s.get('final_balance')
        print(f"  Final Balance: ${fb:.2f}" if isinstance(fb, (int, float)) else f"  Final Balance: {fb}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trading logs for feature engineering")
    parser.add_argument("--backtest", type=str, help="Path to backtest log")
    parser.add_argument("--live", type=str, help="Path to live paper trade log")
    parser.add_argument("--all", action="store_true", help="Analyze all logs in default directories")
    parser.add_argument("--brief", action="store_true", help="Brief output (skip detailed analysis)")
    args = parser.parse_args()

    detailed = not args.brief

    if args.all:
        backtest_dir = Path("./logs/backtest_unified")
        live_dir = Path("./logs/paper_trade_unified")

        if backtest_dir.exists():
            for log_file in sorted(backtest_dir.glob("*.jsonl"))[-3:]:  # Last 3 logs
                entries = load_jsonl(log_file)
                analyze_log(entries, f"BACKTEST: {log_file.name}", detailed)

        if live_dir.exists():
            for log_file in sorted(live_dir.glob("*.jsonl"))[-3:]:  # Last 3 logs
                entries = load_jsonl(log_file)
                analyze_log(entries, f"LIVE: {log_file.name}", detailed)

    else:
        if args.backtest:
            entries = load_jsonl(Path(args.backtest))
            analyze_log(entries, f"BACKTEST: {args.backtest}", detailed)

        if args.live:
            entries = load_jsonl(Path(args.live))
            analyze_log(entries, f"LIVE: {args.live}", detailed)

        if not args.backtest and not args.live:
            print("Usage: python scripts/analyze_trading_logs.py --all")
            print("   or: python scripts/analyze_trading_logs.py --backtest <path> --live <path>")
            print("   Add --brief for shorter output")


if __name__ == "__main__":
    main()
