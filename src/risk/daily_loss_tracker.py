"""
Daily loss tracking for risk management.

Tracks realized PnL per day and triggers trading pause when limit is hit.
"""

from datetime import datetime, timezone
from typing import List, Tuple


class DailyLossTracker:
    """
    Tracks daily realized PnL and enforces loss limits.
    
    Resets at midnight UTC each day.
    """
    
    def __init__(self, starting_balance: float = 1000.0):
        self.starting_balance = starting_balance
        self._daily_pnl = 0.0
        self._trade_history: List[Tuple[datetime, float]] = []  # (time, pnl)
        self._current_date = self._get_utc_date()
        self._was_limit_hit = False
    
    def _get_utc_date(self) -> str:
        """Get current UTC date as string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def _check_day_reset(self) -> None:
        """Reset if we've crossed to a new day."""
        current_date = self._get_utc_date()
        if current_date != self._current_date:
            # New day - reset
            self._daily_pnl = 0.0
            self._trade_history = []
            self._current_date = current_date
            self._was_limit_hit = False
    
    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade's PnL.
        
        Args:
            pnl: Realized PnL from the trade (positive or negative)
        """
        self._check_day_reset()
        self._daily_pnl += pnl
        self._trade_history.append((datetime.now(timezone.utc), pnl))
    
    def get_daily_pnl(self) -> float:
        """Get total realized PnL for today."""
        self._check_day_reset()
        return self._daily_pnl
    
    def get_daily_pnl_pct(self) -> float:
        """Get daily PnL as percentage of starting balance."""
        if self.starting_balance <= 0:
            return 0.0
        return (self._daily_pnl / self.starting_balance) * 100
    
    def is_limit_hit(self, max_loss_pct: float) -> bool:
        """
        Check if daily loss limit has been exceeded.
        
        Args:
            max_loss_pct: Maximum allowed daily loss as percentage (e.g., 5.0 for 5%)
        
        Returns:
            True if loss limit exceeded
        """
        self._check_day_reset()
        
        # Calculate loss percentage (negative PnL = loss)
        loss_pct = abs(min(0, self.get_daily_pnl_pct()))
        
        if loss_pct >= max_loss_pct:
            self._was_limit_hit = True
            return True
        
        return False
    
    def was_limit_hit_today(self) -> bool:
        """Check if limit was hit at any point today (even if PnL recovered)."""
        self._check_day_reset()
        return self._was_limit_hit
    
    def get_trade_count(self) -> int:
        """Get number of trades today."""
        self._check_day_reset()
        return len(self._trade_history)
    
    def reset(self, new_starting_balance: float = None) -> None:
        """
        Force reset the tracker.
        
        Args:
            new_starting_balance: Optional new starting balance
        """
        if new_starting_balance is not None:
            self.starting_balance = new_starting_balance
        self._daily_pnl = 0.0
        self._trade_history = []
        self._current_date = self._get_utc_date()
        self._was_limit_hit = False
