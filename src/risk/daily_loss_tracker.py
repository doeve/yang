"""
Daily loss tracking for risk management.

Tracks realized PnL per day and triggers trading pause when limit is hit.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Tuple


class DailyLossTracker:
    """
    Tracks realized PnL over a rolling window and enforces loss limits.

    Uses a rolling 24-hour window instead of daily resets.
    """

    def __init__(self, starting_balance: float = 1000.0, window_hours: int = 24):
        self.starting_balance = starting_balance
        self.window_hours = window_hours
        self._trade_history: List[Tuple[datetime, float]] = []  # (time, pnl)
        self._was_limit_hit = False
        self._limit_hit_time: datetime | None = None

    def _cleanup_old_trades(self) -> None:
        """Remove trades outside the rolling window."""
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=self.window_hours)

        # Keep only trades within the window
        self._trade_history = [
            (time, pnl) for time, pnl in self._trade_history
            if time >= cutoff_time
        ]

        # Reset limit hit flag if the limit hit time is outside the window
        if self._limit_hit_time and self._limit_hit_time < cutoff_time:
            self._was_limit_hit = False
            self._limit_hit_time = None
    
    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade's PnL.

        Args:
            pnl: Realized PnL from the trade (positive or negative)
        """
        self._cleanup_old_trades()
        self._trade_history.append((datetime.now(timezone.utc), pnl))

    def get_daily_pnl(self) -> float:
        """Get total realized PnL over the rolling window."""
        self._cleanup_old_trades()
        return sum(pnl for _, pnl in self._trade_history)
    
    def get_daily_pnl_pct(self) -> float:
        """Get PnL over the rolling window as percentage of starting balance."""
        if self.starting_balance <= 0:
            return 0.0
        return (self.get_daily_pnl() / self.starting_balance) * 100
    
    def is_limit_hit(self, max_loss_pct: float) -> bool:
        """
        Check if loss limit has been exceeded in the rolling window.

        Args:
            max_loss_pct: Maximum allowed loss as percentage (e.g., 5.0 for 5%)

        Returns:
            True if loss limit exceeded
        """
        self._cleanup_old_trades()

        # Calculate loss percentage (negative PnL = loss)
        loss_pct = abs(min(0, self.get_daily_pnl_pct()))

        if loss_pct >= max_loss_pct:
            if not self._was_limit_hit:
                self._limit_hit_time = datetime.now(timezone.utc)
            self._was_limit_hit = True
            return True

        return False

    def was_limit_hit_today(self) -> bool:
        """Check if limit was hit at any point within the rolling window (even if PnL recovered)."""
        self._cleanup_old_trades()
        return self._was_limit_hit

    def get_trade_count(self) -> int:
        """Get number of trades in the rolling window."""
        self._cleanup_old_trades()
        return len(self._trade_history)
    
    def reset(self, new_starting_balance: float = None) -> None:
        """
        Force reset the tracker.

        Args:
            new_starting_balance: Optional new starting balance
        """
        if new_starting_balance is not None:
            self.starting_balance = new_starting_balance
        self._trade_history = []
        self._was_limit_hit = False
        self._limit_hit_time = None
