"""
Daily loss tracking for risk management.

Tracks realized PnL per day and triggers trading pause when limit is hit.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Literal


class DailyLossTracker:
    """
    Tracks realized PnL over a rolling window and enforces loss limits.

    Uses a rolling 24-hour window instead of daily resets.
    Maintains separate tracking for live and paper modes.
    """

    def __init__(self, starting_balance: float = 1000.0, window_hours: int = 24):
        self.starting_balance = starting_balance
        self.window_hours = window_hours
        self._trade_history: List[Tuple[datetime, float, str]] = []  # (time, pnl, mode)
        self._was_limit_hit_live = False
        self._was_limit_hit_paper = False
        self._limit_hit_time_live: datetime | None = None
        self._limit_hit_time_paper: datetime | None = None

    def _cleanup_old_trades(self) -> None:
        """Remove trades outside the rolling window."""
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=self.window_hours)

        # Keep only trades within the window
        self._trade_history = [
            (time, pnl, mode) for time, pnl, mode in self._trade_history
            if time >= cutoff_time
        ]

        # Reset limit hit flags if their respective times are outside the window
        if self._limit_hit_time_live and self._limit_hit_time_live < cutoff_time:
            self._was_limit_hit_live = False
            self._limit_hit_time_live = None

        if self._limit_hit_time_paper and self._limit_hit_time_paper < cutoff_time:
            self._was_limit_hit_paper = False
            self._limit_hit_time_paper = None
    
    def record_trade(self, pnl: float, mode: Literal["live", "paper"] = "paper") -> None:
        """
        Record a completed trade's PnL.

        Args:
            pnl: Realized PnL from the trade (positive or negative)
            mode: Trading mode for this trade ("live" or "paper")
        """
        self._cleanup_old_trades()
        self._trade_history.append((datetime.now(timezone.utc), pnl, mode))

    def get_daily_pnl(self, mode: Literal["live", "paper", "all"] = "all") -> float:
        """
        Get total realized PnL over the rolling window.

        Args:
            mode: Filter by mode - "live", "paper", or "all" for combined
        """
        self._cleanup_old_trades()
        if mode == "all":
            return sum(pnl for _, pnl, _ in self._trade_history)
        else:
            return sum(pnl for _, pnl, m in self._trade_history if m == mode)

    def get_daily_pnl_pct(self, mode: Literal["live", "paper", "all"] = "all") -> float:
        """
        Get PnL over the rolling window as percentage of starting balance.

        Args:
            mode: Filter by mode - "live", "paper", or "all" for combined
        """
        if self.starting_balance <= 0:
            return 0.0
        return (self.get_daily_pnl(mode=mode) / self.starting_balance) * 100
    
    def is_limit_hit(self, max_loss_pct: float, mode: Literal["live", "paper"] = "paper") -> bool:
        """
        Check if loss limit has been exceeded in the rolling window for the specified mode.

        Args:
            max_loss_pct: Maximum allowed loss as percentage (e.g., 5.0 for 5%)
            mode: Trading mode to check ("live" or "paper")

        Returns:
            True if loss limit exceeded for this mode
        """
        self._cleanup_old_trades()

        # Calculate loss percentage for this mode only (negative PnL = loss)
        loss_pct = abs(min(0, self.get_daily_pnl_pct(mode=mode)))

        if loss_pct >= max_loss_pct:
            if mode == "live":
                if not self._was_limit_hit_live:
                    self._limit_hit_time_live = datetime.now(timezone.utc)
                self._was_limit_hit_live = True
            else:  # paper
                if not self._was_limit_hit_paper:
                    self._limit_hit_time_paper = datetime.now(timezone.utc)
                self._was_limit_hit_paper = True
            return True

        return False

    def was_limit_hit_today(self, mode: Literal["live", "paper"] = "paper") -> bool:
        """
        Check if limit was hit at any point within the rolling window for the specified mode.

        Args:
            mode: Trading mode to check ("live" or "paper")

        Returns:
            True if limit was hit for this mode (even if PnL has recovered)
        """
        self._cleanup_old_trades()
        return self._was_limit_hit_live if mode == "live" else self._was_limit_hit_paper

    def get_trade_count(self) -> int:
        """Get number of trades in the rolling window."""
        self._cleanup_old_trades()
        return len(self._trade_history)
    
    def reset(self, new_starting_balance: float = None, mode: Literal["live", "paper", "all"] = "all") -> None:
        """
        Force reset the tracker.

        Args:
            new_starting_balance: Optional new starting balance
            mode: Which mode to reset - "live", "paper", or "all" (default)
        """
        if new_starting_balance is not None:
            self.starting_balance = new_starting_balance

        if mode == "all":
            self._trade_history = []
            self._was_limit_hit_live = False
            self._was_limit_hit_paper = False
            self._limit_hit_time_live = None
            self._limit_hit_time_paper = None
        elif mode == "live":
            # Remove only live trades
            self._trade_history = [(t, pnl, m) for t, pnl, m in self._trade_history if m != "live"]
            self._was_limit_hit_live = False
            self._limit_hit_time_live = None
        else:  # paper
            # Remove only paper trades
            self._trade_history = [(t, pnl, m) for t, pnl, m in self._trade_history if m != "paper"]
            self._was_limit_hit_paper = False
            self._limit_hit_time_paper = None
