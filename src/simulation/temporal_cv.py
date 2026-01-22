"""
Temporal K-Fold Cross-Validation for Time Series Data.

Implements expanding window cross-validation where training data
always comes before validation data to prevent look-ahead bias.
"""

from typing import Iterator, Tuple, List, Optional
import numpy as np
import pandas as pd


class TemporalKFold:
    """
    K-fold temporal cross-validation with expanding training window.
    
    Unlike standard k-fold, this ensures training data always precedes
    validation data chronologically.
    
    Example with 5 folds and 100 samples:
        Fold 1: Train [0:60]    Val [60:68]
        Fold 2: Train [0:68]    Val [68:76]
        Fold 3: Train [0:76]    Val [76:84]
        Fold 4: Train [0:84]    Val [84:92]
        Fold 5: Train [0:92]    Val [92:100]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        val_ratio: float = 0.2,
        min_train_size: int = 50,
    ):
        """
        Initialize temporal k-fold splitter.
        
        Args:
            n_splits: Number of folds
            val_ratio: Fraction of data to use for total validation
            min_train_size: Minimum training samples required
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError("val_ratio must be between 0 and 1")
            
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.min_train_size = min_train_size
    
    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indices for each fold.
        
        Args:
            X: Data array or DataFrame (only used to determine size)
            y: Target array (unused, for sklearn compatibility)
            groups: Group labels (unused, for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        n_samples = len(X)
        
        # Calculate validation window size
        total_val_samples = int(n_samples * self.val_ratio)
        val_per_fold = max(1, total_val_samples // self.n_splits)
        
        # Start training set at minimum size
        train_end = max(self.min_train_size, n_samples - total_val_samples)
        
        for fold_idx in range(self.n_splits):
            # Expand training set each fold
            fold_train_end = train_end + fold_idx * val_per_fold
            fold_val_start = fold_train_end
            fold_val_end = min(fold_val_start + val_per_fold, n_samples)
            
            if fold_val_end <= fold_val_start:
                break
            
            train_indices = np.arange(0, fold_train_end)
            val_indices = np.arange(fold_val_start, fold_val_end)
            
            yield train_indices, val_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits


class SlidingWindowKFold:
    """
    K-fold with sliding window (fixed training size).
    
    Unlike TemporalKFold, this keeps training window size fixed
    and slides it forward for each fold.
    
    Example with 5 folds and 100 samples:
        Fold 1: Train [0:60]    Val [60:68]
        Fold 2: Train [8:68]    Val [68:76]
        Fold 3: Train [16:76]   Val [76:84]
        ...
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.6,
        val_size: float = 0.1,
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
    
    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        train_len = int(n_samples * self.train_size)
        val_len = int(n_samples * self.val_size)
        
        step = (n_samples - train_len - val_len) // (self.n_splits - 1)
        
        for fold_idx in range(self.n_splits):
            offset = fold_idx * step
            train_start = offset
            train_end = offset + train_len
            val_start = train_end
            val_end = min(val_start + val_len, n_samples)
            
            if val_end <= val_start:
                break
            
            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def print_fold_info(
    data: pd.DataFrame,
    cv: TemporalKFold | SlidingWindowKFold,
    timestamp_col: str = "timestamp",
):
    """Print fold information for debugging."""
    print(f"Total samples: {len(data)}")
    print(f"Folds: {cv.n_splits}")
    print()
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data)):
        train_start_ts = data.iloc[train_idx[0]][timestamp_col]
        train_end_ts = data.iloc[train_idx[-1]][timestamp_col]
        val_start_ts = data.iloc[val_idx[0]][timestamp_col]
        val_end_ts = data.iloc[val_idx[-1]][timestamp_col]
        
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train: [{train_idx[0]}:{train_idx[-1]+1}] ({len(train_idx)} samples)")
        print(f"  Val:   [{val_idx[0]}:{val_idx[-1]+1}] ({len(val_idx)} samples)")
        print()
