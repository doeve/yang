"""Training infrastructure."""

from .trainer import TrainingOrchestrator, TrainingConfig
from .callbacks import (
    MetricsCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateScheduleCallback,
    TensorBoardCallback,
)
from .evaluation import Evaluator, BacktestResult, Trade, generate_report

__all__ = [
    "TrainingOrchestrator",
    "TrainingConfig",
    "MetricsCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LearningRateScheduleCallback",
    "TensorBoardCallback",
    "Evaluator",
    "BacktestResult",
    "Trade",
    "generate_report",
]
