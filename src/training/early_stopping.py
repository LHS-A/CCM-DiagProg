from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingState:
    best_value: float
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    early_stop_epoch: Optional[int] = None
    reason: str = "max_epochs_reached"


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, min_epochs: int, mode: str = "min") -> None:
        if patience < 1 or min_epochs < 0 or min_delta < 0:
            raise ValueError("Invalid early-stopping configuration")
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.mode = mode
        initial = float("inf") if mode == "min" else -float("inf")
        self.state = EarlyStoppingState(best_value=initial)

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        improved = value < self.state.best_value - self.min_delta if self.mode == "min" else value > self.state.best_value + self.min_delta
        if improved:
            self.state.best_value = value
            self.state.best_epoch = epoch
            self.state.epochs_without_improvement = 0
        else:
            self.state.epochs_without_improvement += 1
        stop = epoch >= self.min_epochs and self.state.epochs_without_improvement >= self.patience
        if stop:
            self.state.early_stop_epoch = epoch
            self.state.reason = f"no improvement >= {self.min_delta:g} for {self.patience} consecutive epochs after min_epochs={self.min_epochs}"
        return stop, improved

