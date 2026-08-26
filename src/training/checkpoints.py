from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class SpacedCheckpointKeeper:
    """Retain every configured Nth epoch checkpoint through training end."""

    def __init__(self, directory: Path, interval: int) -> None:
        if interval < 1:
            raise ValueError("interval must be positive")
        self.directory, self.interval = directory, interval
        self.directory.mkdir(parents=True, exist_ok=True)
        self.epochs: list[int] = []

    def save(self, epoch: int, payload: dict[str, Any]) -> None:
        if epoch % self.interval:
            return
        torch.save(payload, self.directory / f"epoch_{epoch:04d}.pt")
        self.epochs.append(epoch)
