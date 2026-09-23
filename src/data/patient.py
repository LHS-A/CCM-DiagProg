from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


def patient_groups(patient_ids: Sequence[str]) -> tuple[list[str], list[list[int]]]:
    """Return stable patient order and row indices for each patient."""
    groups: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        key = str(patient)
        if not key.strip():
            raise ValueError("patient_id cannot be empty")
        groups.setdefault(key, []).append(index)
    return list(groups), list(groups.values())


def patient_mean(values: torch.Tensor, patient_ids: Sequence[str]) -> tuple[torch.Tensor, list[str]]:
    """Average image-level tensors within patient, ignoring NaN values."""
    names, groups = patient_groups(patient_ids)
    rows = []
    for indices in groups:
        current = values[torch.as_tensor(indices, device=values.device)]
        finite = torch.isfinite(current)
        count = finite.sum(0)
        total = torch.where(finite, current, torch.zeros_like(current)).sum(0)
        rows.append(torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float("nan"))))
    return torch.stack(rows), names


def patient_targets(
    values: torch.Tensor,
    masks: torch.Tensor,
    patient_ids: Sequence[str],
    categorical: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Create one target row per patient and reject inconsistent labels."""
    names, groups = patient_groups(patient_ids)
    targets, valid = [], []
    for patient, indices in zip(names, groups):
        index = torch.as_tensor(indices, device=values.device)
        current, current_mask = values[index], masks[index].bool()
        if categorical:
            observed = current.reshape(-1)
            if observed.unique().numel() != 1:
                raise ValueError(f"inconsistent labels for patient {patient}")
            targets.append(observed[0])
            valid.append(torch.tensor(True, device=values.device))
        else:
            finite = current_mask & torch.isfinite(current)
            count = finite.sum(0)
            total = torch.where(finite, current, torch.zeros_like(current)).sum(0)
            targets.append(torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total)))
            valid.append(count > 0)
    return torch.stack(targets), torch.stack(valid), names


def regression_normalizer(
    values: torch.Tensor, masks: torch.Tensor, patient_ids: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only, patient-weighted mean/std for every regression target."""
    aggregated, valid, _ = patient_targets(values, masks, patient_ids, categorical=False)
    numeric = torch.where(valid, aggregated, torch.full_like(aggregated, float("nan")))
    mean = torch.nanmean(numeric, dim=0)
    centered = torch.where(valid, numeric - mean, torch.zeros_like(numeric))
    count = valid.sum(0)
    variance = centered.square().sum(0) / count.clamp_min(1)
    std = variance.sqrt()
    if not torch.isfinite(mean).all() or bool((count == 0).any()):
        raise ValueError("every regression target needs a finite training value")
    return mean, std.clamp_min(1e-8)


class PatientBalancedBatchSampler(Sampler):
    """Uniformly sample patients, then one of their images, for each task epoch."""

    def __init__(self, patient_ids: Sequence[str], batch_size: int, seed: int = 0) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.names, self.groups = patient_groups(patient_ids)
        if not self.names:
            raise ValueError("patient-balanced sampling requires patients")
        self.batch_size = int(batch_size)
        self.samples_per_epoch = len(patient_ids)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return ceil(self.samples_per_epoch / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order: list[int] = []
        while len(order) < self.samples_per_epoch:
            order.extend(rng.permutation(len(self.names)).tolist())
        order = order[: self.samples_per_epoch]
        sampled = [int(rng.choice(self.groups[patient])) for patient in order]
        for start in range(0, len(sampled), self.batch_size):
            yield sampled[start : start + self.batch_size]


def aggregate_predictions(
    prediction: np.ndarray,
    patient_ids: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Average image-level probabilities or continuous predictions per patient."""
    tensor = torch.as_tensor(np.asarray(prediction), dtype=torch.float64)
    aggregated, names = patient_mean(tensor, patient_ids)
    return aggregated.numpy(), names
