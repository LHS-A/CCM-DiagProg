from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _rbf(values: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(values.float(), values.float()).square()
    positive = distance[distance > 0]
    bandwidth = positive.median().clamp_min(1e-6) if len(positive) else distance.new_tensor(1.0)
    return torch.exp(-distance / (2 * bandwidth))


def _center(kernel: torch.Tensor) -> torch.Tensor:
    return kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True) + kernel.mean()


def _target_kernel(target: torch.Tensor, categorical: bool) -> torch.Tensor:
    if categorical:
        return target[:, None].eq(target[None, :]).float()
    return _rbf(target.float().reshape(len(target), -1))


def _bh_mask(p_values: torch.Tensor, alpha: float) -> torch.Tensor:
    order = torch.argsort(p_values)
    thresholds = alpha * torch.arange(1, len(p_values) + 1, device=p_values.device) / len(p_values)
    passed = p_values[order] <= thresholds
    mask = torch.zeros_like(p_values, dtype=torch.bool)
    if passed.any():
        cutoff = torch.where(passed)[0].max()
        mask[order[: cutoff + 1]] = True
    return mask


@torch.no_grad()
def screen_channels(
    descriptors: torch.Tensor,
    target: torch.Tensor,
    nuisance: torch.Tensor,
    categorical: bool,
    retention_ratio: float,
    alpha: float,
    permutations: int,
    regularization: float,
    seed: int,
) -> torch.Tensor:
    """Paper-aligned HSIC relevance then KCI conditional-robustness screening."""
    device = descriptors.device
    n, channels, _ = descriptors.shape
    target_kernel = _center(_target_kernel(target, categorical))
    nuisance_kernel = _center(_rbf(nuisance))
    identity = torch.eye(n, device=device)
    residual = identity - nuisance_kernel @ torch.linalg.solve(nuisance_kernel + regularization * identity, identity)
    generator = torch.Generator(device=device).manual_seed(seed)
    permutations_index = [torch.randperm(n, generator=generator, device=device) for _ in range(permutations)]
    hsic_stats, hsic_p, kci_stats, kci_p = [], [], [], []
    for channel in range(channels):
        feature_kernel = _center(_rbf(descriptors[:, channel]))
        hsic = (feature_kernel * target_kernel).sum() / max((n - 1) ** 2, 1)
        hsic_null = torch.stack([(feature_kernel * target_kernel[p][:, p]).sum() / max((n - 1) ** 2, 1) for p in permutations_index])
        hsic_stats.append(hsic)
        hsic_p.append((1 + (hsic_null >= hsic).sum()).float() / (permutations + 1))
        joint = _center(feature_kernel * nuisance_kernel)
        conditional_x = residual @ joint @ residual
        conditional_y = residual @ target_kernel @ residual
        kci = (conditional_x * conditional_y.T).sum() / n
        kci_null = torch.stack([
            (conditional_x * (residual @ target_kernel[p][:, p] @ residual).T).sum() / n for p in permutations_index
        ])
        kci_stats.append(kci)
        kci_p.append((1 + (kci_null >= kci).sum()).float() / (permutations + 1))
    hsic_stats, hsic_p = torch.stack(hsic_stats), torch.stack(hsic_p)
    kci_stats, kci_p = torch.stack(kci_stats), torch.stack(kci_p)
    candidates = _bh_mask(hsic_p, alpha) & _bh_mask(kci_p, alpha)
    budget = max(1, round(channels * retention_ratio))
    indices = torch.where(candidates)[0]
    if len(indices) == 0:
        raise RuntimeError("No channels survived FDR-corrected HSIC/KCI screening; do not silently substitute static channels")
    if len(indices) > budget:
        indices = indices[torch.topk(kci_stats[indices], budget).indices]
    return indices.sort().values

