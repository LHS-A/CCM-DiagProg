from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import resnet50


class ResNet50Features(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        try:
            network = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:  # torchvision < 0.13
            network = resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layers = nn.Sequential(network.layer1, network.layer2, network.layer3, network.layer4)
        self.out_channels = 2048

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(self.stem(image))


class StructuralPrior(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("prior", torch.eye(channels), persistent=True)
        self.register_buffer("prior_ready", torch.tensor(False), persistent=True)

    @torch.no_grad()
    def set_prior(self, prior: Tensor) -> None:
        if prior.shape != self.prior.shape:
            raise ValueError(f"Expected prior shape {tuple(self.prior.shape)}, got {tuple(prior.shape)}")
        self.prior.copy_(prior)
        self.prior_ready.fill_(True)

    def loss(self, features: Tensor) -> Tensor:
        if not bool(self.prior_ready):
            return features.new_zeros(())
        flat = features.flatten(2)
        minimum = flat.amin(dim=-1, keepdim=True)
        normalized = (flat - minimum) / (flat.amax(dim=-1, keepdim=True) - minimum + 1e-6)
        numerator = torch.einsum("bcp,bdp->bcd", normalized, normalized)
        energy = normalized.square().sum(-1)
        denominator = (energy[:, :, None] * energy[:, None, :] + 1e-6).sqrt()
        dependency = (numerator / denominator).mean(0)
        eye = torch.eye(dependency.shape[0], device=dependency.device, dtype=dependency.dtype)
        dependency = dependency * (1 - eye)
        prior = self.prior.to(dependency.dtype) * (1 - eye)
        dependency = dependency / (dependency.norm() + 1e-6)
        prior = prior / (prior.norm() + 1e-6)
        return (dependency - prior.detach()).square().mean()
