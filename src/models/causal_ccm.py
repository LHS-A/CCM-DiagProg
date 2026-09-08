from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoConfig, AutoModel, BertConfig, BertModel
from src.utils.huggingface import resolve_cached_model


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
        normalized = normalized / (normalized.square().sum(-1, keepdim=True).sqrt() + 1e-6)
        dependency = torch.einsum("bcp,bdp->cd", normalized, normalized) / features.shape[0]
        eye = torch.eye(dependency.shape[0], device=dependency.device, dtype=dependency.dtype)
        dependency = dependency * (1 - eye)
        prior = self.prior.to(dependency.dtype) * (1 - eye)
        dependency = dependency / (dependency.norm() + 1e-6)
        prior = prior / (prior.norm() + 1e-6)
        return (dependency - prior.detach()).square().mean()


class SemanticAlignment(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, attention_dim)
        self.text_projection = nn.Linear(text_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(attention_dim)

    def forward(self, features: Tensor, text_tokens: Tensor, attention_mask: Tensor) -> Tensor:
        visual = features.flatten(2).transpose(1, 2)
        visual = self.visual_projection(visual)
        text = self.text_projection(text_tokens)
        attended, _ = self.attention(visual, text, text, key_padding_mask=~attention_mask.bool())
        return self.norm(visual + attended).mean(dim=1)


class DynamicPredictionHead(nn.Module):
    def __init__(self, feature_dim: int, text_dim: int, output_dim: int, task_count: int,
                 task_dim: int, hidden_dim: int, trunk_width: int = 512,
                 generator_hidden: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.task_embedding = nn.Embedding(task_count, task_dim)
        self.hypernetwork = nn.Sequential(
            nn.Linear(text_dim + task_dim, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, hidden_dim), nn.GELU()
        )
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, generator_hidden), nn.GELU(),
            nn.Linear(generator_hidden, feature_dim * output_dim + output_dim),
        )

    def forward(self, features: Tensor, patient: Tensor, task_id: Tensor) -> tuple[Tensor, Tensor]:
        task = self.task_embedding(task_id)
        code = self.hypernetwork(torch.cat([task, patient], dim=-1))
        dynamic = self.generator(code)
        weights = dynamic[:, : self.feature_dim * self.output_dim].view(-1, self.feature_dim, self.output_dim)
        bias = dynamic[:, self.feature_dim * self.output_dim :]
        prediction = torch.bmm(features.unsqueeze(1), weights).squeeze(1) + bias
        regularization = weights.square().mean() + bias.square().mean()
        return prediction, regularization


class CausalCCM(nn.Module):
    """Complete structural, semantic, hypernetwork, and dynamic-head path."""

    def __init__(self, task: dict[str, Any], model_config: dict[str, Any], task_index: int) -> None:
        super().__init__()
        self.task = task
        self.task_index = task_index
        self.image_only = bool(task.get("image_only", False))
        self.visual_encoder = ResNet50Features(bool(model_config["pretrained_visual"]))
        if self.image_only:
            self.image_only_head = nn.Linear(self.visual_encoder.out_channels, int(task["output_dim"]))
            return
        text_name = resolve_cached_model(model_config["clinical_encoder"])
        if model_config["pretrained_text"]:
            self.text_encoder = AutoModel.from_pretrained(text_name)
        else:
            if model_config.get("text_encoder_config"):
                text_config = BertConfig.from_dict(model_config["text_encoder_config"])
            else:
                try:
                    text_config = AutoConfig.from_pretrained(text_name, local_files_only=True)
                except Exception:
                    text_config = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
            self.text_encoder = AutoModel.from_config(text_config)
        text_dim = int(self.text_encoder.config.hidden_size)
        channels = self.visual_encoder.out_channels
        self.structural_prior = StructuralPrior(channels)
        retained = max(1, round(channels * float(model_config["retained_channel_ratio"])))
        self.register_buffer("selected_channels", torch.arange(retained), persistent=True)
        attention_dim = int(model_config["attention_dim"])
        self.alignment = SemanticAlignment(retained, text_dim, attention_dim)
        self.dynamic_head = DynamicPredictionHead(
            attention_dim, text_dim, int(task["output_dim"]), int(model_config["task_count"]),
            int(model_config["task_embedding_dim"]), int(model_config["hyper_hidden_dim"]),
            int(model_config["hyper_trunk_dims"][0]), int(model_config["generator_hidden_dim"]),
        )
        self.auxiliary_head = nn.Linear(channels, int(task["output_dim"]))

    def set_selected_channels(self, indices: Tensor) -> None:
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("Selected channels must be a non-empty 1D tensor")
        self.selected_channels = indices.to(device=self.selected_channels.device, dtype=torch.long)
        # Screening changes the projection input dimensionality between stages.
        old = self.alignment.visual_projection
        self.alignment.visual_projection = nn.Linear(len(indices), old.out_features).to(old.weight.device)

    def forward(
        self,
        image: Tensor,
        input_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        task_id: Optional[Tensor] = None,
        stage: int = 2,
    ) -> dict[str, Tensor]:
        visual = self.visual_encoder(image)
        if self.image_only:
            prediction = self.image_only_head(F.adaptive_avg_pool2d(visual, 1).flatten(1))
            zero = visual.new_zeros(())
            return {"prediction": prediction, "structural_loss": zero, "hyper_loss": zero}
        structural_loss = self.structural_prior.loss(visual) if stage == 1 else visual.new_zeros(())
        auxiliary = self.auxiliary_head(F.adaptive_avg_pool2d(visual, 1).flatten(1))
        if stage == 1:
            return {"prediction": auxiliary, "structural_loss": structural_loss, "hyper_loss": visual.new_zeros(())}
        selected = visual.index_select(1, self.selected_channels)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_output.last_hidden_state
        patient = text_tokens[:, 0]
        aligned = self.alignment(selected, text_tokens, attention_mask)
        if task_id is None:
            task_id = torch.full((image.shape[0],), self.task_index, device=image.device, dtype=torch.long)
        prediction, hyper_loss = self.dynamic_head(aligned, patient, task_id)
        return {"prediction": prediction, "structural_loss": structural_loss, "hyper_loss": hyper_loss}


def build_model(task: dict[str, Any], model_config: dict[str, Any]) -> CausalCCM:
    task_index = {"task1": 0, "task2": 1, "task3": 2, "task4": 4, "task5": 5}[task["id"]]
    return CausalCCM(task, model_config, task_index)
