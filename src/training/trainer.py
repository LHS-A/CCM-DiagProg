from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models.screening import screen_channels
from src.training.checkpoints import SpacedCheckpointKeeper
from src.training.early_stopping import EarlyStopping
from src.utils.io import ensure_dir


def masked_task_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "classification":
        return nn.functional.cross_entropy(prediction, target)
    valid = mask.bool()
    if not valid.any():
        raise RuntimeError("Batch has no valid regression targets")
    return (prediction[valid] - target[valid]).square().mean()


class Trainer:
    def __init__(self, model: nn.Module, task: dict[str, Any], config: dict[str, Any], device: torch.device) -> None:
        self.model, self.task, self.config, self.device = model.to(device), task, config, device
        train = config["training"]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(train["max_epochs"]), eta_min=float(train["min_learning_rate"])
        )

    def _epoch(self, loader: DataLoader, train: bool, stage: int) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        losses, targets, predictions = [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                tensor = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                output = self.model(tensor["image"], tensor["input_ids"], tensor["attention_mask"], stage=stage)
                task_loss = masked_task_loss(output["prediction"], tensor["target"], tensor["target_mask"], self.task["kind"])
                loss = task_loss + float(self.config["model"]["structural_loss_weight"]) * output["structural_loss"]
                if stage == 2:
                    loss = loss + float(self.config["model"]["hyper_loss_weight"]) * output["hyper_loss"]
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                losses.append(float(loss.detach()))
                targets.append(tensor["target"].detach().cpu().numpy())
                predictions.append(output["prediction"].detach().cpu().numpy())
        return float(np.mean(losses)), np.concatenate(targets), np.concatenate(predictions)

    @torch.no_grad()
    def _estimate_structural_prior(self, loader: DataLoader) -> None:
        """Estimate Pi A Pi^T from training-only clinical variables after visual warm-up."""
        self.model.eval()
        visual_rows, clinical_rows = [], []
        for batch in loader:
            clinical = batch["clinical_structured"].float()
            if clinical.shape[1] == 0:
                raise RuntimeError("Structural-prior estimation requires released non-target clinical metadata")
            features = self.model.visual_encoder(batch["image"].to(self.device))
            visual_rows.append(torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1).cpu())
            clinical_rows.append(clinical)
        visual = torch.cat(visual_rows).float()
        clinical = torch.cat(clinical_rows).float()
        visual = (visual - visual.mean(0)) / (visual.std(0) + 1e-6)
        clinical = (clinical - clinical.mean(0)) / (clinical.std(0) + 1e-6)
        association = (visual.T @ clinical / max(len(visual) - 1, 1)).abs()
        assignment = torch.softmax(association / float(self.config["model"]["structural_temperature"]), dim=1)
        adjacency = (clinical.T @ clinical / max(len(clinical) - 1, 1)).abs()
        prior = assignment @ adjacency @ assignment.T
        self.model.structural_prior.set_prior(prior.to(self.device))

    @torch.no_grad()
    def _screen_channels(self, loader: DataLoader) -> None:
        self.model.eval()
        descriptors, targets, nuisance = [], [], []
        limit = int(self.config["model"]["screening_max_samples"])
        seen = 0
        for batch in loader:
            image = batch["image"].to(self.device)
            features = self.model.visual_encoder(image)
            descriptor = torch.stack([features.mean((2, 3)), features.amax((2, 3))], dim=-1)
            take = min(len(image), limit - seen)
            descriptors.append(descriptor[:take])
            targets.append(batch["target"][:take].to(self.device))
            nuisance.append(batch["clinical_structured"][:take].to(self.device).float())
            seen += take
            if seen >= limit:
                break
        indices = screen_channels(
            torch.cat(descriptors), torch.cat(targets), torch.cat(nuisance),
            self.task["kind"] == "classification", float(self.config["model"]["retained_channel_ratio"]),
            float(self.config["model"]["screening_fdr"]), int(self.config["model"]["screening_permutations"]),
            float(self.config["model"]["kci_regularization"]), 3407,
        )
        self.model.set_selected_channels(indices)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, output_dir: Path) -> dict[str, Any]:
        output_dir = ensure_dir(output_dir)
        train_cfg = self.config["training"]
        stage1_epochs = 0 if self.task.get("image_only", False) else min(int(train_cfg["stage1_epochs"]), int(train_cfg["max_epochs"]))
        warmup_epochs = min(int(train_cfg.get("visual_warmup_epochs", 15)), stage1_epochs)
        keeper = SpacedCheckpointKeeper(output_dir / "spaced_checkpoints", int(train_cfg.get("checkpoint_interval_epochs", 10)))

        def payload(epoch: int, stage: int) -> dict[str, Any]:
            return {
                "model_state": self.model.state_dict(), "task": self.task, "config": self.config,
                "epoch": epoch, "stage": stage,
                "text_encoder_config": self.model.text_encoder.config.to_dict() if hasattr(self.model, "text_encoder") else None,
            }

        history = []
        for epoch in range(1, stage1_epochs + 1):
            if epoch == warmup_epochs + 1:
                self._estimate_structural_prior(train_loader)
            train_loss, _, _ = self._epoch(train_loader, True, stage=1)
            val_loss, _, _ = self._epoch(val_loader, False, stage=1)
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise FloatingPointError(f"Numerical failure during Stage I at epoch {epoch}")
            self.scheduler.step()
            history.append({"epoch": epoch, "stage": 1, "train_loss": train_loss, "validation_loss": val_loss})
            keeper.save(epoch, payload(epoch, 1))
        if not self.task.get("image_only", False):
            if not bool(self.model.structural_prior.prior_ready):
                self._estimate_structural_prior(train_loader)
            self._screen_channels(train_loader)
            for parameter in self.model.visual_encoder.parameters():
                parameter.requires_grad = False
            # Stage-II modules (including the newly dimensioned projection) get a fresh optimizer.
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable, lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, int(train_cfg["max_epochs"]) - stage1_epochs),
                eta_min=float(train_cfg["min_learning_rate"]),
            )
        early_cfg = train_cfg["early_stopping"]
        # min_epochs is expressed on the complete two-stage timeline.
        effective_min_epochs = max(0, int(early_cfg["min_epochs"]) - stage1_epochs)
        stopper = EarlyStopping(int(early_cfg["patience"]), float(early_cfg["min_delta"]), effective_min_epochs, "min")
        checkpoint = output_dir / "best_model.pt"
        final_epoch = stage1_epochs
        for local_epoch in range(1, int(train_cfg["max_epochs"]) - stage1_epochs + 1):
            epoch = stage1_epochs + local_epoch
            train_loss, _, _ = self._epoch(train_loader, True, stage=2)
            val_loss, target, prediction = self._epoch(val_loader, False, stage=2)
            self.scheduler.step()
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                history.append({"epoch": epoch, "stage": 2, "train_loss": train_loss, "validation_loss": val_loss, "status": "numerical_failure"})
                stopper.state.early_stop_epoch = local_epoch
                stopper.state.reason = f"numerical failure at epoch {epoch}"
                final_epoch = epoch
                break
            if self.task["kind"] == "classification":
                val_metrics = classification_metrics(target, prediction, self.task["classes"])
            else:
                val_metrics = regression_metrics(target, prediction, self.task["targets"])
            record = {"epoch": epoch, "stage": 2, "train_loss": train_loss, "validation_loss": val_loss, **val_metrics}
            history.append(record)
            keeper.save(epoch, payload(epoch, 2))
            stop, improved = stopper.step(val_loss, local_epoch)
            if improved:
                torch.save(payload(epoch, 2), checkpoint)
            final_epoch = epoch
            if bool(early_cfg["enabled"]) and stop:
                stopper.state.reason = (
                    f"no improvement >= {float(early_cfg['min_delta']):g} for {int(early_cfg['patience'])} "
                    f"consecutive epochs after configured min_epochs={int(early_cfg['min_epochs'])}"
                )
                break
        state = stopper.state
        result = {
            "best_epoch": stage1_epochs + state.best_epoch,
            "early_stop_epoch": final_epoch if state.early_stop_epoch is not None else None,
            "epochs_without_improvement": state.epochs_without_improvement,
            "best_validation_value": state.best_value,
            "best_validation_loss": state.best_value,
            "early_stopping_reason": state.reason,
            "epochs_completed": final_epoch,
            "checkpoint": str(checkpoint),
            "spaced_checkpoints": [str(output_dir / "spaced_checkpoints" / f"epoch_{epoch:04d}.pt") for epoch in keeper.epochs],
        }
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, default=float) + "\n", encoding="utf-8")
        (output_dir / "RESULTS.md").write_text(
            "# Training result\n\n" + "\n".join(f"- {key}: {value}" for key, value in result.items()) + "\n",
            encoding="utf-8",
        )
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
        return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, task: dict[str, Any], device: torch.device, train_sd: np.ndarray | None = None) -> dict[str, Any]:
    model.eval()
    targets, predictions = [], []
    for batch in loader:
        output = model(batch["image"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device), stage=2)
        targets.append(batch["target"].numpy())
        predictions.append(output["prediction"].cpu().numpy())
    target, prediction = np.concatenate(targets), np.concatenate(predictions)
    if task["kind"] == "classification":
        return classification_metrics(target, prediction, task["classes"])
    return regression_metrics(target, prediction, task["targets"], train_sd)
