#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import get_task, load_config
from src.data.dataset import CCMManifestDataset
from src.models import UnifiedCausalCCM
from src.utils.huggingface import resolve_cached_model
from src.utils.io import set_seed

IDENTITIES = [
    ("task1", 0, None), ("task2", 1, None),
    ("task3", 2, [1, 2, 3, 4]), ("task3", 3, [0]),
    ("task4", 4, None), ("task5", 5, None),
]


def loaders(cfg, task, manifest, tokenizer):
    frame = pd.read_csv(manifest)
    result = {}
    for split in ("train", "validation", "test"):
        subset = frame[frame.split.eq(split)].reset_index(drop=True)
        dataset = CCMManifestDataset(
            subset, task, tokenizer, int(cfg["data"]["input_resolution"]),
            max_length=int(cfg["model"]["max_sequence_length"]),
            clinical_missingness=None if split == "train" else float(cfg["model"]["clinical_missingness_eval"]),
        )
        result[split] = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]),
                                   shuffle=split == "train", num_workers=int(cfg["data"]["num_workers"]),
                                   pin_memory=torch.cuda.is_available())
    return result


def run_epoch(model, loader, optimizer, device, identity, indices, kind, stage, cfg):
    model.train(optimizer is not None); values = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device); ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device); available = batch["clinical_available"].to(device)
            target = batch["target"].to(device); mask = batch["target_mask"].to(device)
            if indices is not None: target, mask = target[:, indices], mask[:, indices]
            output = model(image, ids, attention, identity, available, stage)
            if kind == "classification": loss = F.cross_entropy(output["prediction"], target)
            else: loss = (output["prediction"][mask.bool()] - target[mask.bool()]).square().mean()
            loss = loss + float(cfg["model"]["structural_loss_weight"]) * output["prior_loss"]
            loss = loss + float(cfg["model"]["hyper_loss_weight"]) * output["hyper_loss"]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            values.append(float(loss.detach()))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/unified_six_task"))
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    cfg = load_config(args.config); set_seed(args.seed, True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for training")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(resolve_cached_model(cfg["model"]["clinical_encoder"]))
    model = UnifiedCausalCCM(cfg["model"]).to(device); args.output.mkdir(parents=True, exist_ok=True)
    history = []; global_epoch = 0
    for task_id, identity, indices in IDENTITIES:
        task = get_task(cfg, task_id); data = loaders(cfg, task, args.manifests_dir / f"{task_id}.csv", tokenizer)
        best = float("inf"); stale = 0; best_state = None
        stages = [(1, int(cfg["training"]["stage1_epochs"])),
                  (2, int(cfg["training"]["max_epochs"]) - int(cfg["training"]["stage1_epochs"]))]
        for stage, epochs in stages:
            if stage == 2:
                for parameter in model.visual_encoder.parameters(): parameter.requires_grad = False
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=float(cfg["training"]["learning_rate"]),
                                          weight_decay=float(cfg["training"]["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max(1, epochs), eta_min=float(cfg["training"]["min_learning_rate"]))
            for local_epoch in range(1, epochs + 1):
                global_epoch += 1
                train_loss = run_epoch(model, data["train"], optimizer, device, identity, indices, task["kind"], stage, cfg)
                val_loss = run_epoch(model, data["validation"], None, device, identity, indices, task["kind"], stage, cfg)
                scheduler.step()
                record = {"global_epoch": global_epoch, "identity": identity, "task": task_id,
                          "stage": stage, "epoch": local_epoch, "train_loss": train_loss,
                          "validation_loss": val_loss}
                history.append(record); print(record, flush=True)
                payload = {"model_state": model.state_dict(), "config": cfg, "identity": identity,
                           "global_epoch": global_epoch, "history": history}
                if global_epoch % int(cfg["training"]["checkpoint_interval_epochs"]) == 0:
                    torch.save(payload, args.output / f"epoch_{global_epoch:04d}.pt")
                if stage == 2:
                    if val_loss < best - float(cfg["training"]["early_stopping"]["min_delta"]):
                        best, stale = val_loss, 0
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        torch.save(payload, args.output / f"best_task_{identity}.pt")
                    else: stale += 1
                    if local_epoch >= int(cfg["training"]["early_stopping"]["min_epochs"]) and stale >= int(cfg["training"]["early_stopping"]["patience"]): break
        if best_state is not None: model.load_state_dict(best_state)
        for parameter in model.visual_encoder.parameters(): parameter.requires_grad = True
    torch.save({"model_state": model.state_dict(), "config": cfg, "history": history}, args.output / "best_model.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__": main()

