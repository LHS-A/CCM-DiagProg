#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import get_task, load_config
from src.data.dataset import CCMManifestDataset
from src.models import UnifiedCausalCCM
from src.utils.huggingface import resolve_cached_model

IDENTITY_TASK = {0: "task1", 1: "task2", 2: "task3", 3: "task3", 4: "task4", 5: "task5"}
IDENTITY_TARGETS = {2: ["CFS", "TBUT", "SIT", "OSDI"], 3: ["HbA1c"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity", type=int, choices=range(6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-missingness", type=float, choices=(0.0, 0.5, 1.0), default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config); task = get_task(cfg, IDENTITY_TASK[args.identity])
    frame = pd.read_csv(args.manifest).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve_cached_model(cfg["model"]["clinical_encoder"]))
    dataset = CCMManifestDataset(frame, task, tokenizer, int(cfg["data"]["input_resolution"]),
                                 max_length=int(cfg["model"]["max_sequence_length"]),
                                 clinical_missingness=args.text_missingness)
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = UnifiedCausalCCM({**cfg["model"], "pretrained_visual": False})
    model.load_state_dict(payload["model_state"]); model.eval().to(args.device)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(args.device), batch["input_ids"].to(args.device),
                           batch["attention_mask"].to(args.device), args.identity,
                           batch["clinical_available"].to(args.device), stage=2)["prediction"]
            predictions.append(output.cpu().numpy())
    prediction = np.concatenate(predictions)
    names = IDENTITY_TARGETS.get(args.identity, task.get("classes", task.get("targets")))
    output = frame[[c for c in ("image_path", "patient_id", "split") if c in frame]].copy()
    for index, name in enumerate(names): output[f"prediction_{name}"] = prediction[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__": main()

