from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CCMManifestDataset
from src.models import build_model
from src.utils.huggingface import resolve_cached_model


def tokenizer_for(config: dict[str, Any], local_only: bool = False):
    name = resolve_cached_model(config["model"]["clinical_encoder"])
    return AutoTokenizer.from_pretrained(name, local_files_only=local_only)


def make_loaders(config: dict[str, Any], task: dict[str, Any], manifest_path: Path):
    frame = pd.read_csv(manifest_path)
    tokenizer = None if task.get("image_only", False) else tokenizer_for(config)
    loaders = {}
    for split in ["train", "validation", "test"]:
        subset = frame[frame.split == split]
        loaders[split] = make_loader_for_frame(config, task, subset, tokenizer, shuffle=split == "train")
    train_sd = None
    if task["kind"] == "regression":
        train_sd = frame.loc[frame.split == "train", task["targets"]].to_numpy(float).std(axis=0, ddof=1)
    return loaders, train_sd


def make_loader_for_frame(config: dict[str, Any], task: dict[str, Any], frame: pd.DataFrame, tokenizer=None, shuffle: bool = False):
    tokenizer = tokenizer if task.get("image_only", False) else (tokenizer or tokenizer_for(config))
    dataset = CCMManifestDataset(
        frame, task, tokenizer, int(config["data"]["input_resolution"]),
        max_length=int(config["model"]["max_sequence_length"]),
    )
    return DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=shuffle,
        num_workers=int(config["data"]["num_workers"]), pin_memory=torch.cuda.is_available(),
    )


def load_final_model(checkpoint: Path, task: dict[str, Any], config: dict[str, Any], device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    model_config = dict(config["model"])
    # A complete checkpoint already contains encoder weights and must not access the network.
    model_config["pretrained_visual"] = False
    model_config["pretrained_text"] = False
    model_config["text_encoder_config"] = payload.get("text_encoder_config")
    model = build_model(task, model_config)
    selected = payload["model_state"].get("selected_channels")
    if selected is not None and selected.shape != model.selected_channels.shape:
        model.set_selected_channels(selected)
    model.load_state_dict(payload["model_state"], strict=True)
    if not task.get("image_only", False):
        for parameter in model.visual_encoder.parameters():
            parameter.requires_grad = False
    return model.to(device)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n", encoding="utf-8")
