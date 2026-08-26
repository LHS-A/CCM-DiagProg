from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CCMManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        task: dict[str, Any],
        tokenizer: Any,
        resolution: int = 384,
        transform: Optional[Callable] = None,
        max_length: int = 128,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.task = task
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.transform = transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def _image(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.resolution, self.resolution), Image.BILINEAR)
            if self.transform is not None:
                return self.transform(image)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy(array)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = None if self.task.get("image_only", False) else self.tokenizer(
            str(row.get("clinical_text", "No clinical context available.")), max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        if self.task["kind"] == "classification":
            target = torch.tensor(int(row.label), dtype=torch.long)
            mask = torch.tensor(True)
        else:
            values = np.asarray([row[x] for x in self.task["targets"]], dtype=np.float32)
            target = torch.from_numpy(np.nan_to_num(values, nan=0.0))
            mask = torch.from_numpy(np.isfinite(values))
        structured_columns = sorted(x for x in self.frame.columns if x.startswith("clinical_") and x != "clinical_text")
        structured = np.asarray([row[x] for x in structured_columns], dtype=np.float32) if structured_columns else np.empty(0, dtype=np.float32)
        structured = np.nan_to_num(structured, nan=0.0)
        return {
            "image": self._image(row.image_path),
            "input_ids": torch.empty(0, dtype=torch.long) if encoded is None else encoded["input_ids"].squeeze(0),
            "attention_mask": torch.empty(0, dtype=torch.long) if encoded is None else encoded["attention_mask"].squeeze(0),
            "target": target,
            "target_mask": mask,
            "clinical_structured": torch.from_numpy(structured),
            "patient_id": str(row.patient_id),
            "image_path": str(row.image_path),
        }
