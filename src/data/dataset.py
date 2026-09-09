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
        clinical_missingness: float | None = None,
        excluded_clinical_fields: Optional[set[str]] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.task = task
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.transform = transform
        self.max_length = max_length
        self.clinical_missingness = clinical_missingness
        self.excluded_clinical_fields = {x.casefold() for x in (excluded_clinical_fields or set())}
        if clinical_missingness is not None and not 0.0 <= clinical_missingness <= 1.0:
            raise ValueError("clinical_missingness must lie in [0,1]")
        def permitted(column: str) -> bool:
            key=column.casefold();short=key[len("clinical_"):] if key.startswith("clinical_") else key
            return key not in self.excluded_clinical_fields and short not in self.excluded_clinical_fields
        self.clinical_structured_columns=tuple(sorted(
            x for x in self.frame.columns if x.startswith("clinical_") and x!="clinical_text" and permitted(x)
        ))

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
        text=str(row.get("clinical_text", "")).strip(); fields=[x.strip() for x in text.split(";") if x.strip()]
        def allowed(name: str) -> bool:
            key=name.strip().casefold(); short=key[len("clinical_"):] if key.startswith("clinical_") else key
            return key not in self.excluded_clinical_fields and short not in self.excluded_clinical_fields
        fields=[x for x in fields if allowed(x.split(":",1)[0])]
        unavailable=(not fields) or text.casefold().startswith("clinical context unavailable")
        q=float(torch.rand(())) if self.clinical_missingness is None else float(self.clinical_missingness)
        n=int(np.floor(q*len(fields)+0.5));order=torch.randperm(len(fields)).tolist();removed=set(order[:n]);fields=[x for i,x in enumerate(fields) if i not in removed]
        available=(not unavailable) and bool(fields); sentence="; ".join(fields) if available else "No clinical context available."
        encoded = self.tokenizer(
            sentence, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        if self.task["kind"] == "classification":
            target = torch.tensor(int(row.label), dtype=torch.long)
            mask = torch.tensor(True)
        else:
            values = np.asarray([row[x] for x in self.task["targets"]], dtype=np.float32)
            target = torch.from_numpy(np.nan_to_num(values, nan=0.0))
            mask = torch.from_numpy(np.isfinite(values))
        structured = np.asarray([row[x] for x in self.clinical_structured_columns], dtype=np.float32) if self.clinical_structured_columns else np.empty(0, dtype=np.float32)
        return {
            "image": self._image(row.image_path),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "clinical_available": torch.tensor(available,dtype=torch.bool),
            "target": target,
            "target_mask": mask,
            "clinical_structured": torch.from_numpy(structured),
            "patient_id": str(row.patient_id),
            "image_path": str(row.image_path),
        }
