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
        allowed_clinical_fields: Optional[set[str]] = None,
        partition: Optional[str] = None,
        relation_clinical_fields: Optional[list[str]] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.task = task
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.transform = transform
        self.max_length = max_length
        self.clinical_missingness = clinical_missingness
        self.excluded_clinical_fields = {x.casefold() for x in (excluded_clinical_fields or set())}
        self.allowed_clinical_fields = {x.casefold() for x in (allowed_clinical_fields or set())}
        inferred=set(self.frame["split"].dropna().astype(str)) if "split" in self.frame else set()
        self.partition=partition or (next(iter(inferred)) if len(inferred)==1 else "unspecified")
        if partition is not None and inferred and inferred != {partition}:
            raise ValueError(f"Dataset provenance mismatch: declared {partition!r}, rows contain {sorted(inferred)}")
        if clinical_missingness is not None and not 0.0 <= clinical_missingness <= 1.0:
            raise ValueError("clinical_missingness must lie in [0,1]")
        def permitted(column: str) -> bool:
            key=column.casefold();short=key[len("clinical_"):] if key.startswith("clinical_") else key
            included=not self.allowed_clinical_fields or key in self.allowed_clinical_fields or short in self.allowed_clinical_fields
            return included and key not in self.excluded_clinical_fields and short not in self.excluded_clinical_fields
        nuisance_columns=tuple(sorted(
            x for x in self.frame.columns if x.startswith("clinical_") and x!="clinical_text" and permitted(x)
        ))
        configured_types={str(k).casefold():str(v).casefold() for k,v in task.get("clinical_field_types",{}).items()}
        def encode(columns,expand_nominal):
            arrays=[];names=[];types=[]
            for column in columns:
                if column not in self.frame:continue
                source=self.frame[column];numeric=pd.to_numeric(source,errors="coerce")
                present=source.notna() & source.astype(str).str.strip().ne("")
                inferred="continuous" if int(numeric.notna().sum())==int(present.sum()) else "nominal"
                if inferred=="continuous" and numeric[present].nunique()<=2:inferred="binary"
                key=column.casefold();short=key[len("clinical_"):] if key.startswith("clinical_") else key
                kind=configured_types.get(key,configured_types.get(short,inferred))
                if kind not in {"continuous","ordinal","binary","nominal"}:raise ValueError(f"unsupported clinical type {kind!r} for {column}")
                if kind!="nominal":
                    arrays.append(numeric.to_numpy(np.float32)[:,None]);names.append(column);types.append(kind)
                else:
                    categories=sorted(source[present].astype(str).unique())
                    if expand_nominal:
                        for category in categories:
                            value=np.where(~present,np.nan,(source.astype(str)==category).astype(float)).astype(np.float32)
                            arrays.append(value[:,None]);names.append(f"{column}={category}");types.append("binary")
                    else:
                        mapping={category:i for i,category in enumerate(categories)}
                        value=np.array([mapping.get(str(x),np.nan) if ok else np.nan for x,ok in zip(source,present)],np.float32)
                        arrays.append(value[:,None]);names.append(column);types.append("nominal")
            matrix=np.concatenate(arrays,axis=1) if arrays else np.empty((len(self.frame),0),dtype=np.float32)
            return tuple(names),tuple(types),matrix
        self.clinical_structured_columns,self.clinical_structured_types,self._nuisance_matrix=encode(nuisance_columns,True)
        relation_fields=list(dict.fromkeys(relation_clinical_fields or nuisance_columns))
        self.clinical_relation_columns,self.clinical_relation_types,self._relation_matrix=encode(relation_fields,False)

    @staticmethod
    def _present(value: Any) -> bool:
        return pd.notna(value) and str(value).strip() not in {"", "nan", "None"}

    @staticmethod
    def _default_slot(field: str) -> str:
        key=field.casefold();key=key[len("clinical_"):] if key.startswith("clinical_") else key
        if key in {"age", "patient_age"}: return "age"
        if key in {"sex", "gender"}: return "sex"
        if "symptom" in key or key in {"osdi", "pain", "pain_score"}: return "symptom_text"
        if any(x in key for x in ("ocular", "eye", "cornea", "tbut", "but", "cfs", "sit")): return "ocular_status"
        if any(x in key for x in ("history", "duration", "treatment", "medication")): return "medical_history"
        return "systemic_status"

    def _clinical_fields(self, row: pd.Series) -> list[tuple[str,str,str]]:
        """Return task-safe (template slot, field name, value) triples."""
        configured=self.task.get("clinical_template_fields",{})
        fields=[];seen=set()
        for slot,columns in configured.items():
            for column in columns:
                key=column.casefold();short=key[len("clinical_"):] if key.startswith("clinical_") else key
                included=not self.allowed_clinical_fields or key in self.allowed_clinical_fields or short in self.allowed_clinical_fields
                if column in row and included and self._present(row[column]) and key not in self.excluded_clinical_fields and short not in self.excluded_clinical_fields:
                    fields.append((slot,column,str(row[column]).strip()));seen.update((key,short))
        for column,value in row.items():
            key=str(column).casefold();short=key[len("clinical_"):] if key.startswith("clinical_") else key
            if not key.startswith("clinical_") or key=="clinical_text" or key in seen:continue
            included=not self.allowed_clinical_fields or key in self.allowed_clinical_fields or short in self.allowed_clinical_fields
            if not included or key in self.excluded_clinical_fields or short in self.excluded_clinical_fields or not self._present(value):continue
            fields.append((self._default_slot(str(column)),str(column),str(value).strip()));seen.update((key,short))
        # A standardized key:value clinical_text remains a supported transport
        # format, but is always rendered through the fixed paper template.
        text=str(row.get("clinical_text", ""))
        for item in (x.strip() for x in text.split(";") if x.strip()):
            if ":" not in item: continue
            name,value=(x.strip() for x in item.split(":",1));key=name.casefold()
            short=key[len("clinical_"):] if key.startswith("clinical_") else key
            included=not self.allowed_clinical_fields or key in self.allowed_clinical_fields or short in self.allowed_clinical_fields
            if key in seen or not included or key in self.excluded_clinical_fields or short in self.excluded_clinical_fields or not value: continue
            fields.append((self._default_slot(name),name,value));seen.add(key)
        return fields

    @staticmethod
    def _serialize_clinical(fields: list[tuple[str,str,str]]) -> str:
        grouped={slot:[] for slot in ("age","sex","systemic_status","medical_history","ocular_status","symptom_text")}
        for slot,name,value in fields:
            if slot not in grouped: raise ValueError(f"unsupported clinical template slot {slot!r}")
            rendered=value if slot in {"age","sex"} else f"{name} {value}"
            grouped[slot].append(rendered)
        age=grouped["age"][0] if grouped["age"] else None;sex=grouped["sex"][0] if grouped["sex"] else None
        if age and sex: opening=f"A {age}-year-old {sex} patient"
        elif age: opening=f"A {age}-year-old patient"
        elif sex: opening=f"A {sex} patient"
        else: opening="Patient"
        phrases=[]
        for slot,label in (("systemic_status","systemic status"),("medical_history","medical history"),("ocular_status","ocular status"),("symptom_text","presenting symptoms")):
            if grouped[slot]:phrases.append(f"{label} {', '.join(grouped[slot])}")
        if len(phrases)>1:body=", ".join(phrases[:-1])+", and "+phrases[-1]
        else:body=phrases[0] if phrases else ""
        return opening+(" with "+body if body else "")+"."

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
        fields=self._clinical_fields(row);unavailable=not fields
        if self.clinical_missingness is None:
            n=int(torch.randint(0,len(fields)+1,()).item())
        else:
            n=int(np.floor(float(self.clinical_missingness)*len(fields)+0.5))
        order=torch.randperm(len(fields)).tolist();removed=set(order[:n]);fields=[x for i,x in enumerate(fields) if i not in removed]
        available=(not unavailable) and bool(fields);sentence=self._serialize_clinical(fields) if available else ""
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
        structured=self._nuisance_matrix[index]
        relation=self._relation_matrix[index]
        return {
            "image": self._image(row.image_path),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "clinical_available": torch.tensor(available,dtype=torch.bool),
            "target": target,
            "target_mask": mask,
            "clinical_structured": torch.from_numpy(structured),
            "clinical_relation": torch.from_numpy(relation),
            "patient_id": str(row.patient_id),
            "image_path": str(row.image_path),
            "data_partition": self.partition,
        }
