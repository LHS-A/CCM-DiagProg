# CCM-DiagProg

Official implementation of the unified Causal-CCM framework for diagnosis, clinical metric regression, and prognosis from corneal confocal microscopy images.

## Tasks

| Identity | Task | Output |
|---:|---|---|
| 1 | Ocular-surface diagnosis | HC, DED, NCP |
| 2 | Systemic neuropathy diagnosis | HC, NODPN, DPN |
| 3 | Ocular-surface metric regression | CFS, TBUT, SIT, OSDI |
| 4 | Glycemic metric regression | HbA1c |
| 5 | One-month prognosis | ΔCFS, ΔTBUT, ΔSIT, ΔOSDI |
| 6 | Six-month prognosis | Non-persistent, Persistent |

All identities share one ResNet-50 visual encoder, ClinicalBERT encoder, task embeddings, semantic alignment module, and hypernetwork. Each identity has its own dynamic parameter generator.

## Installation

```bash
conda create -n ccm-diagprog python=3.8 -y
conda activate ccm-diagprog
pip install -e .
```

## Manifest format

Prepare `task1.csv` through `task5.csv` in one manifest directory. Each CSV contains:

```text
image_path,patient_id,split,clinical_text
```

`split` is `train`, `validation`, or `test`. Classification manifests also contain `label` and `class_name`. Regression manifests contain the target columns defined in `configs/default.yaml`.

Task 3 supplies two shared-model identities: ocular-surface regression and HbA1c regression.

## Training

Train the six identities sequentially in one shared model:

```bash
python train.py \
  --config configs/default.yaml \
  --manifests-dir /path/to/manifests \
  --output checkpoints/unified_six_task
```

The trainer saves the best checkpoint for each identity, a final shared checkpoint, and periodic checkpoints every 10 epochs.

## Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --manifest /path/to/test.csv \
  --checkpoint checkpoints/unified_six_task/best_task_0.pt \
  --identity 0 \
  --output predictions.csv
```

Text-missing evaluation is controlled with `--text-missingness`:

```bash
python infer.py ... --text-missingness 0.0
python infer.py ... --text-missingness 0.5
python infer.py ... --text-missingness 1.0
```

Identity indices are zero-based and follow the task table above.

## Test

```bash
pytest
```

