# CCM-DiagProg

CCM-DiagProg is a unified framework for diagnosis, clinical metric regression, and prognosis from corneal confocal microscopy images. It combines visual representation learning, clinical text encoding, cross-modal fusion, causal channel screening, and task-conditioned prediction heads.

## Tasks

| Task | Objective | Output |
|---|---|---|
| Task 1 | Ocular neuroimmune diagnosis | HC / DED / NCP |
| Task 2 | Systemic small-fiber neuropathy diagnosis | HC / NODPN / DPN |
| Task 3 | Clinical reference metric regression | HbA1c / CFS / TBUT / SIT / OSDI |
| Task 4 | One-month treatment prognosis | ΔCFS / ΔTBUT / ΔSIT / ΔOSDI |
| Task 5 | Six-month postoperative prognosis | Non-persistent / Persistent |

## Installation

```bash
conda create -n ccm-diagprog python=3.8 -y
conda activate ccm-diagprog
pip install -e .
```

## Data preparation

Create one CSV manifest for each task. Every manifest must contain `image_path`, `patient_id`, and `split`, where `split` is `train`, `validation`, or `test`.

Classification manifests additionally contain:

```text
label,class_name
```

Regression manifests additionally contain their target columns and optional structured clinical fields prefixed with `clinical_`.

Place the manifests at the paths configured in `configs/task1.yaml` through `configs/task5.yaml`, or pass a path directly with `--manifest`.

## Training

Run one task at a time:

```bash
python train.py --config configs/task1.yaml
python train.py --config configs/task2.yaml
python train.py --config configs/task3.yaml
python train.py --config configs/task4.yaml
python train.py --config configs/task5.yaml
```

To use another manifest:

```bash
python train.py \
  --config configs/task1.yaml \
  --manifest Dataset/task1/manifest.csv
```

The best checkpoint and periodic checkpoints are saved under `checkpoints/<task>/`.

## Evaluation

```bash
python evaluate.py \
  --config configs/task1.yaml \
  --manifest Dataset/task1/manifest.csv \
  --checkpoint checkpoints/task1/best_model.pt
```

Use the corresponding configuration and checkpoint paths for Tasks 2–5.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Project structure

```text
configs/        Task configurations
scripts/        Shared training and evaluation utilities
src/data/       Dataset, manifest, audit, and split logic
src/models/     Causal-CCM architecture and channel screening
src/training/   Training, early stopping, and checkpoints
src/evaluation/ Metrics
train.py        Training entry point
evaluate.py     Evaluation entry point
```
