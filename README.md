# CCM-DiagProg

Official PyTorch implementation of the unified prior-guided and patient-adaptive framework for corneal confocal microscopy analysis.

The model jointly supports six clinical objectives through one shared visual encoder, Tiny ClinicalBERT semantic encoder, task-conditioned feature subsets, shared hypernetwork, and task-specific parameter generators. The default `nlpie/tiny-clinicalbert` output is projected from 312 to the framework's fixed 768-dimensional patient-semantic space.

| Identity | Objective | Output |
|---:|---|---|
| 0 | Ocular neuroimmune diagnosis | HC, DED, NCP |
| 1 | Systemic small-fiber neuropathy diagnosis | HC, NODPN, DPN |
| 2 | Ocular clinical metric regression | CFS, TBUT, SIT, OSDI |
| 3 | Glycemic metric regression | HbA1c |
| 4 | One-month treatment prognosis | ΔCFS, ΔTBUT, ΔSIT, ΔOSDI |
| 5 | Six-month postoperative prognosis | Non-persistent, Persistent |

## Method

The implementation contains:

- min–max-normalized visual relation modeling and prior-guided off-diagonal alignment;
- Pearson, Spearman, distance-correlation, and k-NN normalized mutual-information clinical relations;
- adaptive bootstrap stability weighting and clinical-to-visual prior projection;
- task-conditioned HSIC relevance and KCI conditional-robustness screening with GCV, sequential permutation testing, and Benjamini–Hochberg correction;
- leakage-controlled clinical serialization and stochastic clinical-context dropout;
- eight-head semantic-to-visual cross-attention;
- a shared `800 → 512 → 256` hypernetwork and six task-specific dynamic parameter generators;
- two-stage, equally weighted joint optimization across all six task identities.

## Installation

```bash
conda create -n ccm-diagprog python=3.8 -y
conda activate ccm-diagprog
pip install -e .
```

Tiny ClinicalBERT and ImageNet-pretrained ResNet-50 weights are downloaded through their standard Hugging Face and torchvision interfaces when they are not already cached.

## Data manifests

Prepare five CSV manifests in one directory:

```text
task1.csv
task2.csv
task3.csv
task4.csv
task5.csv
```

Every manifest contains:

```text
image_path,patient_id,split,clinical_text
```

`split` must be `train`, `validation`, or `test`. Classification manifests additionally contain `label` and `class_name`. Regression manifests contain the target columns declared in `configs/default.yaml`. Structured clinical fields use the `clinical_` prefix. Direct targets, future variables, and label-construction proxies must be listed under each task's `clinical_exclude` configuration.

Task 3 is divided into identities 2 and 3 while sharing the same manifest.

## Training

```bash
python train.py \
  --config configs/default.yaml \
  --manifests-dir /path/to/manifests \
  --output checkpoints/unified_six_task \
  --seed 3407
```

Training first warms up the visual auxiliary heads, constructs and fixes the clinical relation priors, performs prior alignment and task-conditioned channel screening, then freezes the refined visual encoder and trains semantic contextualization and the hypernetwork. All six objectives receive weight `1/6`. Periodic checkpoints are written every 10 epochs and the best Stage-II shared model is saved as `best_model.pt`.

Feature screening is independent for all six identities. `retained_channel_ratio` is configured by identity, while RBF bandwidths, KCI regularization, permutation counts, FDR decisions, rankings, and retained channel indices are estimated separately from each identity's training patients. The complete per-identity evidence is written to `relation_prior_audit.json` and `channel_screening_audit.json`.

To resume after a completed Stage-I checkpoint:

```bash
python train.py \
  --config configs/default.yaml \
  --manifests-dir /path/to/manifests \
  --output checkpoints/unified_six_task \
  --resume-stage1 checkpoints/unified_six_task/epoch_0030.pt
```

## Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --manifest /path/to/test.csv \
  --checkpoint checkpoints/unified_six_task/best_model.pt \
  --identity 0 \
  --text-missingness 0.0 \
  --output predictions.csv \
  --metrics metrics.csv
```

Clinical-context availability can be evaluated without changing the model:

```bash
python infer.py ... --text-missingness 0.0
python infer.py ... --text-missingness 0.5
python infer.py ... --text-missingness 1.0
```

## Inference complexity

For one 384×384 image and 128 Tiny ClinicalBERT tokens, the complete inference deployment contains 43.291 M parameters and requires 25.906 GFLOPs (`1 MAC = 2 FLOPs`).

## Tests

```bash
pytest -q
```

The repository intentionally excludes datasets, trained weights, predictions, and experiment-specific result files.

The paper-to-code execution map is provided in `METHOD_REPRODUCTION_AUDIT.md`.
