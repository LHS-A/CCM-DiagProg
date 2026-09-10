# CCM-DiagProg

Official implementation of the unified Causal-CCM framework for diagnosis, clinical metric regression, and prognosis from corneal confocal microscopy images.

## Tasks

| Identity | Task | Output |
|---:|---|---|
| 0 | Ocular-surface diagnosis | HC, DED, NCP |
| 1 | Systemic neuropathy diagnosis | HC, NODPN, DPN |
| 2 | Ocular-surface metric regression | CFS, TBUT, SIT, OSDI |
| 3 | Glycemic metric regression | HbA1c |
| 4 | One-month prognosis | CFS_1m, TBUT_1m, SIT_1m, OSDI_1m |
| 5 | Six-month prognosis | Non-persistent, Persistent |

All identities share one ResNet-50 visual encoder, Tiny ClinicalBERT encoder, task embeddings, semantic alignment module, and hypernetwork. Each identity has its own dynamic parameter generator. The default `nlpie/tiny-clinicalbert` representation is projected from 312 to the framework's fixed 768-dimensional patient-semantic space. Training implements multi-view clinical relation priors with 1,000 bootstrap resamples, prior-guided visual alignment, sequential-permutation HSIC/KCI channel screening (up to 10,000 permutations) with GCV, clinical context dropout, and six-task-balanced Stage-II optimization.

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
All rows belonging to one patient must occur in exactly one partition. The
trainer rejects any patient overlap. Non-target structured clinical variables
used by the relation prior use the `clinical_` prefix. Target, future and proxy
fields are excluded by the identity-specific rules in `configs/default.yaml`.

## Training

Train the six identities sequentially in one shared model:

```bash
python train.py \
  --config configs/default.yaml \
  --manifests-dir /path/to/manifests \
  --output checkpoints/unified_six_task
```

The trainer saves `best_model.pt`, periodic checkpoints every 10 epochs, and
JSON audits for relation-prior construction and channel screening. AdamW uses
the paper's cosine schedule from `1e-4` to `1e-6`.

Feature screening is independent for all six identities. `retained_channel_ratio` is configured by identity, while RBF bandwidths, KCI regularization, permutation counts, FDR decisions, rankings, and retained channel indices are estimated separately from each identity's training patients. KCI is run and BH-corrected only inside the corresponding HSIC candidate family. The complete evidence is written to `relation_prior_audit.json`, `channel_screening_audit.json`, and `screening/<identity>/`.

Clinical relation priors are also task- and fold-specific. For every identity,
the trainer independently constructs `C_clin,t`, multi-view `A_rel,t`, `Pi_t`
and `M_prior,t` from that fold's training patients. Exact state and provenance
are stored under `relation_priors/<identity>/` as clinical metadata and hashes,
normalization statistics, relation-view weights, `A_rel.json`, `Pi.pt`, and
`M_prior.pt`. No cross-task or cross-fold relation cache is used.

Run all five patient-level folds (test fold, following validation fold, and
three training folds) with:

```bash
python scripts/train_unified_five_folds.py \
  --source-manifests /path/to/base_manifests \
  --work-dir artifacts/unified_five_folds \
  --output checkpoints/unified_five_folds
```

Every fold launches a new shared model and independently reconstructs its
training-only relation priors, kernels, permutation tests and six channel sets.

Resume a completed Stage-1 run with `--resume-stage1 <checkpoint>`. The
checkpoint stores all six priors and all six independently selected channel
sets; legacy shared-mask checkpoints are intentionally rejected.

## Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --manifest /path/to/test.csv \
  --checkpoint checkpoints/unified_six_task/best_model.pt \
  --identity 0 \
  --output predictions.csv \
  --metrics metrics.csv
```

Text-missing evaluation is controlled with `--text-missingness`:

```bash
python infer.py ... --text-missingness 0.0
python infer.py ... --text-missingness 0.5
python infer.py ... --text-missingness 1.0
```

Identity indices are zero-based and follow the task table above.

## Inference complexity

For one 384×384 image and 128 Tiny ClinicalBERT tokens, the complete inference deployment contains 43.291 M parameters and requires 25.906 GFLOPs (`1 MAC = 2 FLOPs`).

## Test

```bash
python -m compileall train.py infer.py src scripts
pytest -q
python scripts/smoke_unified_real_data.py --device cuda
```

The smoke test reads real Task-3 training images and executes reduced-cost
Stage 1, prior construction, HSIC, candidate-only KCI, channel fixation, Stage
2 backward, checkpoint reload and inference. It never modifies dataset files.

The formula-level execution map and expected artifact structure are documented
in `PAPER_CODE_AUDIT.md` and `METHOD_REPRODUCTION_AUDIT.md`.

Datasets, patient records, trained checkpoints, predictions, caches and private
experiment outputs are intentionally excluded from the public repository.
