# CCM-DiagProg

`UnifiedCausalCCM` is the public end-to-end model API used by every official
training and inference entry point for the shared six-task framework.

Official implementation of the unified Causal-CCM framework for diagnosis, clinical metric regression, and prognosis from corneal confocal microscopy images.

## Tasks

| Identity | Task | Output |
|---:|---|---|
| 0 | Ocular-surface diagnosis | HC, DED, NCP |
| 1 | Systemic neuropathy diagnosis | HC, DPN−, DPN+ |
| 2 | Ocular-surface metric regression | TBUT, CFS, SIT, OSDI |
| 3 | Glycemic metric regression | HbA1c |
| 4 | One-month prognosis | TBUT_1m, CFS_1m, SIT_1m, OSDI_1m |
| 5 | Six-month prognosis | NPD, PDE |

All identities share one ResNet-50 visual encoder, Tiny ClinicalBERT encoder, task embeddings, semantic alignment module, and hypernetwork. Each identity has its own dynamic parameter generator. The default `nlpie/tiny-clinicalbert` representation is projected from 312 to the framework's fixed 768-dimensional patient-semantic space. Training implements multi-view clinical relation priors with 1,000 bootstrap resamples, prior-guided visual alignment, permutation-tested HSIC/KCI feature filtering with GCV, clinical context dropout, and six-setting-balanced Stage-II optimization.

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
trainer also rejects a patient assigned to different partitions in different
tasks. Non-target structured clinical variables
use the `clinical_` prefix and may be declared as `continuous`, `ordinal`,
`binary`, or `nominal`. Target, future and proxy fields are excluded by the
identity-specific allow/exclude rules in `configs/default.yaml`. Safe fields
are rendered through the fixed paper template; missing phrases are omitted.

## Training

Train all six identities with task-balanced updates in one shared model:

```bash
python train.py \
  --config configs/default.yaml \
  --manifests-dir /path/to/manifests \
  --output checkpoints/unified_six_task
```

The trainer saves `best_model.pt`, periodic checkpoints every 10 epochs, and
JSON audits for relation-prior construction and channel filtering. AdamW uses
the paper's cosine schedule from `1e-4` to `1e-6`.

Feature filtering is independent for all six prediction settings. The fixed
paper value is `rho=0.30`. RBF bandwidths, KCI regularization, permutation
tests, FDR decisions, rankings, and retained channel indices are estimated
separately from each setting's training patients. GAP/GMP descriptors are
first averaged across each patient's images. HSIC-significant channels form
the coarse set `S_HSIC`; KCI is evaluated only on that set, and
`S_sig = S_HSIC ∩ S_KCI`. If `S_sig` is smaller than
`K=max(1,floor(rho*C))`, every significant channel is retained and the
remaining positions are filled from all still-unselected channels in
descending HSIC-score order. The complete evidence is written to
`relation_prior_audit.json`, `channel_filtering_audit.json`, and
`feature_filtering/<identity>/`.

Clinical relation priors are also task- and fold-specific. For every identity,
the trainer independently constructs `C_clin,t`, multi-view `A_rel,t`, `Pi_t`
and `M_prior,t` from that fold's training patients. Exact state and provenance
are stored under `relation_priors/<identity>/` as clinical metadata and hashes,
normalization statistics, relation-view weights, `A_rel.json`, `Pi.pt`, and
`M_prior.pt`. No cross-task or cross-fold relation cache is used.
Each patient contributes once: channel GAP responses and clinical rows are
aggregated before association estimation. Bootstrap stability uses
`tau_v=0.05`; channel-to-clinical soft correspondence uses `tau_r=0.5`.

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
One global patient-to-fold map is shared across tasks. Patient-balanced
sampling chooses patients uniformly and then one image per selected patient;
the six task mini-batches contain 3/3/3/3/2/2 images, for a total batch size of
16. The loss is averaged image-within-patient, patient-within-task, then
equally across tasks. Regression targets are standardized with training-fold
patient statistics stored in the checkpoint.

The warm-up, prior-alignment and semantic-hypernetwork phases restore their
best validation checkpoint before continuing. The final checkpoint stores all
six priors and all six independently selected channel sets.

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
Predictions and metrics are patient-level: image probabilities are averaged
for classification/prognosis and continuous image predictions are averaged
for regression. Regression outputs are inverse-transformed using the saved
training-fold normalizer. Internal point estimates pool held-out-fold patient
predictions; external point estimates average patient predictions from the
five fold-specific models before metrics are calculated.

Evaluate all held-out folds and the fixed external cohorts with:

```bash
python scripts/evaluate_unified_five_folds.py \
  --fold-manifests artifacts/unified_five_folds \
  --checkpoints checkpoints/unified_five_folds \
  --external-manifests /path/to/external_manifests \
  --output results/unified_five_folds
```

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
