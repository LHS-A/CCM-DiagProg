# Reproduction audit

## 1. Methodology specification

The sole methodological authority is the 14-page `Paper.pdf` created on
2026-09-23, SHA-256
`2a158f4e3b5c6df8037092086612799c30ebed743c16174f8b66a7e98d8a03dd`.
The executable contract is one shared six-identity model with two-stage
training, task/fold-specific relation priors and channel subsets, task-safe
patient semantics, dynamic prediction, training-only statistics, and
patient-level evaluation.

## 2. Paper-to-code mapping

| Paper component | Eq./section | Python implementation | Before | Fix | After |
|:--|:--|:--|:--|:--|:--:|
| Visual relation | Eq. (1) | `StructuralPrior.loss` | implemented | verified min-max maps, cosine relation and epsilon | PASS |
| Clinical relation | Eq. (2) | `association_views`, `bootstrap_stability` | Pearson normalization and stability temperature wrong | exact Pearson/Spearman, dCor, NMI, applicable views, `tau_v=0.05` | PASS |
| Clinical-to-visual projection | Sec. 2.1.2 | `collect_statistics`, `build_relation_prior` | image-level observations; no projection temperature | patient means, multi-view association, row softmax with `tau_r=0.5` | PASS |
| Off-diagonal normalization | Eq. (3) | `StructuralPrior.loss` | implemented | deterministic reference test | PASS |
| Prior loss | Eq. (4) | `StructuralPrior.loss` | extra `C(C-1)` divisor | exact squared Frobenius norm; detached prior | PASS |
| Patient channel descriptor | Sec. 2.1.3 | `collect_statistics`, `patient_mean` | each image was an observation | mean `[GAP,GMP]` over each patient's images | PASS |
| HSIC coarse filter | Sec. 2.1.3 | `filter_channels` | image-level input | patient input, paper kernels, permutation and BH-FDR | PASS |
| KCI fine filter | Sec. 2.1.3 | `filter_channels`, `gcv_regularization` | image-level input | only `S_HSIC`, admissible `Z_t`, GCV, permutation and BH-FDR | PASS |
| Top-rho | Sec. 2.1.3 | `top_rho_retention` | aborted when `S_HSIC` was short | exact `K`; KCI rank or all-unselected HSIC fallback | PASS |
| Safe patient context | Sec. 2.2.1 | dataset, training helpers, config | present | explicit available/target/future/proxy rules and phrase omission | PASS |
| Context dropout | Sec. 2.2.1 | `CCMManifestDataset.__getitem__` | present | discrete uniform removal; patient-consistent ablation context | PASS |
| Tiny ClinicalBERT | Sec. 2.2.1/3.2 | config, `UnifiedCausalCCM` | present | confirmed Tiny model; mixed batches encode only available rows | PASS |
| Semantic contextualization | Eq. (5) | `SemanticContext` | implemented | verified visual Q, clinical K/V, residual, LayerNorm, pooling | PASS |
| Dynamic prediction | Eq. (6) | `UnifiedCausalCCM.forward` | implemented | tested patient/task conditioning and generated shapes | PASS |
| Stage 1 loss | Eq. (7) | `balanced_epoch`, `optimize_phase` | wrong Eq. (4) scale | warm-up, fixed prior, refinement and channel fixing | PASS |
| Hierarchical task loss | Eq. (8) | `sample_prediction_loss`, `patient_average` | flat image loss; 96 images/update | image→patient→task means; total batch 16 | PASS |
| Hyper regularization | Eq. (9) | model and `balanced_epoch` | repeated per image | one norm per represented patient-task pair | PASS |
| Regression scaling | Sec. 2.4 | patient utilities, train, infer | raw-unit optimization | training-patient mean/std and inverse transform | PASS |
| Five-fold inference | Sec. 3.2 | fold scripts and infer | per-task fold maps; metric averaging | shared patient folds, pooled OOF, external prediction ensemble | PASS |

## 3. Critical discrepancies found

- **Critical:** relation prior and HSIC/KCI treated images rather than patients
  as independent observations.
- **Critical:** `tau_v=0.05` and `tau_r=0.5` were absent.
- **Critical:** Eq. (4) contained an extra channel-count divisor.
- **Critical:** Top-rho aborted instead of using the paper fallback.
- **Critical:** regression lacked training-fold standardization and inverse
  transformation.
- **Critical:** final metrics were image-level, and external point estimates
  averaged metrics rather than five-model predictions.
- **Critical:** patient partitions were generated independently by task.
- **Major:** loss and dynamic regularization weighted patients by image count.
- **Major:** six batches of 16 created 96 images per update.
- **Major:** zero-context rows in mixed batches entered BERT.
- **Major:** Pearson normalization used inconsistent denominators.

All discrepancies were corrected in executable paths and covered by tests.

## 4. Paper-unspecified implementation decisions

| Detail | Implementation | Basis |
|:--|:--|:--|
| Tiny identifier | `nlpie/tiny-clinicalbert` (4 layers, hidden 312, 12 heads), projected to 768 | compatible pre-audit code |
| Numerical epsilon | `1e-6` | compatible pre-audit code |
| Bootstrap | 1,000 patient resamples | compatible pre-audit code |
| HSIC/KCI calibration | 1,000 permutations, +1 correction, BH-FDR 0.05 | compatible pre-audit code |
| KCI GCV | 50 log-spaced values in `[1e-6,1e1]` | compatible pre-audit code |
| Stage 1 caps | warm-up 100; alignment up to 100; patience 20 | compatible pre-audit code |
| Preprocessing | 384×384 bilinear resize and ImageNet normalization | compatible pre-audit code |
| Cross-task folds | one seeded global patient map; fold `k` test, `k+1` validation | minimal leakage-free rule |

## 5. Data leakage audit

| Check | Result |
|:--|:--:|
| Patient-disjoint partitions within every task | PASS |
| Same patient retains one partition across tasks | PASS |
| Normalization, `C_clin`, `A_rel`, `Pi`, `M_prior` use train only | PASS |
| HSIC/KCI/GCV/FDR/channel selection use train only | PASS |
| External data excluded from development | PASS |
| Target/future/proxy fields excluded from semantic context and KCI `Z_t` | PASS |
| One-month outcomes excluded from short-term inputs | PASS |
| Postoperative outcomes excluded from long-term baseline inputs | PASS |
| Wrong task/fold state has no silent fallback | PASS |

## 6. Equation verification

| Equation | Tensor computation | Reduction/gradient | Result |
|:--|:--|:--|:--:|
| Eq. (1) | `B×C×H×W → C×C` | image mean; cosine relation | PASS |
| Eq. (2) | `N_patient×M → M×M×4 → M×M` | applicable-view stability weighting | PASS |
| Eq. (3) | diagonal removal and Frobenius normalization | epsilon-stabilized | PASS |
| Eq. (4) | squared Frobenius difference | gradient only to visual path | PASS |
| Eq. (5) | visual Q, clinical K/V | residual, LayerNorm, spatial mean | PASS |
| Eq. (6) | `[e_t||h_cls] → g → W,b → prediction` | dynamic per patient-task | PASS |
| Eq. (7) | `L_aux + lambda_prior L_prior` | `lambda_prior=1` | PASS |
| Eq. (8) | image → patient → task means | six tasks equally weighted | PASS |
| Eq. (9) | task loss + mean patient-task dynamic norm | `lambda_hyper=5e-4` | PASS |

## 7. Training-stage verification

Warm-up, post-warm-up prior estimation, fixed-prior refinement,
HSIC→KCI→Top-rho, frozen Stage 2 visual encoder, patient-balanced sampling,
total 16-image joint batch, six-identity forward/backward and dynamic
prediction all passed.

## 8. Inference verification

Full, partial and zero context passed. Zero context bypasses BERT and uses
task-only conditioning. Inference loads fixed priors/subsets without
re-estimation, aggregates image probabilities/continuous outputs by patient,
inverse-transforms regression, pools internal OOF predictions and ensembles
external patient predictions across the five fold-specific models.

## 9. Tests executed

| Command | Repository | Exit | Result |
|:--|:--|--:|:--|
| `python -m compileall -q train.py infer.py src scripts tests` | Final | 0 | PASS |
| `pytest -q` | Final | 0 | 38 passed |
| `python -m compileall -q train.py infer.py src scripts tests` | GitHub | 0 | PASS |
| `pytest -q` | GitHub | 0 | 31 passed |
| `python scripts/smoke_unified_real_data.py --manifest artifacts/one_fold_splits/task3.csv --device cpu` | Final | 0 | `REAL_DATA_PIPELINE_SMOKE_PASS 8 20` |
| `nvidia-smi` | environment | nonzero | NOT RUNTIME-VERIFIED: driver unavailable to this process |

The real-data smoke executed image loading, Stage 1 backward, patient-level
`C_clin → A_rel → Pi → M_prior`, prior-alignment backward, patient-level
HSIC/KCI/Top-rho, Stage 2 backward, checkpoint reload and inference without
modifying data.

## 10. Files changed

- `train.py`: leakage guards, patient sampler/loss, regression normalization,
  total batch allocation and exact stages.
- `infer.py`: inverse scaling and patient-level metrics.
- `src/data/patient.py`: aggregation, normalizers and sampler.
- `src/data/dataset.py`: patient-consistent ablation context.
- `src/models/causal_ccm.py`: exact Eq. (4).
- `src/models/relations.py`: exact correlation, `tau_v`, `tau_r` and audit.
- `src/models/filtering.py`: exact HSIC/KCI/Top-rho fallback.
- `src/models/unified_causal_ccm.py`: BERT bypass and pair regularization.
- fold scripts: shared patient folds, OOF pooling and external ensemble.
- config, README and tests: current method and verification contract.

## 11. Dual-directory SHA-256 parity

`CORE_METHODOLOGY_FILES.txt` lists 21 core files. Final and GitHub hashes
matched for every entry after independent tests.

| File | Final/GitHub SHA-256 | Result |
|:--|:--|:--:|
| `train.py` | `6abe8a75d8f9abf7b1d818b9a37061f48a1aa3f70b4a24c6969e111b68e0ca04` | PASS |
| `infer.py` | `383e291a1b63a82895ab1ed5b70bbda76eba5ca1ba0b03a233eed8ca351c3c69` | PASS |
| `src/data/patient.py` | `cf4c66511108a4b0bc706af649e5e821ed85698fde1ffbb37083fe7997e724b0` | PASS |
| `src/models/relations.py` | `aef11a61a7c52d534aefd84f552781d9927af972fc07cfe7a6ed6642e49d3b24` | PASS |
| `src/models/filtering.py` | `618d7b46e8f88e6ed71866770bae8f8de4851ec7c26c43928683485306c1cdbd` | PASS |
| `src/models/unified_causal_ccm.py` | `a90024e3c294db7dff6df3dd4fe7d5befc37ca6529eb843865224041e44dac59` | PASS |

## 12. Remaining limitations

- GPU runtime verification was unavailable because this process cannot
  communicate with the NVIDIA driver. CPU equation/module tests and a
  real-image two-stage smoke run passed.
- A full 500-epoch five-fold retraining was outside this code audit. The full
  path is implemented; the reduced real-data execution was verified.
