# Paper-to-code reproduction audit

The authoritative execution path is `train.py` → `UnifiedCausalCCM` and
`infer.py` → `UnifiedCausalCCM`. Statistics used by relation construction and
screening are guarded as training-partition-only and are recomputed inside
every cross-validation fold.

## Numbered equations

| Paper equation | Mathematical operation | Implementation | Called in | Test |
|:--|:--|:--|:--|:--:|
| Eq. (1) | Per-channel min–max normalization and normalized spatial co-activation, producing `M_vis` | `src/models/causal_ccm.py::StructuralPrior.loss` | Stage 1 prior-alignment batches | PASS |
| Eq. (2) | Clinical adjacency, channel–clinical assignment `Pi`, and `M_prior = Pi A Pi^T` | `src/models/relations.py::association_views`, `adaptive_bootstrap`, `build_relation_prior` | `train.py::construct_priors` | PASS |
| Eq. (3) | Remove diagonal, Frobenius-normalize both matrices, stop-gradient prior, mean squared alignment | `src/models/causal_ccm.py::StructuralPrior.loss` | Stage 1 after warm-up | PASS |
| Eq. (4) | Centered-kernel HSIC statistic and permutation significance | `src/models/screening.py::screen_channels` | Per identity in `train.py::screen_all` | PASS |
| Eq. (5) | Conditional joint kernel, GCV residualization, KCI statistic and candidate-family permutations | `src/models/screening.py::gcv_regularization`, `screen_channels` | Strictly after HSIC candidates, per identity | PASS |
| Eq. (6) | `G_t(H([e_t || h_cls])) -> [vec(W); b]` | `src/models/unified_causal_ccm.py::UnifiedCausalCCM.forward` | Every Stage-2 train/inference call | PASS |
| Eq. (7) | Equal task-balanced auxiliary loss plus structural alignment | `train.py::balanced_epoch` | Stage 1 | PASS |
| Eq. (8) | Equal task-balanced downstream loss plus dynamic-parameter regularization | `train.py::balanced_epoch` | Stage 2 | PASS |

## Component and execution-path audit

| Paper component | File / function | Training | Inference | Automated evidence | Status |
|:--|:--|:--:|:--:|:--|:--:|
| Shared ResNet-50 visual encoder | `unified_causal_ccm.py::visual_encoder` | Yes | Yes | six-task forward | PASS |
| Visual dependency and prior alignment | `causal_ccm.py::StructuralPrior.loss` | Stage 1 | fixed encoder | formula and gradient tests | PASS |
| Pearson/Spearman/dCor/NMI clinical relations | `relations.py::association_views` | train only | fixed prior | relation tests | PASS |
| Adaptive bootstrap stability weighting | `relations.py::adaptive_bootstrap` | train only | no | deterministic audit metadata | PASS |
| Clinical-to-visual projection | `relations.py::build_relation_prior` | train only | fixed | shape/metadata checks | PASS |
| Task-specific HSIC | `screening.py::screen_channels` | six independent calls | fixed indices | local result objects | PASS |
| HSIC candidate → KCI | `screening.py::screen_channels` | candidate family only | fixed indices | finite KCI p-values only for HSIC candidates | PASS |
| Permutation + BH/FDR | `screening.py::_sequential_many`, `_bh` | per identity | no | audit vectors | PASS |
| Task-specific `rho_t` and `S_t` | `train.py::task_setting`, `screen_all` | per identity/fold | restored | six buffers and per-task files | PASS |
| Actual task-specific projection | `SemanticContext.set_input_dim/forward` | Yes | Yes | selected width assertions | PASS |
| Target/future/proxy masking | `dataset.py`, per-identity config | Yes | Yes | safe-field tests | PASS |
| Context dropout 0–100% | `dataset.py::__getitem__` | resampled | configured/resampled | 0/partial/full tests | PASS |
| ClinicalBERT tokens and `[CLS]` | `UnifiedCausalCCM.forward` | Stage 2 | Yes | forward tests | PASS |
| Cross-attention and residual visual path | `SemanticContext.forward` | Stage 2 | Yes | zero-context finite prediction | PASS |
| Patient–task conditioning | `UnifiedCausalCCM.forward` | Stage 2 | Yes | patient and task perturbation tests | PASS |
| Shared hypernetwork and task generators | `hyper`, `generators` | Stage 2 | Yes | generated-code/output tests | PASS |
| Dynamic `W` and `b` | `UnifiedCausalCCM.forward` | Stage 2 | Yes | checkpoint/forward tests | PASS |
| Two-stage freezing | `set_stage_trainability`, `balanced_epoch` | Yes | n/a | parameter-state tests | PASS |
| AdamW and cosine annealing | `train.py::learning_rate` | every epoch | n/a | endpoint test | PASS |
| 5-fold patient-level 6:2:2 orchestration | `scripts/train_unified_five_folds.py` | five isolated runs | fold checkpoints | patient-disjoint test | PASS |
| Complete checkpoint inference | `infer.py` | n/a | six identities | save/load prediction test | PASS |

## Runtime verification

- `python -m compileall`: **PASS**.
- Full unit suite: **PASS**.
- Real Task-3 image smoke flow (Stage 1 → prior → HSIC → candidate-only
  KCI → Stage 2 backward → checkpoint reload → inference): **PASS** on CUDA.
- Five-fold generation logic and patient-disjointness: **PASS** with the
  automated controlled fixture.
- Five-fold generation from the currently loaded local classification
  manifests: **IMPLEMENTED BUT NOT RUNTIME-VERIFIED**. The preflight guard
  correctly rejected at least one `patient_id` associated with inconsistent
  class labels. Those released manifests also lack structured non-target
  clinical variables required for task-specific `Z_t`; the code does not
  fabricate either patient identity or clinical conditioning variables.

## Saved screening provenance

Each fold writes independent artifacts beneath:

```text
checkpoints/unified_five_folds/fold_<k>/screening/<task_identity>/
├── hsic_statistics.json
├── hsic_pvalues.json
├── hsic_rejected.json
├── kci_statistics.json
├── kci_pvalues.json
├── kci_rejected.json
├── retention_ratio.json
└── selected_channels.json
```

The accompanying `relation_prior_audit.json` and
`channel_screening_audit.json` record partition provenance, patient IDs,
normalization, kernel/GCV values, permutation counts, bootstrap estimates and
selected channels. No validation, test or external loader is accepted by the
statistics path.
