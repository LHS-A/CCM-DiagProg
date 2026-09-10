# Paper-to-code reproduction audit

The authoritative execution path is `train.py` → `UnifiedCausalCCM` and
`infer.py` → `UnifiedCausalCCM`. Statistics used by relation construction and
screening are guarded as training-partition-only and are recomputed inside
every cross-validation fold.

## Numbered equations

| Paper equation | Mathematical operation | Implementation | Called in | Test |
|:--|:--|:--|:--|:--:|
| Eq. (1) | Min–max channel normalization and normalized spatial co-activation `M_vis` | `src/models/causal_ccm.py::StructuralPrior.loss` | Stage-1 relation alignment | PASS |
| Eq. (2) | Pearson, Spearman, dCor and NMI clinical relation views | `src/models/relations.py::association_views` | 1,000 task/fold bootstrap samples in `construct_priors` | PASS |
| Eq. (3) | `M_prior,t = Pi_t A_rel,t Pi_t^T` | `src/models/relations.py::build_relation_prior` | Once per task/fold after warm-up | PASS |
| Eq. (4) | Off-diagonal Frobenius normalization and stop-gradient `L_prior,t` | `src/models/causal_ccm.py::StructuralPrior.loss` | Correct task prior on every Stage-1 batch | PASS |
| Eq. (5) | Centered-kernel HSIC and sequential permutation significance | `src/models/screening.py::screen_channels` | Per task/fold | PASS |
| Eq. (6) | Joint kernel, GCV residualization and candidate-only KCI | `src/models/screening.py::gcv_regularization`, `screen_channels` | After the corresponding HSIC candidate set | PASS |
| Eq. (7) | Projected visual queries, ClinicalBERT keys/values, residual attention and LayerNorm | `src/models/unified_causal_ccm.py::SemanticContext.forward` | Stage 2 and inference | PASS |
| Eq. (8) | `G_t(H([e_t || h_cls])) -> [vec(W); b]` | `src/models/unified_causal_ccm.py::UnifiedCausalCCM.forward` | Stage 2 and inference | PASS |
| Eq. (9) | `f_s,t^T W_s,t + b_s,t` | `src/models/unified_causal_ccm.py::UnifiedCausalCCM.forward` | Stage 2 and inference | PASS |
| Eq. (10) | Mean squared Frobenius/Euclidean magnitude of generated `W,b` | `UnifiedCausalCCM.forward`, `train.py::balanced_epoch` | Stage-2 objective | PASS |

## Component and execution-path audit

| Paper component | File / function | Training | Inference | Automated evidence | Status |
|:--|:--|:--:|:--:|:--|:--:|
| Shared ResNet-50 visual encoder | `unified_causal_ccm.py::visual_encoder` | Yes | Yes | six-task forward | PASS |
| Visual dependency and prior alignment | `causal_ccm.py::StructuralPrior.loss` | Stage 1 | fixed encoder | formula and gradient tests | PASS |
| Pearson/Spearman/dCor/NMI clinical relations | `relations.py::association_views` | train only | fixed prior | relation tests | PASS |
| Adaptive bootstrap stability weighting | `relations.py::adaptive_bootstrap` | train only | no | deterministic audit metadata | PASS |
| Clinical-to-visual projection | `relations.py::build_relation_prior` | train only | fixed | shape/metadata checks | PASS |
| Task/fold prior isolation | `train.py::construct_priors`, `UnifiedCausalCCM.prior_for` | six independent states per fold | restored buffers | wrong-ID and cross-task routing tests | PASS |
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

Each task additionally saves `relation_priors/<identity>/C_clin_metadata.json`,
`A_rel.json`, `Pi.pt`, `M_prior.pt`, and `relation_view_weights.json`. The
metadata records task ID, fold ID, cohort, source-manifest digest, clinical
columns, normalized clinical-matrix digest, training-patient provenance and
training-only normalization statistics. Raw clinical rows are not duplicated
into public artifacts.
