# Reproduction audit

## Authoritative manuscript

The sole methodology source is the 2026-09-17 revision of `Paper.pdf`, with
SHA-256 `92c394b244ca2b44bb1a019ac6d4358fa4c2365c454fbb58fb05c6cdbd91c2a2`.
Older manuscripts, checkpoints, reports and implementations were not treated
as methodological evidence.

## Paper-to-code mapping

| Paper component | Equation/section | Python implementation | Core shapes | Train-time behavior | Inference behavior | Before | Modification | After |
|:--|:--|:--|:--|:--|:--|:--|:--|:--:|
| Shared visual encoder | Sec. 2.1.1 | `ResNet50Features` | `B×3×384×384 → B×2048×12×12` | warm-up and relation alignment | frozen | implemented | backbone/config assertions | PASS |
| Visual relation map | Eq. (1) | `StructuralPrior.loss` | `M_vis: 2048×2048` | mini-batch relation estimate | not re-estimated | epsilon was inside square root | exact norm product plus epsilon | PASS |
| Multi-view clinical relation | Eq. (2) | `association_views`, `bootstrap_stability` | `C_clin,t: N_t×M_t`, views `M_t×M_t×4` | task/fold/train only | fixed | every encoded column used every view | type-applicable views and pairwise weight renormalization | PASS |
| Clinical-to-visual projection | Sec. 2.1.2 inline definition | `build_relation_prior` | `Pi_t: 2048×M_t` | task/fold/train only | restored | incomplete type handling | typed multi-view association and row softmax | PASS |
| Clinical prior | Sec. 2.1.2 inline definition | `build_relation_prior`, `construct_priors` | `Pi_t A_rel,t Pi_t^T: 2048×2048` | constructed once after warm-up | fixed/restored | task state existed but estimator was approximate | exact task/fold provenance and typed estimator | PASS |
| Off-diagonal normalization | Eq. (3) | `StructuralPrior.loss` | `2048×2048` | current task prior | fixed | present | formula comments and exact path verification | PASS |
| Prior alignment loss | Eq. (4) | `StructuralPrior.loss` | scalar | Stage 1 gradient to visual encoder only | not optimized | divided by `C²` | divided by `C(C−1)` with stopped prior gradient | PASS |
| Channel descriptor | Sec. 2.1.3 | `collect_statistics` | `N_t×2048×2` | `[GAP,GMP]`, train patients only | restored selection | present | non-varying nuisance removal | PASS |
| HSIC coarse filtering | Sec. 2.1.3 | `filter_channels` | 2048 scores and BH decisions | statistically supported channels form `S_HSIC` | fixed | previous revision retained all evaluated channels | restore significance-defined coarse set | PASS |
| KCI fine filtering | Sec. 2.1.3 | `filter_channels`, `gcv_regularization` | `|S_HSIC|` conditional scores | train-only `Z_t`, RBF/GCV/BH | fixed | KCI evaluated all channels | KCI restricted to `S_HSIC`; `S_sig=S_HSIC∩S_KCI` | PASS |
| Top-rho retention | Sec. 2.1.3 | `top_rho_retention` | `K=floor(0.3×2048)=614` | fixed after Stage 1 | task subset restored | `round` and incomplete budget | floor, KCI ranking and exact HSIC backfill | PASS |
| Task-safe context | Sec. 2.2.1 | `task_available_fields`, `task_exclusions`, `CCMManifestDataset` | field list | target/future/proxy removed | identical mask | implicit exclusion only | explicit task allow/exclude rules | PASS |
| Context dropout | Sec. 2.2.1 inline definition | `CCMManifestDataset.__getitem__` | `k∈{0,…,n_clin}` | field-level uniform removal | disabled unless ablation requested | implemented | zero-context emits empty text, not a missing token | PASS |
| Fixed clinical sentence | Sec. 2.2.1 | `_clinical_fields`, `_serialize_clinical` | text length ≤128 tokens | generated after masking/dropout | generated from available fields | semicolon concatenation | fixed paper scaffold and omitted missing phrases | PASS |
| Tiny ClinicalBERT | Sec. 2.2.1/3.2 | `UnifiedCausalCCM.text_encoder` | `B×L×312 → B×L×768` | trained in Stage 2 | active when context exists | implemented | config/dimension checks | PASS |
| Semantic visual contextualization | Eq. (5) | `SemanticContext.forward` | `B×144×614 → B×144×256 → B×256` | Stage 2 | active | implemented | configurable exact dimensions and head assertion | PASS |
| Dynamic setting generator | Eq. (6) | `UnifiedCausalCCM.forward` | `W:B×256×o_t`, `b:B×o_t` | Stage 2 | generated per patient/setting | implemented | checked against revised setting terminology and image index | PASS |
| Relation-stage objective | Eq. (7) | `balanced_epoch` | scalar | `L_aux+lambda_prior L_prior` | n/a | implemented | exact Eq. (4) feeds objective | PASS |
| Setting-balanced objective | Eq. (8) | `balanced_epoch` | scalar | within-setting mean then equal setting mean | n/a | implemented | six-setting optimizer-step test | PASS |
| Hypernetwork regularization | Eq. (9) | `UnifiedCausalCCM.forward`, `balanced_epoch` | scalar | image-patient-setting tuple mean of squared `W,b` norms | n/a | implemented | numerical equation test retained | PASS |
| Five-fold evaluation | Sec. 3.1–3.3 | `evaluate_unified_five_folds.py`, `infer.py` | fold metrics and mean±SD | no external access | five fixed models | single-checkpoint entry only | full internal/external orchestrator | PASS |
| Statistical comparison | Sec. 3.2 | `src/evaluation/statistics.py` | paired case arrays | n/a | post-inference only | repeated images were not programmatically aggregated | case aggregation, AUC/MAE-specific paired permutation, binary DeLong, Bonferroni | PASS |

## Reproduction assumptions

| Paper unspecified detail | Chosen implementation | Reason | Affected code | Changes paper method? |
|:--|:--|:--|:--|:--:|
| Exact Tiny checkpoint | `nlpie/tiny-clinicalbert` | public Tiny clinical BERT with reproducible tokenizer/model ID | config, model | No |
| Tiny hidden width versus semantic width | learned `312→768` projection | preserves the fixed patient-semantic and hypernetwork dimensions | model | No |
| Token limit | 128 with fixed padding/truncation | covers the fixed scaffold without excessive computation | dataset/config | No |
| Attention configuration | 256 dimensions, 8 heads | standard divisible multi-head configuration | model/config | No |
| Task/latent dimensions | task 32, semantic 768, hyper input 800, trunk `512→256` | minimal fixed architecture matching all stated tensor roles | model/config | No |
| Numerical epsilon | `1e-6` for relation normalization | standard stable nonzero denominator | structural prior | No |
| Clinical missing values | training mean for ordered variables; training mode for nominal/binary | deterministic, type-appropriate and leakage-free | relations | No |
| Non-informative clinical columns | discard all-missing or non-varying columns inside each fold | undefined relations carry no information | training statistics | No |
| NMI estimator | `k`-NN with `k=5`; empirical joint MI for categorical pairs; `I/sqrt(HxHy)` | standard continuous/discrete estimators | relations/config | No |
| Bootstrap count/unit | 1,000 image-sample resamples with replacement; sample variance | follows the paper's sample-level definition | relations/training | No |
| HSIC/KCI calibration | 1,000 fixed permutations with +1 correction and BH FDR 0.05 | deterministic standard nonparametric calibration without a channel proportion | filtering/config | No |
| KCI regularization | GCV over 50 log-spaced values in `[1e-6,1e1]` | standard data-driven residualization | filtering/config | No |
| Multi-image training statistics | every training image contributes one sample; patient IDs define folds and remain in provenance | follows the paper's sample-level `x_s^i` definition | `collect_statistics` | No |
| Image preprocessing | bilinear resize to 384 and ImageNet mean/std; no undocumented augmentation | does not introduce an unstated stochastic imaging method | dataset | No |
| Regression scale | raw clinical units with MSE | Eq. (8) specifies MSE and the paper reports native-unit metrics | training | No |
| Task batching | one batch per identity per optimizer step; shorter loaders cycle | implements equal task contribution despite cohort size differences | `balanced_epoch` | No |
| Stage convergence | warm-up and relation alignment capped at 100 epochs each; validation patience 20; Stage 2 uses the remaining global 500-epoch budget | operationalizes “after convergence” while preserving the global limit | train/config | No |
| Optimizer details not stated | PyTorch AdamW default betas/epsilon | standard implementation with no new tuned parameter | train | No |
| Five-fold assignment | seed 3407; stratified patient folds for classification, shuffled patient K-fold otherwise; fold `k` test and `k+1` validation | deterministic patient-disjoint 6:2:2 rotation | fold runner | No |
| Partial-context evaluation | half-up rounding of `qK`, newly sampled on every access | deterministic count with stochastic field subset | dataset/infer | No |
| Paired test resamples | 10,000, seed 3407 | stable two-sided randomization estimate | evaluation statistics | No |

## Bugs found

### Critical

- Eq. (4) used the wrong averaging denominator.
- The previous revision's all-channel HSIC pool contradicted the revised paper's significance-defined `S_HSIC`.
- KCI was consequently evaluated outside `S_HSIC`, contrary to the revised coarse-to-fine procedure.
- The final top-rho set could contain fewer than the required fixed number of channels.
- Clinical text bypassed the fixed linguistic template.
- Mixed clinical variable types did not control applicable relation views.
- The reported macro ACC path used overall accuracy instead of the paper's macro class-wise ACC.

### Major

- Eq. (1) placed epsilon inside the square root rather than after the product of norms.
- `round(rho*C)` was used instead of `floor(rho*C)`.
- Categorical variables were expanded and then treated as ordered numeric quantities.
- Training phases did not restore their best validation state before the following phase.
- No single public command evaluated all five internal folds and fixed external cohorts.
- Several architecture widths were hard-coded instead of checked against the reproducibility configuration.

### Minor

- Configuration retained switches that did not affect the authoritative execution path.
- Missing context was serialized as an artificial sentence instead of bypassing the semantic encoder with zero tensors.
- Existing audit documentation used equation numbers and terminology from an older manuscript revision.

## Files changed

| File | Purpose |
|:--|:--|
| `src/models/causal_ccm.py` | exact Eq. (1), Eq. (3) and Eq. (4) reductions |
| `src/models/relations.py` | typed multi-view relations, pairwise stability weighting and fixed bootstrap |
| `src/models/filtering.py` | significance-defined HSIC pool, pool-restricted KCI and exact top-rho budget |
| `src/models/unified_causal_ccm.py` | checked dimensions and dynamic parameter shapes |
| `src/data/dataset.py` | safe-field allowlists, typed variables, context dropout and fixed serialization |
| `src/data/manifests.py` | standardized baseline clinical fields without cross-identity leakage |
| `train.py` | task/fold statistics, phase convergence, best-state restore and exact objectives |
| `infer.py` | held-out filtering and paper-defined macro ACC |
| `scripts/train_unified_five_folds.py` | deterministic patient-level fold orchestration |
| `scripts/evaluate_unified_five_folds.py` | complete fold-specific internal/external evaluation |
| `src/evaluation/statistics.py` | case aggregation and metric-specific paper statistical comparisons |
| `configs/default.yaml` | fixed reproducibility choices and explicit task-safe fields/types |
| `tests/test_reproduction_audit.py` | equation, leakage, task-state and two-stage tests |
| `tests/test_latest_method.py` | updated bootstrap/filtering contract |
| `tests/test_paper_statistics.py` | permutation, DeLong and multiplicity tests |
| `README.md` | runnable public training, inference and evaluation workflow |

## Tests performed

- `python -m compileall -q .`: PASS in both directories.
- `python -m py_compile` over all core model/data/evaluation/entry-point files:
  PASS in both directories.
- Final repository full suite: **33 passed**.
- GitHub repository full suite: **27 passed**.
- Synthetic six-identity Stage 1 and Stage 2 optimizer steps: PASS, including
  backward gradients and task-balanced losses.
- Full/partial/zero clinical-context forward paths: PASS.
- Real Task-3 image flow on CPU: data loading → warm-up backward → typed
  `C_clin` → `A_rel` → `Pi` → `M_prior` → relation backward → HSIC → KCI →
  exact top-rho → Stage-2 backward → checkpoint reload → inference: PASS.
- CUDA rerun: NOT RUNTIME-VERIFIED because the current process cannot
  communicate with the installed NVIDIA driver.

## Dual-directory parity

Core methodology parity: **PASS**. Every file below is byte-identical.

| Core file | SHA-256 |
|:--|:--|
| `train.py` | `16f94f2ee3b05afa987cb960bb8a583961eedbdceb7a1bc375ada542d4edfbc1` |
| `infer.py` | `1de13719594bb3fa4b698e26e934b48bb870692746f1aebb70595ca807eaeaeb` |
| `scripts/train_unified_five_folds.py` | `d7a308f377d94e818d581b70b063fb5c3367c18ec9eb750212505927f0daf802` |
| `scripts/evaluate_unified_five_folds.py` | `2a8d6d4d3e953e23ffab245c793f732292f6818fdf2aaa78d72fb480bcc09be6` |
| `scripts/smoke_unified_real_data.py` | `5b0a4dcb80fad78605f7249da205d5ac845c0c63b4fc07800e2c7350eb681e04` |
| `src/models/__init__.py` | `6a90c426325ffb98c7b41d1049af538abdc28fe82640a219b9ffbfd7ae01137c` |
| `src/models/causal_ccm.py` | `078476b5a73cfb4e33f142109e71685f3388cb62ee314cee79afaec33e3ae934` |
| `src/models/relations.py` | `d660b4f4df1544842927fbb8c186344974e36b90f40f48ab8a40936f36a1566f` |
| `src/models/filtering.py` | `08866fab5f067d0e0f92eb6a6dce9b61cd5fbd81d0b2e4be3a48414ddae01622` |
| `src/models/unified_causal_ccm.py` | `d6d343bcde84e4c80d7a1f631731e0b71eba5d941d3ff0db37660ded48a241a2` |
| `src/data/audit.py` | `247cfd107a04b0cbfdc76ed56c935c863db6c3ea5a0393e8844df9087bf85922` |
| `src/data/dataset.py` | `62ed3e7df9da378a671c73932a7053de0605e7f8a66f1a37884a219eacdc7a81` |
| `src/data/manifests.py` | `31305ba02018bb878bb34818fd0a1a4bb862f63b8bca42ef58722783fdb3fe28` |
| `src/evaluation/__init__.py` | `b16644812215ba558b30b14ef40ec517468c97426f0a17290388e9f184089354` |
| `src/evaluation/statistics.py` | `cf6134d72f35c3c7d8e356a40c7149599682591aaac9d9d60697d2902b28c9fb` |
| `tests/test_latest_method.py` | `a6f0bff7b724cfbd20218bf6cba2b18cb700eeee296b55ed14f0cc6246af5b1f` |
| `tests/test_paper_statistics.py` | `df2510090f9563ba12ff6bb7b301a3da38e0ed4378c14fc949ab73051aa49fc0` |
| `tests/test_reproduction_audit.py` | `0aabe1bcd641c813e61225a3daa506a5063669ba5c19a77b745ce573bd3c7641` |

## Remaining limitations

- The currently loaded classification manifests contain at least one inferred
  patient identifier associated with inconsistent labels and do not provide
  authoritative structured clinical metadata. The strict pipeline rejects
  these rows rather than fabricating identity or clinical context.
- The NVIDIA driver is not visible in the current execution environment.
  Formula/module tests and the real-image end-to-end smoke flow can be verified
  on CPU, but a new full six-task training run cannot be represented as a CUDA
  runtime pass in this audit.
