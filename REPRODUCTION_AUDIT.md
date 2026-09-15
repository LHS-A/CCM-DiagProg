# Reproduction audit

## Authoritative manuscript

The sole methodology source is `Paper.pdf`, created 2026-09-16, with SHA-256
`bb3d05ec9f2c169a6700cce426c24d716d3cba72902e81c4199926ef2c7e7890`.
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
| HSIC coarse pool | Sec. 2.1.3 | `screen_channels` | 2048 scores | all evaluated channels retained | fixed | significance prematurely reduced pool | all scores enter `S_HSIC` | PASS |
| KCI filtering | Sec. 2.1.3 | `screen_channels`, `gcv_regularization` | 2048 conditional scores | train-only `Z_t`, RBF/GCV | fixed | candidate-only implementation | KCI applied to complete paper coarse pool | PASS |
| Top-rho retention | Sec. 2.1.3 | `top_rho_retention` | `K=floor(0.3×2048)=614` | fixed after Stage 1 | task subset restored | `round` and incomplete budget | floor, KCI ranking and exact HSIC backfill | PASS |
| Task-safe context | Sec. 2.2.1 | `task_available_fields`, `task_exclusions`, `CCMManifestDataset` | field list | target/future/proxy removed | identical mask | implicit exclusion only | explicit task allow/exclude rules | PASS |
| Context dropout | Eq. (5) | `CCMManifestDataset.__getitem__` | `k∈{0,…,K}` | field-level uniform removal | disabled unless ablation requested | implemented | zero-context emits empty text, not a missing token | PASS |
| Fixed clinical sentence | Sec. 2.2.1 | `_clinical_fields`, `_serialize_clinical` | text length ≤128 tokens | generated after masking/dropout | generated from available fields | semicolon concatenation | fixed paper scaffold and omitted missing phrases | PASS |
| Tiny ClinicalBERT | Sec. 2.2.1/3.2 | `UnifiedCausalCCM.text_encoder` | `B×L×312 → B×L×768` | trained in Stage 2 | active when context exists | implemented | config/dimension checks | PASS |
| Semantic visual contextualization | Eq. (6) | `SemanticContext.forward` | `B×144×614 → B×144×256 → B×256` | Stage 2 | active | implemented | configurable exact dimensions and head assertion | PASS |
| Dynamic task generator | Eq. (7) | `UnifiedCausalCCM.forward` | `W:B×256×o_t`, `b:B×o_t` | Stage 2 | generated per patient/task | implemented | removed hard-coded internal widths | PASS |
| Relation-stage objective | Eq. (8) | `balanced_epoch` | scalar | `L_aux+lambda_prior L_prior` | n/a | implemented | exact Eq. (4) feeds objective | PASS |
| Task-balanced objective | Eq. (9) | `balanced_epoch` | scalar | within-task mean then equal task mean | n/a | implemented | six-identity optimizer-step test | PASS |
| Hypernetwork regularization | Eq. (10) | `UnifiedCausalCCM.forward`, `balanced_epoch` | scalar | sample mean of squared `W,b` norms | n/a | reduction previously corrected | numerical equation test retained | PASS |
| Five-fold evaluation | Sec. 3.1–3.3 | `evaluate_unified_five_folds.py`, `infer.py` | fold metrics and mean±SD | no external access | five fixed models | single-checkpoint entry only | full internal/external orchestrator | PASS |
| Statistical comparison | Sec. 3.2 | `src/evaluation/statistics.py` | paired case arrays | n/a | post-inference only | absent | paired permutation, DeLong, Bonferroni | PASS |

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
| HSIC/KCI calibration | 1,000 fixed permutations with +1 correction and BH FDR 0.05 | deterministic standard nonparametric calibration without a channel proportion | screening/config | No |
| KCI regularization | GCV over 50 log-spaced values in `[1e-6,1e1]` | standard data-driven residualization | screening/config | No |
| Multi-image training statistics | every training image contributes one sample; patient IDs define folds and remain in provenance | follows the paper's sample-level `x_s^i` definition | `collect_statistics` | No |
| Image preprocessing | bilinear resize to 384 and ImageNet mean/std; no undocumented augmentation | does not introduce an unstated stochastic imaging method | dataset | No |
| Regression scale | raw clinical units with MSE | Eq. (9) specifies MSE and the paper reports native-unit metrics | training | No |
| Task batching | one batch per identity per optimizer step; shorter loaders cycle | implements equal task contribution despite cohort size differences | `balanced_epoch` | No |
| Stage convergence | warm-up and relation alignment capped at 100 epochs each; validation patience 20; Stage 2 uses the remaining global 500-epoch budget | operationalizes “after convergence” while preserving the global limit | train/config | No |
| Optimizer details not stated | PyTorch AdamW default betas/epsilon | standard implementation with no new tuned parameter | train | No |
| Five-fold assignment | seed 3407; stratified patient folds for classification, shuffled patient K-fold otherwise; fold `k` test and `k+1` validation | deterministic patient-disjoint 6:2:2 rotation | fold runner | No |
| Partial-context evaluation | half-up rounding of `qK`, newly sampled on every access | deterministic count with stochastic field subset | dataset/infer | No |
| Paired test resamples | 10,000, seed 3407 | stable two-sided randomization estimate | evaluation statistics | No |

## Bugs found

### Critical

- Eq. (4) used the wrong averaging denominator.
- HSIC significance incorrectly truncated the paper's complete coarse pool.
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
- Existing audit documentation used equation numbers from an older manuscript revision.

## Files changed

| File | Purpose |
|:--|:--|
| `src/models/causal_ccm.py` | exact Eq. (1), Eq. (3) and Eq. (4) reductions |
| `src/models/relations.py` | typed multi-view relations, pairwise stability weighting and fixed bootstrap |
| `src/models/screening.py` | paper HSIC pool, KCI filtering and exact top-rho budget |
| `src/models/unified_causal_ccm.py` | checked dimensions and dynamic parameter shapes |
| `src/data/dataset.py` | safe-field allowlists, typed variables, context dropout and fixed serialization |
| `src/data/manifests.py` | standardized baseline clinical fields without cross-identity leakage |
| `train.py` | task/fold statistics, phase convergence, best-state restore and exact objectives |
| `infer.py` | held-out filtering and paper-defined macro ACC |
| `scripts/train_unified_five_folds.py` | deterministic patient-level fold orchestration |
| `scripts/evaluate_unified_five_folds.py` | complete fold-specific internal/external evaluation |
| `src/evaluation/statistics.py` | paper statistical comparison procedures |
| `configs/default.yaml` | fixed reproducibility choices and explicit task-safe fields/types |
| `tests/test_reproduction_audit.py` | equation, leakage, task-state and two-stage tests |
| `tests/test_latest_method.py` | updated bootstrap/filtering contract |
| `tests/test_paper_statistics.py` | permutation, DeLong and multiplicity tests |
| `README.md` | runnable public training, inference and evaluation workflow |

## Tests performed

- `python -m compileall -q .`: PASS in both directories.
- `python -m py_compile` over all core model/data/evaluation/entry-point files:
  PASS in both directories.
- Final repository full suite: **29 passed**.
- GitHub repository full suite: **23 passed**.
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
| `train.py` | `df6894d7a5d8d7d50c4f5990052bdf82d447ae1277b8367b49615ea9253e0191` |
| `infer.py` | `1de13719594bb3fa4b698e26e934b48bb870692746f1aebb70595ca807eaeaeb` |
| `scripts/train_unified_five_folds.py` | `d7a308f377d94e818d581b70b063fb5c3367c18ec9eb750212505927f0daf802` |
| `scripts/evaluate_unified_five_folds.py` | `2a8d6d4d3e953e23ffab245c793f732292f6818fdf2aaa78d72fb480bcc09be6` |
| `scripts/smoke_unified_real_data.py` | `68ae1bd36e30653b32b6b3225f06ca808a68e220117bfe1ff5434b019b077488` |
| `src/models/__init__.py` | `6a90c426325ffb98c7b41d1049af538abdc28fe82640a219b9ffbfd7ae01137c` |
| `src/models/causal_ccm.py` | `078476b5a73cfb4e33f142109e71685f3388cb62ee314cee79afaec33e3ae934` |
| `src/models/relations.py` | `d660b4f4df1544842927fbb8c186344974e36b90f40f48ab8a40936f36a1566f` |
| `src/models/screening.py` | `939f3e6b2ccd8fc4c39189e196e05309e96b175aa22538dbcdbcff9e44398e01` |
| `src/models/unified_causal_ccm.py` | `d6d343bcde84e4c80d7a1f631731e0b71eba5d941d3ff0db37660ded48a241a2` |
| `src/data/audit.py` | `247cfd107a04b0cbfdc76ed56c935c863db6c3ea5a0393e8844df9087bf85922` |
| `src/data/dataset.py` | `62ed3e7df9da378a671c73932a7053de0605e7f8a66f1a37884a219eacdc7a81` |
| `src/data/manifests.py` | `31305ba02018bb878bb34818fd0a1a4bb862f63b8bca42ef58722783fdb3fe28` |
| `src/evaluation/__init__.py` | `a64bb3d7a94aa02d9f12ab311e06cd107127867b5edf67af1362d560f6a57f72` |
| `src/evaluation/statistics.py` | `cc020d7a73fc2f4ec3d6ff0442544a0bdbb7210de12cf8cca7c83dbe8428316a` |
| `tests/test_latest_method.py` | `a1191ecc1dea6487e68c256f988f626f0475b781266b86a365b093409828a75b` |
| `tests/test_paper_statistics.py` | `90bf417cd52531cce20947e2af47dcd9057c0027c229eb3628a8d36a2c87c067` |
| `tests/test_reproduction_audit.py` | `0e599907f3a33fa0ec8410936794c40896a7903dce8d2c6c490710108c10519c` |

## Remaining limitations

- The currently loaded classification manifests contain at least one inferred
  patient identifier associated with inconsistent labels and do not provide
  authoritative structured clinical metadata. The strict pipeline rejects
  these rows rather than fabricating identity or clinical context.
- The NVIDIA driver is not visible in the current execution environment.
  Formula/module tests and the real-image end-to-end smoke flow can be verified
  on CPU, but a new full six-task training run cannot be represented as a CUDA
  runtime pass in this audit.
