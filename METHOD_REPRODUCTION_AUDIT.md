# Method reproduction audit

| Method component | Source file | Class / function | Training | Inference | Status |
|:--|:--|:--|:--:|:--:|:--:|
| Visual representation | `src/models/causal_ccm.py` | `ResNet50Features.forward` | Yes | Yes | ✓ |
| Min–max visual dependency modeling | `src/models/causal_ccm.py` | `StructuralPrior.loss` | Yes | Fixed | ✓ |
| Multi-view clinical relation modeling | `src/models/relations.py` | `association_views` | Yes | Fixed | ✓ |
| Adaptive bootstrap stability weighting | `src/models/relations.py` | `adaptive_bootstrap` | Yes | Fixed | ✓ |
| Clinical-to-visual relation projection | `src/models/relations.py` | `build_relation_prior` | Yes | Fixed | ✓ |
| Clinically induced structural prior | `src/models/relations.py` | `build_relation_prior` | Yes | Fixed | ✓ |
| Off-diagonal prior alignment | `src/models/causal_ccm.py` | `StructuralPrior.loss` | Yes | No | ✓ |
| HSIC target-relevance screening | `src/models/screening.py` | `screen_channels` | Training patients only | Fixed indices | ✓ |
| KCI conditional screening and GCV | `src/models/screening.py` | `screen_channels`, `gcv_regularization` | HSIC candidates from training patients only | Fixed indices | ✓ |
| Sequential permutation and BH/FDR | `src/models/screening.py` | `_sequential_many`, `_bh` | Training patients only | Fixed indices | ✓ |
| Task-specific Top-ρ retention | `train.py` | `task_setting`, `screen_all` | Independently for six identities | Restored from checkpoint | ✓ |
| Task-specific feature projection | `src/models/unified_causal_ccm.py` | `set_channels`, `SemanticContext.forward` | Yes | Yes | ✓ |
| Target/proxy masking | `src/data/dataset.py` | `CCMManifestDataset.__getitem__` | Yes | Yes | ✓ |
| Clinical-context dropout | `src/data/dataset.py` | `CCMManifestDataset.__getitem__` | Resampled | Resampled | ✓ |
| Tiny ClinicalBERT encoding | `src/models/unified_causal_ccm.py` | `text_encoder`, `text_projection` | Stage II | Yes | ✓ |
| Semantic visual contextualization | `src/models/unified_causal_ccm.py` | `SemanticContext.forward` | Stage II | Yes | ✓ |
| Residual visual pathway | `src/models/unified_causal_ccm.py` | `SemanticContext.forward` | Stage II | Yes, including missing text | ✓ |
| Patient–task conditioning | `src/models/unified_causal_ccm.py` | `UnifiedCausalCCM.forward` | Stage II | Yes | ✓ |
| Shared hypernetwork | `src/models/unified_causal_ccm.py` | `hyper` | Stage II | Yes | ✓ |
| Task-specific generators | `src/models/unified_causal_ccm.py` | `generators` | Stage II | Yes | ✓ |
| Dynamic prediction parameters | `src/models/unified_causal_ccm.py` | `UnifiedCausalCCM.forward` | Stage II | Yes | ✓ |
| Classification/regression/prognosis outputs | `train.py`, `infer.py` | `prediction_loss`, metric/output branches | Yes | Yes | ✓ |
| Two-stage optimization | `train.py` | `main`, `balanced_epoch` | Yes | N/A | ✓ |
| Paper cosine learning-rate schedule | `train.py` | `learning_rate`, `set_learning_rate` | Every epoch | N/A | ✓ |
| Patient-level five-fold 6:2:2 evaluation | `scripts/train_unified_five_folds.py` | `patient_folds`, `create_manifests`, `main` | Fresh model and screening per fold | Fold checkpoint | ✓ |
| Stage-specific freezing | `src/models/unified_causal_ccm.py` | `set_stage_trainability` | Yes | N/A | ✓ |
| Task-balanced losses and regularization | `train.py` | `balanced_epoch` | Yes | N/A | ✓ |
| Checkpointed task subsets | `src/models/unified_causal_ccm.py` | `channel_indices`, `channel_counts`, `load_state_dict` | Saved | Restored and used | ✓ |
| Complete requested-task inference | `infer.py` | `main` | N/A | Yes | ✓ |

All clinical relations, normalization statistics, bootstrap estimates, kernel bandwidths, GCV coefficients, permutation tests, FDR decisions, and channel retention decisions are computed in `train.py` exclusively from each identity's `train` loader. Validation and test loaders are not passed into either prior construction or screening.
