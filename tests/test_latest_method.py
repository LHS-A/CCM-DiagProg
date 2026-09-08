import torch
from src.models.unified_causal_ccm import OUTPUT_DIMS, TASKS
from src.models.relations import adaptive_bootstrap, build_relation_prior, distance_correlation, normalized_mutual_information
from src.models.screening import gcv_regularization, screen_channels
import numpy as np


def test_six_task_contract():
    assert len(TASKS) == 6
    assert OUTPUT_DIMS == (3, 3, 4, 1, 4, 1)
    assert sum(OUTPUT_DIMS) == 16


def test_multiview_relations_and_adaptive_bootstrap():
    rng = np.random.default_rng(7); x = rng.normal(size=80); y = x + rng.normal(scale=.05, size=80)
    assert distance_correlation(x, y) > .9
    assert normalized_mutual_information(x, y, 5) > 0
    result = adaptive_bootstrap(lambda index: np.stack((x[index].mean() * np.ones(4),)), len(x), 9,
                                initial=10, increment=5, maximum=20, confidence=.95)
    assert result.iterations <= 20 and np.isclose(result.weights.sum(), 1)


def test_gcv_uses_configured_candidates():
    x = torch.linspace(-1, 1, 16)[:, None]
    kernel = torch.exp(-torch.cdist(x, x).square())
    candidates = torch.logspace(-6, 1, 50)
    selected = gcv_regularization(kernel, candidates)
    assert bool((selected == candidates).any())


def test_training_only_prior_and_screening_pipeline():
    rng=np.random.default_rng(12);clinical=rng.normal(size=(32,2));features=np.stack((clinical[:,0],clinical[:,1],clinical.sum(1),rng.normal(size=32)),1)
    prior,audit=build_relation_prior(torch.tensor(features,dtype=torch.float32),clinical,seed=3,k=2,initial=5,increment=5,maximum=10)
    assert prior.shape==(4,4) and torch.isfinite(prior).all() and audit['clinical_bootstraps']<=10
    descriptors=torch.tensor(np.stack((features,features**2),-1),dtype=torch.float32);target=torch.tensor(clinical[:,0],dtype=torch.float32);nuisance=torch.tensor(clinical[:,1:],dtype=torch.float32)
    selected,details=screen_channels(descriptors,target,nuisance,False,.75,.2,5,permutation_min=25,permutation_max=100,permutation_confidence=.90,gcv_candidates=5)
    assert len(selected)>0 and details['lambda_kci']>0
