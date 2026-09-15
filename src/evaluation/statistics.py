from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm


def paired_permutation_test(target: np.ndarray,prediction_a: np.ndarray,prediction_b: np.ndarray,
                            metric: Callable[[np.ndarray,np.ndarray],float],*,resamples: int=10000,
                            seed: int=3407) -> tuple[float,float]:
    """Two-sided paired randomization test after case-level aggregation."""
    target=np.asarray(target);a=np.asarray(prediction_a);b=np.asarray(prediction_b)
    if len(target)!=len(a) or len(a)!=len(b):raise ValueError('paired inputs must have equal case counts')
    observed=float(metric(target,a)-metric(target,b));rng=np.random.default_rng(seed);exceed=0
    for _ in range(resamples):
        swap=rng.integers(0,2,len(target),dtype=bool);shape=(len(target),)+(1,)*(a.ndim-1)
        swap=swap.reshape(shape);left=np.where(swap,b,a);right=np.where(swap,a,b)
        exceed+=abs(float(metric(target,left)-metric(target,right)))>=abs(observed)
    return observed,float((exceed+1)/(resamples+1))


def _placements(positive: np.ndarray,negative: np.ndarray) -> tuple[float,np.ndarray,np.ndarray]:
    comparison=(positive[:,None]>negative[None,:]).astype(float)+.5*(positive[:,None]==negative[None,:])
    v10=comparison.mean(1);v01=comparison.mean(0)
    return float(v10.mean()),v10,v01


def delong_auc_test(target: np.ndarray,score_a: np.ndarray,score_b: np.ndarray) -> tuple[float,float,float]:
    """Two-sided paired DeLong test for two binary-classification AUCs."""
    target=np.asarray(target).astype(bool);scores=np.stack((score_a,score_b))
    if target.all() or (~target).all():raise ValueError('DeLong requires both binary classes')
    auc=[];positive=[];negative=[]
    for score in scores:
        value,v10,v01=_placements(np.asarray(score)[target],np.asarray(score)[~target]);auc.append(value);positive.append(v10);negative.append(v01)
    covariance=np.cov(np.stack(positive),ddof=1)/target.sum()+np.cov(np.stack(negative),ddof=1)/(~target).sum()
    variance=float(covariance[0,0]+covariance[1,1]-2*covariance[0,1]);difference=float(auc[0]-auc[1])
    p=1.0 if variance<=0 and difference==0 else (0.0 if variance<=0 else float(2*norm.sf(abs(difference)/np.sqrt(variance))))
    return difference,p,variance


def bonferroni(pvalues: np.ndarray|list[float]) -> np.ndarray:
    values=np.asarray(pvalues,float);return np.minimum(values*values.size,1.0)
