from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error,roc_auc_score


def aggregate_case_predictions(case_ids: np.ndarray,target: np.ndarray,*predictions: np.ndarray,
                               classification: bool|None=None) -> tuple[np.ndarray,...]:
    """Average repeated image predictions before a paper statistical test.

    Classification labels must be constant within a case. Continuous targets
    are averaged, matching the case-level aggregation used for predictions.
    """
    cases=np.asarray(case_ids).astype(str);truth=np.asarray(target);values=[np.asarray(x) for x in predictions]
    if any(len(x)!=len(cases) for x in (truth,*values)):raise ValueError('case IDs, targets and predictions must have equal row counts')
    unique=[];groups=[]
    for case in cases:
        if case not in unique:unique.append(case)
    for case in unique:groups.append(np.flatnonzero(cases==case))
    if classification is None:classification=np.issubdtype(truth.dtype,np.integer) and truth.ndim==1
    aggregated_truth=[];aggregated=[[] for _ in values]
    for indices in groups:
        observed=truth[indices]
        if classification:
            if len(np.unique(observed))!=1:raise ValueError('classification target is inconsistent within a case')
            aggregated_truth.append(observed[0])
        else:aggregated_truth.append(observed.mean(axis=0))
        for destination,value in zip(aggregated,values):destination.append(value[indices].mean(axis=0))
    return (np.asarray(aggregated_truth),*(np.asarray(x) for x in aggregated))


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


def paper_statistical_test(case_ids: np.ndarray,target: np.ndarray,prediction_a: np.ndarray,prediction_b: np.ndarray,
                           analysis: str,*,resamples: int=10000,seed: int=3407) -> tuple[float,float]:
    """Run the manuscript-specified paired case-level comparison.

    ``analysis`` is one of ``multiclass_auc``, ``binary_auc`` or
    ``regression_mae``. Multiplicity correction is deliberately performed by
    the caller across the complete family of baseline comparisons.
    """
    truth,a,b=aggregate_case_predictions(case_ids,target,prediction_a,prediction_b,classification=analysis!='regression_mae')
    if analysis=='multiclass_auc':
        metric=lambda y,p:roc_auc_score(y,p,multi_class='ovr',average='macro')
        return paired_permutation_test(truth,a,b,metric,resamples=resamples,seed=seed)
    if analysis=='regression_mae':
        return paired_permutation_test(truth,a,b,mean_absolute_error,resamples=resamples,seed=seed)
    if analysis=='binary_auc':
        difference,pvalue,_=delong_auc_test(truth,a,b);return difference,pvalue
    raise ValueError(f'unsupported paper statistical analysis {analysis!r}')
