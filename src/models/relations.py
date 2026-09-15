from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import hashlib

import numpy as np
import torch
from scipy.spatial import cKDTree, distance_matrix
from scipy.special import digamma
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if len(x) < 4: return 0.0
    a, b = distance_matrix(x[:, None], x[:, None]), distance_matrix(y[:, None], y[:, None])
    a = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    b = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov = np.mean(a * b); dvarx = np.mean(a * a); dvary = np.mean(b * b)
    return float(np.sqrt(max(dcov, 0.0) / np.sqrt(max(dvarx * dvary, 1e-24))))


def _knn_entropy(x: np.ndarray, k: int = 5) -> float:
    x = np.asarray(x, float).reshape(-1, 1); n = len(x)
    if n <= k or np.ptp(x) == 0: return 0.0
    eps = cKDTree(x).query(x, k=k + 1, p=2)[0][:, -1].clip(1e-12)
    return float(digamma(n) - digamma(k) + np.log(2.0) + np.mean(np.log(eps)))


def normalized_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    x, y = _finite_pair(x, y)
    if len(x) <= k or np.ptp(x) == 0 or np.ptp(y) == 0: return 0.0
    mi = mutual_info_regression(x[:, None], y, n_neighbors=k, random_state=0)[0]
    hx, hy = _knn_entropy(x, k), _knn_entropy(y, k)
    denominator = np.sqrt(max(abs(hx * hy), 1e-12))
    return float(np.clip(mi / denominator, 0.0, 1.0))


def relation_signature(x: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    x, y = _finite_pair(x, y)
    if len(x) < max(6, k + 1) or np.ptp(x) == 0 or np.ptp(y) == 0: return np.zeros(4)
    return np.array([abs(pearsonr(x, y)[0]), abs(spearmanr(x, y)[0]),
                     distance_correlation(x, y), normalized_mutual_information(x, y, k)])


def _entropy_discrete(value: np.ndarray) -> float:
    _,counts=np.unique(value,return_counts=True);probability=counts/counts.sum()
    return float(-(probability*np.log(probability)).sum())


def association_views(x: np.ndarray, y: np.ndarray, k: int = 5, chunk: int = 128,
                      x_types: list[str] | tuple[str,...] | None = None,
                      y_types: list[str] | tuple[str,...] | None = None) -> np.ndarray:
    """Eq. (2) applicable Pearson, Spearman, dCor and NMI views."""
    x=np.asarray(x,float);y=np.asarray(y,float);n,p=x.shape;m=y.shape[1]
    x_types=list(x_types or ["continuous"]*p);y_types=list(y_types or ["continuous"]*m)
    if len(x_types)!=p or len(y_types)!=m:raise ValueError("clinical type count does not match matrix width")
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("association_views requires finite, training-imputed arrays")
    ordered={"continuous","ordinal","binary"}
    def corr(a,b):
        a=(a-a.mean(0))/(a.std(0)+1e-12);b=(b-b.mean(0))/(b.std(0)+1e-12);return np.abs(a.T@b/max(n-1,1)).clip(0,1)
    pearson=corr(x,y);spearman=corr(rankdata(x,axis=0),rankdata(y,axis=0))
    applicable=np.array([[a in ordered and b in ordered for b in y_types] for a in x_types])
    pearson=np.where(applicable,pearson,np.nan);spearman=np.where(applicable,spearman,np.nan);dcor=np.zeros((p,m))
    for start in range(0,p,chunk):
        xb=x[:,start:start+chunk];dx=np.empty((n,n,xb.shape[1]))
        for local,kind in enumerate(x_types[start:start+xb.shape[1]]):
            dx[:,:,local]=(xb[:,None,local]!=xb[None,:,local]) if kind in {"nominal","binary"} else np.abs(xb[:,None,local]-xb[None,:,local])
        dx=dx-dx.mean(0,keepdims=True)-dx.mean(1,keepdims=True)+dx.mean((0,1),keepdims=True)
        vx=np.mean(dx*dx,axis=(0,1))
        for j in range(m):
            dy=(y[:,None,j]!=y[None,:,j]).astype(float) if y_types[j] in {"nominal","binary"} else np.abs(y[:,None,j]-y[None,:,j]);dy=dy-dy.mean(0)-dy.mean(1)[:,None]+dy.mean();vy=np.mean(dy*dy)
            cov=np.mean(dx*dy[:,:,None],axis=(0,1));dcor[start:start+len(xb.T),j]=np.sqrt(np.maximum(cov,0)/np.sqrt(np.maximum(vx*vy,1e-24)))
    discrete={"nominal","binary"};nmi=np.zeros((p,m));hx=np.array([_entropy_discrete(x[:,i]) if x_types[i] in discrete else _knn_entropy(x[:,i],min(k,n-1)) for i in range(p)])
    for j in range(m):
        if np.ptp(y[:,j])==0:continue
        yj=y[:,j];hy=_entropy_discrete(yj) if y_types[j] in discrete else _knn_entropy(yj,min(k,n-1));mi=np.zeros(p)
        continuous=[i for i,kind in enumerate(x_types) if kind not in discrete]
        nominal=[i for i,kind in enumerate(x_types) if kind in discrete]
        if y_types[j] in discrete:
            if continuous:mi[continuous]=mutual_info_classif(x[:,continuous],yj.astype(int),discrete_features=False,n_neighbors=min(k,n-1),random_state=0)
            for i in nominal:mi[i]=mutual_info_score(x[:,i],yj)
        else:
            if continuous:mi[continuous]=mutual_info_regression(x[:,continuous],yj,discrete_features=False,n_neighbors=min(k,n-1),random_state=0)
            for i in nominal:mi[i]=mutual_info_regression(x[:,[i]],yj,discrete_features=True,n_neighbors=min(k,n-1),random_state=0)[0]
        nmi[:,j]=np.clip(mi/np.sqrt(np.maximum(np.abs(hx*hy),1e-12)),0,1)
    return np.stack((pearson,spearman,dcor,nmi),-1)


@dataclass
class BootstrapResult:
    aggregate: np.ndarray
    weights: np.ndarray
    iterations: int


def _nan_mean_variance(value: np.ndarray,axis: int) -> tuple[np.ndarray,np.ndarray]:
    valid=np.isfinite(value);count=valid.sum(axis=axis);total=np.where(valid,value,0.0).sum(axis=axis)
    mean=np.divide(total,count,out=np.full_like(total,np.nan,dtype=float),where=count>0)
    expanded=np.expand_dims(mean,axis);squares=np.where(valid,(value-expanded)**2,0.0).sum(axis=axis)
    variance=np.divide(squares,count-1,out=np.full_like(total,np.nan,dtype=float),where=count>1)
    return mean,variance


def bootstrap_stability(estimator: Callable[[np.ndarray], np.ndarray], n: int, seed: int,
                        resamples: int = 1000, off_diagonal: bool = False) -> BootstrapResult:
    if n<2 or resamples<2:raise ValueError('bootstrap stability requires at least two samples and resamples')
    rng=np.random.default_rng(seed);indices=[rng.integers(0,n,n) for _ in range(resamples)]
    estimates=Parallel(n_jobs=min(8,resamples),prefer='threads')(delayed(estimator)(index) for index in indices)
    stack=np.stack(estimates);entries=stack.reshape(len(stack),-1,stack.shape[-1])
    if off_diagonal:
        if stack.shape[1]!=stack.shape[2]:raise ValueError('off-diagonal bootstrap requires square relation matrices')
        entries=entries[:,~np.eye(stack.shape[1],dtype=bool).reshape(-1)]
        if not entries.shape[1]:raise ValueError('at least two clinical variables are required')
    _,entry_variance=_nan_mean_variance(entries,0)
    valid_variance=np.isfinite(entry_variance);count=valid_variance.sum(0)
    per_view=np.divide(np.where(valid_variance,entry_variance,0).sum(0),count,out=np.full(entry_variance.shape[-1],np.nan),where=count>0);valid_view=np.isfinite(per_view)
    if not valid_view.any():raise RuntimeError('no applicable clinical relation view')
    weights=np.zeros_like(per_view);log_weight=-per_view[valid_view];value=np.exp(log_weight-log_weight.max());weights[valid_view]=value/value.sum()
    mean_views,_=_nan_mean_variance(stack,0);available=np.isfinite(mean_views)
    pair_weights=available*weights;pair_weights/=pair_weights.sum(-1,keepdims=True).clip(1e-12)
    aggregate=np.nansum(mean_views*pair_weights,axis=-1)
    return BootstrapResult(aggregate,weights,resamples)


def build_relation_prior(features: torch.Tensor, clinical: np.ndarray, *, seed: int, k: int = 5,
                         resamples: int = 1000,
                         clinical_columns: list[str] | None = None,
                         clinical_types: list[str] | None = None) -> tuple[torch.Tensor, dict]:
    """Training-only multi-view clinical prior projected into visual channel space."""
    z = features.detach().float().cpu().numpy(); clinical = np.asarray(clinical, float)
    n, channels = z.shape; variables = clinical.shape[1]
    clinical_columns=list(clinical_columns or [f"clinical_{i}" for i in range(variables)])
    clinical_types=list(clinical_types or ["continuous"]*variables)
    if len(clinical_columns)!=variables:raise ValueError('clinical column count does not match C_clin')
    if len(clinical_types)!=variables:raise ValueError('clinical type count does not match C_clin')
    clinical_mean=np.zeros(variables);clinical_std=np.ones(variables);normalized=np.empty_like(clinical)
    for column,kind in enumerate(clinical_types):
        values=clinical[:,column];finite=values[np.isfinite(values)]
        if not len(finite):raise ValueError(f"clinical variable {clinical_columns[column]} has no observed training values")
        if kind in {"nominal","binary"}:
            unique,counts=np.unique(finite,return_counts=True);fill=unique[np.argmax(counts)]
            clinical_mean[column]=fill;normalized[:,column]=np.where(np.isfinite(values),values,fill)
        else:
            mean=float(finite.mean());std=float(finite.std());clinical_mean[column]=mean;clinical_std[column]=std
            normalized[:,column]=(np.where(np.isfinite(values),values,mean)-mean)/(std+1e-8)
    clinical=normalized

    def clinical_estimate(index):
        return association_views(clinical[index],clinical[index],k,x_types=clinical_types,y_types=clinical_types)

    clinical_result = bootstrap_stability(clinical_estimate,n,seed,resamples,off_diagonal=True)
    a_rel = clinical_result.aggregate

    def projection_estimate(index):
        return association_views(z[index],clinical[index],k,x_types=["continuous"]*channels,y_types=clinical_types)

    projection_result = bootstrap_stability(projection_estimate,n,seed+104729,resamples)
    r = projection_result.aggregate
    r = r - r.max(1, keepdims=True); projection = np.exp(r); projection /= projection.sum(1, keepdims=True)
    prior = projection @ a_rel @ projection.T
    metadata = {"training_samples":int(n),"visual_channels":int(channels),"clinical_variables":int(variables),
                "clinical_columns":clinical_columns,"clinical_types":clinical_types,"clinical_matrix_shape":[int(n),int(variables)],
                "clinical_matrix_sha256":hashlib.sha256(np.ascontiguousarray(clinical,dtype=np.float64).tobytes()).hexdigest(),
                "clinical_normalization_mean":clinical_mean.tolist(),"clinical_normalization_std":clinical_std.tolist(),
                "A_rel":a_rel.tolist(),"Pi":projection.tolist(),
                "clinical_weights": clinical_result.weights.tolist(), "clinical_bootstraps": clinical_result.iterations,
                "projection_weights": projection_result.weights.tolist(), "projection_bootstraps": projection_result.iterations,
                "seed":int(seed),"nmi_neighbors":int(k)}
    return torch.from_numpy(prior).float(), metadata
