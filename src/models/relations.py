from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import hashlib

import numpy as np
import torch
from scipy.spatial import cKDTree, distance_matrix
from scipy.special import digamma
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.feature_selection import mutual_info_regression
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


def association_views(x: np.ndarray, y: np.ndarray, k: int = 5, chunk: int = 128) -> np.ndarray:
    """Matrix form of the four pairwise views, returning [x_dim,y_dim,4]."""
    x=np.nan_to_num(np.asarray(x,float));y=np.nan_to_num(np.asarray(y,float));n,p=x.shape;m=y.shape[1]
    def corr(a,b):
        a=(a-a.mean(0))/(a.std(0)+1e-12);b=(b-b.mean(0))/(b.std(0)+1e-12);return np.abs(a.T@b/max(n-1,1)).clip(0,1)
    pearson=corr(x,y);spearman=corr(rankdata(x,axis=0),rankdata(y,axis=0));dcor=np.zeros((p,m))
    for start in range(0,p,chunk):
        xb=x[:,start:start+chunk];dx=np.abs(xb[:,None,:]-xb[None,:,:]);dx=dx-dx.mean(0,keepdims=True)-dx.mean(1,keepdims=True)+dx.mean((0,1),keepdims=True)
        vx=np.mean(dx*dx,axis=(0,1))
        for j in range(m):
            dy=np.abs(y[:,None,j]-y[None,:,j]);dy=dy-dy.mean(0)-dy.mean(1)[:,None]+dy.mean();vy=np.mean(dy*dy)
            cov=np.mean(dx*dy[:,:,None],axis=(0,1));dcor[start:start+len(xb.T),j]=np.sqrt(np.maximum(cov,0)/np.sqrt(np.maximum(vx*vy,1e-24)))
    nmi=np.zeros((p,m));hx=np.zeros(p)
    for start in range(0,p,chunk):
        xb=x[:,start:start+chunk];dx=np.abs(xb[:,None,:]-xb[None,:,:]);eps=np.partition(dx,k,axis=1)[:,k,:].clip(1e-12)
        hx[start:start+xb.shape[1]]=digamma(n)-digamma(k)+np.log(2.0)+np.log(eps).mean(0)
    for j in range(m):
        if np.ptp(y[:,j])==0:continue
        yj=y[:,j];dy=np.abs(yj[:,None]-yj[None,:]);hy=_knn_entropy(yj,min(k,n-1));mi=np.zeros(p)
        for start in range(0,p,chunk):
            xb=x[:,start:start+chunk];dx=np.abs(xb[:,None,:]-xb[None,:,:]);joint=np.maximum(dx,dy[:,:,None]);eps=np.partition(joint,k,axis=1)[:,k,:]
            nx=(dx<eps[:,None,:]-1e-12).sum(1)-1;ny=(dy[:,:,None]<eps[:,None,:]-1e-12).sum(1)-1
            mi[start:start+xb.shape[1]]=np.maximum(0,digamma(n)+digamma(k)-np.mean(digamma(nx+1)+digamma(ny+1),axis=0))
        nmi[:,j]=np.clip(mi/np.sqrt(np.maximum(np.abs(hx*hy),1e-12)),0,1)
    return np.stack((pearson,spearman,dcor,nmi),-1)


@dataclass
class BootstrapResult:
    aggregate: np.ndarray
    weights: np.ndarray
    iterations: int


def adaptive_bootstrap(estimator: Callable[[np.ndarray], np.ndarray], n: int, seed: int,
                       initial: int = 1000, increment: int = 1000, maximum: int = 1000,
                       confidence: float = 0.95, off_diagonal: bool = False) -> BootstrapResult:
    rng = np.random.default_rng(seed); estimates: list[np.ndarray] = []; previous = None
    z = 1.959963984540054 if confidence == 0.95 else 2.5758293035489004
    target = initial
    while True:
        needed=target-len(estimates)
        if needed:
            indices=[rng.integers(0,n,n) for _ in range(needed)]
            estimates.extend(Parallel(n_jobs=min(8,needed),prefer='threads')(delayed(estimator)(index) for index in indices))
        stack = np.stack(estimates)
        # Variance of each relation view across bootstrap replicates.
        entries=stack.reshape(len(stack),-1,stack.shape[-1])
        if off_diagonal:
            if stack.shape[1]!=stack.shape[2]:raise ValueError('off-diagonal bootstrap requires square relation matrices')
            mask=~np.eye(stack.shape[1],dtype=bool).reshape(-1);entries=entries[:,mask]
            if not entries.shape[1]:raise ValueError('at least two clinical variables are required')
        per_view = entries.var(0, ddof=1).mean(0)
        weights = np.exp(-per_view - np.max(-per_view)); weights /= weights.sum()
        aggregate = np.tensordot(stack.mean(0), weights, axes=([-1], [0]))
        weight_samples = np.exp(-entries.var(1))
        weight_samples /= weight_samples.sum(1, keepdims=True)
        uncertainty = z * weight_samples.std(0, ddof=1) / np.sqrt(len(stack))
        if previous is not None and np.all(np.abs(weights - previous) <= uncertainty): break
        if target >= maximum: break
        previous = weights.copy(); target = min(maximum, target + increment)
    return BootstrapResult(aggregate, weights, len(estimates))


def build_relation_prior(features: torch.Tensor, clinical: np.ndarray, *, seed: int, k: int = 5,
                         initial: int = 1000, increment: int = 1000, maximum: int = 1000,
                         confidence: float = 0.95,
                         clinical_columns: list[str] | None = None) -> tuple[torch.Tensor, dict]:
    """Training-only multi-view clinical prior projected into visual channel space."""
    z = features.detach().float().cpu().numpy(); clinical = np.asarray(clinical, float)
    n, channels = z.shape; variables = clinical.shape[1]
    clinical_columns=list(clinical_columns or [f"clinical_{i}" for i in range(variables)])
    if len(clinical_columns)!=variables:raise ValueError('clinical column count does not match C_clin')
    clinical_mean=np.nanmean(clinical,0);clinical_std=np.nanstd(clinical,0)
    clinical = (clinical - clinical_mean) / (clinical_std + 1e-8)
    clinical = np.nan_to_num(clinical)

    def clinical_estimate(index):
        return association_views(clinical[index],clinical[index],k)

    clinical_result = adaptive_bootstrap(clinical_estimate, n, seed, initial, increment, maximum, confidence,off_diagonal=True)
    a_rel = clinical_result.aggregate

    def projection_estimate(index):
        return association_views(z[index],clinical[index],k)

    projection_result = adaptive_bootstrap(projection_estimate, n, seed + 104729, initial, increment, maximum, confidence)
    r = projection_result.aggregate
    r = r - r.max(1, keepdims=True); projection = np.exp(r); projection /= projection.sum(1, keepdims=True)
    prior = projection @ a_rel @ projection.T
    metadata = {"training_samples":int(n),"visual_channels":int(channels),"clinical_variables":int(variables),
                "clinical_columns":clinical_columns,"clinical_matrix_shape":[int(n),int(variables)],
                "clinical_matrix_sha256":hashlib.sha256(np.ascontiguousarray(clinical,dtype=np.float64).tobytes()).hexdigest(),
                "clinical_normalization_mean":clinical_mean.tolist(),"clinical_normalization_std":clinical_std.tolist(),
                "A_rel":a_rel.tolist(),"Pi":projection.tolist(),
                "clinical_weights": clinical_result.weights.tolist(), "clinical_bootstraps": clinical_result.iterations,
                "projection_weights": projection_result.weights.tolist(), "projection_bootstraps": projection_result.iterations,
                "seed":int(seed),"nmi_neighbors":int(k)}
    return torch.from_numpy(prior).float(), metadata
