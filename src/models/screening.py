from __future__ import annotations
import numpy as np
import torch
from scipy.stats import beta

def _rbf(x):
    d=torch.cdist(x.float(),x.float()).square(); p=d[d>0]; h=p.median().clamp_min(1e-6) if len(p) else d.new_tensor(1.0)
    return torch.exp(-d/(2*h))
def _center(k): return k-k.mean(0,keepdim=True)-k.mean(1,keepdim=True)+k.mean()
def _center_batch(k): return k-k.mean(-2,keepdim=True)-k.mean(-1,keepdim=True)+k.mean((-2,-1),keepdim=True)
def _target_kernel(y,categorical): return y[:,None].eq(y[None]).float() if categorical else _rbf(y.float().reshape(len(y),-1))
def _bh(p,alpha):
    order=torch.argsort(p); passed=p[order]<=alpha*torch.arange(1,len(p)+1,device=p.device)/len(p); out=torch.zeros_like(p,dtype=torch.bool)
    if passed.any(): out[order[:int(torch.where(passed)[0].max())+1]]=True
    return out
def gcv_regularization(k,candidates):
    n=len(k); eye=torch.eye(n,device=k.device,dtype=k.dtype); scores=[]
    for value in candidates:
        residual=eye-k@torch.linalg.solve(k+value*eye,eye)
        scores.append((residual@k).square().sum()/n/residual.diagonal().mean().square().clamp_min(1e-12))
    return candidates[torch.argmin(torch.stack(scores))]
def _sequential(stat,null,minimum,maximum,confidence,boundary):
    exceed=used=0; tail=(1-confidence)/2
    while used<maximum:
        count=min(25,maximum-used); values=null(count); exceed+=int((values>=stat).sum()); used+=count
        if used<minimum: continue
        lo=0 if exceed==0 else float(beta.ppf(tail,exceed,used-exceed+1)); hi=1 if exceed==used else float(beta.ppf(1-tail,exceed+1,used-exceed))
        if hi<boundary or lo>boundary: break
    return (1+exceed)/(1+used),used

def _sequential_many(stat,null,minimum,maximum,confidence,boundary):
    """The same sequential test as _sequential, evaluated for all channels."""
    channels=len(stat); exceed=torch.zeros(channels,dtype=torch.long,device=stat.device); used=torch.zeros_like(exceed); active=torch.ones(channels,dtype=torch.bool,device=stat.device); tail=(1-confidence)/2
    while active.any():
        count=min(25,maximum-int(used[active].min()))
        values=null(count)
        exceed[active]+=(values[:,active]>=stat[active]).sum(0);used[active]+=count
        eligible=active&(used>=minimum)
        if eligible.any():
            e=exceed[eligible].cpu().numpy();u=used[eligible].cpu().numpy()
            lo=np.where(e==0,0.0,beta.ppf(tail,e,u-e+1));hi=np.where(e==u,1.0,beta.ppf(1-tail,e+1,u-e))
            stop=torch.as_tensor((hi<boundary)|(lo>boundary),device=stat.device)
            positions=torch.where(eligible)[0];active[positions[stop]]=False
        active&=used<maximum
    return ((1+exceed).float()/(1+used).float()),used

@torch.no_grad()
def screen_channels(descriptors,target,nuisance,categorical,retention_ratio,alpha,seed,
                    permutation_min=200,permutation_max=5000,permutation_confidence=.99,
                    gcv_min=1e-6,gcv_max=1e1,gcv_candidates=50):
    """Sequential-permutation HSIC then KCI screening with BH and GCV."""
    device=descriptors.device; n,channels,_=descriptors.shape; ky=_center(_target_kernel(target,categorical)); kz=_center(_rbf(nuisance))
    candidates=torch.logspace(np.log10(gcv_min),np.log10(gcv_max),gcv_candidates,device=device); lam=gcv_regularization(kz,candidates)
    eye=torch.eye(n,device=device); residual=eye-kz@torch.linalg.solve(kz+lam*eye,eye); rng=torch.Generator(device=device).manual_seed(seed); boundary=alpha/channels
    x=descriptors.permute(1,0,2).float();dist=(x[:,:,None,:]-x[:,None,:,:]).square().sum(-1);positive=dist[dist>0];bandwidth=positive.median().clamp_min(1e-6) if len(positive) else dist.new_tensor(1.0);kx=_center_batch(torch.exp(-dist/(2*bandwidth)))
    hs=(kx*ky).sum((-2,-1))/max((n-1)**2,1)
    def hnull(count):
        permutations=torch.stack([torch.randperm(n,generator=rng,device=device) for _ in range(count)])
        pky=torch.stack([ky[p][:,p] for p in permutations]);return torch.einsum('cij,pij->pc',kx,pky)/max((n-1)**2,1)
    hp_t,hn_t=_sequential_many(hs,hnull,permutation_min,permutation_max,permutation_confidence,boundary)
    joint=_center_batch(kx*kz);rx=torch.einsum('ij,cjk,kl->cil',residual,joint,residual);ry=residual@ky@residual;ks=(rx*ry.T).sum((-2,-1))/n
    def knull(count):
        permutations=torch.stack([torch.randperm(n,generator=rng,device=device) for _ in range(count)])
        pry=torch.stack([residual@ky[p][:,p]@residual for p in permutations]);return torch.einsum('cij,pji->pc',rx,pry)/n
    kp_t,kn_t=_sequential_many(ks,knull,permutation_min,permutation_max,permutation_confidence,boundary)
    significant=_bh(hp_t,alpha)&_bh(kp_t,alpha); indices=torch.where(significant)[0]
    budget=max(1,round(channels*retention_ratio));ks_t=ks;fallback=False
    # Finite sequential Monte-Carlo tests can yield an empty intersection after
    # BH correction.  The Methods do not define this degenerate boundary case;
    # retain the prescribed budget by the joint HSIC/KCI evidence ordering so
    # the architecture remains executable, and expose the decision in audit.
    if not len(indices):
        fallback=True
        joint_score=torch.maximum(hp_t,kp_t)
        indices=torch.topk(joint_score,min(budget,channels),largest=False).indices
    if len(indices)>budget: indices=indices[torch.topk(ks_t[indices],budget).indices]
    return indices.sort().values,{'lambda_kci':float(lam),'hsic_p':hp_t.cpu().tolist(),'kci_p':kp_t.cpu().tolist(),'hsic_permutations':hn_t.cpu().tolist(),'kci_permutations':kn_t.cpu().tolist(),'empty_intersection_fallback':fallback,'retention_budget':budget}
