from __future__ import annotations
import numpy as np
import torch
from scipy.stats import beta

def _rbf(x):
    kernel,_=_rbf_with_bandwidth(x);return kernel
def _rbf_with_bandwidth(x):
    d=torch.cdist(x.float(),x.float()).square();p=d[d>0];h=p.median().clamp_min(1e-6) if len(p) else d.new_tensor(1.0)
    return torch.exp(-d/(2*h)),h
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
                    permutation_min=200,permutation_max=10000,permutation_confidence=.99,
                    gcv_min=1e-6,gcv_max=1e1,gcv_candidates=50):
    """Sequential-permutation HSIC then KCI screening with BH and GCV."""
    device=descriptors.device;n,channels,_=descriptors.shape
    raw_ky=_target_kernel(target,categorical);ky=_center(raw_ky);raw_kz,nuisance_bandwidth=_rbf_with_bandwidth(nuisance);kz=_center(raw_kz)
    candidates=torch.logspace(np.log10(gcv_min),np.log10(gcv_max),gcv_candidates,device=device); lam=gcv_regularization(kz,candidates)
    eye=torch.eye(n,device=device); residual=eye-kz@torch.linalg.solve(kz+lam*eye,eye); rng=torch.Generator(device=device).manual_seed(seed); boundary=alpha/channels
    x=descriptors.permute(1,0,2).float();dist=(x[:,:,None,:]-x[:,None,:,:]).square().sum(-1)
    flattened=dist.flatten(1).masked_fill(dist.flatten(1)<=0,float('nan'));bandwidth=torch.nanmedian(flattened,dim=1).values
    bandwidth=torch.nan_to_num(bandwidth,nan=1.0).clamp_min(1e-6);kx=_center_batch(torch.exp(-dist/(2*bandwidth[:,None,None])))
    hs=(kx*ky).sum((-2,-1))/max((n-1)**2,1)
    def hnull(count):
        permutations=torch.stack([torch.randperm(n,generator=rng,device=device) for _ in range(count)])
        pky=torch.stack([ky[p][:,p] for p in permutations]);return torch.einsum('cij,pij->pc',kx,pky)/max((n-1)**2,1)
    hp_t,hn_t=_sequential_many(hs,hnull,permutation_min,permutation_max,permutation_confidence,boundary)
    hsic_rejected=_bh(hp_t,alpha);candidates_idx=torch.where(hsic_rejected)[0]
    if not len(candidates_idx):
        raise RuntimeError('HSIC produced no target-relevant candidate channels')
    # Paper order is strict: KCI is evaluated and multiplicity-corrected only
    # inside the HSIC candidate family, not over all visual channels.
    candidate_kx=kx.index_select(0,candidates_idx)
    joint=_center_batch(candidate_kx*kz);rx=torch.einsum('ij,cjk,kl->cil',residual,joint,residual);ry=residual@ky@residual;candidate_ks=(rx*ry.T).sum((-2,-1))/n
    def knull(count):
        permutations=torch.stack([torch.randperm(n,generator=rng,device=device) for _ in range(count)])
        pry=torch.stack([residual@ky[p][:,p]@residual for p in permutations]);return torch.einsum('cij,pji->pc',rx,pry)/n
    kp_candidate,kn_candidate=_sequential_many(candidate_ks,knull,permutation_min,permutation_max,permutation_confidence,alpha/len(candidates_idx))
    candidate_rejected=_bh(kp_candidate,alpha)
    kp_t=torch.full((channels,),float('nan'),device=device);kp_t[candidates_idx]=kp_candidate
    kn_t=torch.zeros(channels,dtype=torch.long,device=device);kn_t[candidates_idx]=kn_candidate
    ks=torch.full((channels,),float('-inf'),device=device);ks[candidates_idx]=candidate_ks
    kci_rejected=torch.zeros(channels,dtype=torch.bool,device=device);kci_rejected[candidates_idx]=candidate_rejected
    significant=hsic_rejected&kci_rejected;indices=torch.where(significant)[0]
    budget=max(1,round(channels*retention_ratio));ks_t=ks
    if not len(indices):raise RuntimeError('HSIC/KCI intersection is empty for this task; no non-significant fallback is permitted')
    if len(indices)>budget: indices=indices[torch.topk(ks_t[indices],budget).indices]
    ranking=torch.argsort(ks_t,descending=True)
    details={'lambda_kci':float(lam),'descriptor_bandwidths':bandwidth.cpu().tolist(),'nuisance_bandwidth':float(nuisance_bandwidth),
             'target_kernel':'label' if categorical else 'rbf_median_heuristic','hsic_statistics':hs.cpu().tolist(),'hsic_candidates':candidates_idx.cpu().tolist(),'kci_candidate_count':len(candidates_idx),'kci_statistics':ks.cpu().tolist(),
             'hsic_p':hp_t.cpu().tolist(),'kci_p':kp_t.cpu().tolist(),'hsic_rejected':hsic_rejected.cpu().tolist(),
             'kci_rejected':kci_rejected.cpu().tolist(),'significant_intersection':significant.cpu().tolist(),
             'conditional_ranking':ranking.cpu().tolist(),'hsic_permutations':hn_t.cpu().tolist(),'kci_permutations':kn_t.cpu().tolist(),
             'retention_ratio':float(retention_ratio),'retention_budget':budget,'selected_count':len(indices),'selected_channels':indices.sort().values.cpu().tolist()}
    return indices.sort().values,details
