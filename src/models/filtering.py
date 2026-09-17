from __future__ import annotations
import numpy as np
import torch

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
def _permutation_many(stat,null,resamples):
    """Deterministic fixed-count permutation p-values with +1 correction."""
    exceed=torch.zeros(len(stat),dtype=torch.long,device=stat.device);used=0
    while used<resamples:
        count=min(25,resamples-used);exceed+=(null(count)>=stat).sum(0);used+=count
    return (1+exceed).float()/(1+resamples),torch.full_like(exceed,resamples)

def top_rho_retention(hsic_scores,kci_scores,hsic_candidates,significant,retention_ratio,total_channels):
    """Retain exactly floor(rho*C) channels using the paper's two rankings."""
    budget=max(1,int(np.floor(total_channels*retention_ratio)))
    if significant.numel()!=total_channels:raise ValueError('significant mask must be defined over all visual channels')
    if len(hsic_candidates)<budget:
        raise RuntimeError(f'HSIC coarse candidate set has {len(hsic_candidates)} channels, fewer than fixed top-rho budget {budget}')
    selected=torch.where(significant)[0]
    if len(selected)>=budget:
        selected=selected[torch.topk(kci_scores[selected],budget).indices]
    else:
        hsic_only=hsic_candidates[~significant[hsic_candidates]];needed=budget-len(selected)
        if len(hsic_only)<needed:raise RuntimeError('HSIC-only candidates cannot complete the fixed top-rho budget')
        selected=torch.cat((selected,hsic_only[torch.topk(hsic_scores[hsic_only],needed).indices]))
    return selected.sort().values,budget

@torch.no_grad()
def filter_channels(descriptors,target,nuisance,categorical,retention_ratio,alpha,seed,
                    permutation_resamples=1000,
                    gcv_min=1e-6,gcv_max=1e1,gcv_candidates=50):
    """Paper coarse HSIC, fine KCI and fixed-budget top-rho filtering."""
    device=descriptors.device;n,channels,_=descriptors.shape
    raw_ky=_target_kernel(target,categorical);ky=_center(raw_ky);raw_kz,nuisance_bandwidth=_rbf_with_bandwidth(nuisance);kz=_center(raw_kz)
    candidates=torch.logspace(np.log10(gcv_min),np.log10(gcv_max),gcv_candidates,device=device); lam=gcv_regularization(kz,candidates)
    eye=torch.eye(n,device=device); residual=eye-kz@torch.linalg.solve(kz+lam*eye,eye); rng=torch.Generator(device=device).manual_seed(seed)
    x=descriptors.permute(1,0,2).float();dist=(x[:,:,None,:]-x[:,None,:,:]).square().sum(-1)
    flattened=dist.flatten(1).masked_fill(dist.flatten(1)<=0,float('nan'));bandwidth=torch.nanmedian(flattened,dim=1).values
    bandwidth=torch.nan_to_num(bandwidth,nan=1.0).clamp_min(1e-6);kx=_center_batch(torch.exp(-dist/(2*bandwidth[:,None,None])))
    hs=(kx*ky).sum((-2,-1))/max((n-1)**2,1)
    def hnull(count):
        permutations=torch.stack([torch.randperm(n,generator=rng,device=device) for _ in range(count)])
        pky=torch.stack([ky[p][:,p] for p in permutations]);return torch.einsum('cij,pij->pc',kx,pky)/max((n-1)**2,1)
    hp_t,hn_t=_permutation_many(hs,hnull,permutation_resamples)
    hsic_rejected=_bh(hp_t,alpha)
    # Sec. 2.1.3: only marginally supported channels form S_HSIC. KCI is
    # evaluated exclusively on this task-specific coarse candidate set.
    candidates_idx=torch.where(hsic_rejected)[0]
    budget=max(1,int(np.floor(channels*retention_ratio)))
    if len(candidates_idx)<budget:
        raise RuntimeError(f'HSIC coarse candidate set has {len(candidates_idx)} channels, fewer than fixed top-rho budget {budget}')
    candidate_kx=kx.index_select(0,candidates_idx)
    joint=_center_batch(candidate_kx*kz);rx=torch.einsum('ij,cjk,kl->cil',residual,joint,residual);ry=residual@ky@residual;candidate_ks=(rx*ry.T).sum((-2,-1))/n
    def knull(count):
        permutations=torch.stack([torch.randperm(n,generator=rng,device=device) for _ in range(count)])
        pry=torch.stack([residual@ky[p][:,p]@residual for p in permutations]);return torch.einsum('cij,pji->pc',rx,pry)/n
    kp_candidate,kn_candidate=_permutation_many(candidate_ks,knull,permutation_resamples)
    candidate_rejected=_bh(kp_candidate,alpha)
    kp_t=torch.full((channels,),float('nan'),device=device);kp_t[candidates_idx]=kp_candidate
    kn_t=torch.zeros(channels,dtype=torch.long,device=device);kn_t[candidates_idx]=kn_candidate
    ks=torch.full((channels,),float('-inf'),device=device);ks[candidates_idx]=candidate_ks
    kci_rejected=torch.zeros(channels,dtype=torch.bool,device=device);kci_rejected[candidates_idx]=candidate_rejected
    significant=hsic_rejected & kci_rejected
    # S_sig=S_HSIC intersection S_KCI. If it is short, only S_HSIC\S_sig is
    # eligible for HSIC-score backfilling; no out-of-pool fallback is allowed.
    indices,budget=top_rho_retention(hs,ks,candidates_idx,significant,retention_ratio,channels)
    ranking=torch.argsort(ks,descending=True)
    details={'lambda_kci':float(lam),'descriptor_bandwidths':bandwidth.cpu().tolist(),'nuisance_bandwidth':float(nuisance_bandwidth),
             'target_kernel':'label' if categorical else 'rbf_median_heuristic','hsic_statistics':hs.cpu().tolist(),'hsic_candidates':candidates_idx.cpu().tolist(),'kci_candidate_count':len(candidates_idx),'kci_statistics':ks.cpu().tolist(),
             'hsic_p':hp_t.cpu().tolist(),'kci_p':kp_t.cpu().tolist(),'hsic_rejected':hsic_rejected.cpu().tolist(),
             'kci_rejected':kci_rejected.cpu().tolist(),'significant_intersection':significant.cpu().tolist(),
             'conditional_ranking':ranking.cpu().tolist(),'hsic_permutations':hn_t.cpu().tolist(),'kci_permutations':kn_t.cpu().tolist(),
             'retention_ratio':float(retention_ratio),'retention_budget':budget,'selected_count':len(indices),'selected_channels':indices.sort().values.cpu().tolist()}
    return indices.sort().values,details

# Public terminology follows the manuscript. A compatibility alias is not
# retained because old checkpoints/code are explicitly outside this revision.
