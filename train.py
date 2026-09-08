#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from itertools import cycle
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.config import get_task,load_config
from src.data.dataset import CCMManifestDataset
from src.models import UnifiedCausalCCM
from src.models.relations import build_relation_prior
from src.models.screening import screen_channels
from src.utils.huggingface import resolve_cached_model
from src.utils.io import set_seed

SPECS=[('task1',0,None),('task2',1,None),('task3',2,[1,2,3,4]),('task3',3,[0]),('task4',4,None),('task5',5,None)]

def make_loaders(cfg,task,path,tokenizer):
    frame=pd.read_csv(path);result={}
    for split in ('train','validation','test'):
        subset=frame[frame.split.eq(split)].reset_index(drop=True)
        dataset=CCMManifestDataset(subset,task,tokenizer,int(cfg['data']['input_resolution']),max_length=int(cfg['model']['max_sequence_length']),clinical_missingness=None if split=='train' else float(cfg['model']['clinical_missingness_eval']),excluded_clinical_fields=set(task.get('clinical_exclude',[])))
        result[split]=DataLoader(dataset,batch_size=int(cfg['training']['batch_size']),shuffle=split=='train',num_workers=int(cfg['data']['num_workers']),pin_memory=True)
    return result
def select(tensor,indices): return tensor if indices is None else tensor[:,indices]
def prediction_loss(pred,target,mask,kind):
    if kind=='classification':
        return F.binary_cross_entropy_with_logits(pred[:,0],target.float()) if pred.shape[1]==1 else F.cross_entropy(pred,target)
    valid=mask.bool();return (pred[valid]-target[valid]).square().mean()
def to_device(batch,device): return {k:v.to(device) for k,v in batch.items() if torch.is_tensor(v)}

def balanced_epoch(model,entries,optimizer,device,stage,cfg):
    training=optimizer is not None;model.train(training);iterators=[cycle(x['loaders']['train' if training else 'validation']) for x in entries]
    steps=max(len(x['loaders']['train' if training else 'validation']) for x in entries);losses=[]
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for _ in range(steps):
            task_losses=[]
            if training: optimizer.zero_grad(set_to_none=True)
            for entry,it in zip(entries,iterators):
                b=to_device(next(it),device);target=select(b['target'],entry['indices']);mask=select(b['target_mask'],entry['indices'])
                out=model(b['image'],b['input_ids'],b['attention_mask'],entry['identity'],b['clinical_available'],stage)
                value=prediction_loss(out['prediction'],target,mask,entry['task']['kind'])
                if stage==1:value=value+float(cfg['model']['structural_loss_weight'])*out['prior_loss']
                else:value=value+float(cfg['model']['hyper_loss_weight'])*out['hyper_loss']
                task_losses.append(float(value.detach()))
                if training: (value/len(entries)).backward()
            if training: optimizer.step()
            losses.append(float(np.mean(task_losses))) # exactly 1/6 for every task identity
    return float(np.mean(losses))

@torch.no_grad()
def collect_statistics(model,entry,device):
    model.eval();gap=[];descriptors=[];targets=[];nuisance=[];patients=[]
    for batch in entry['loaders']['train']:
        b=to_device(batch,device);feature=model.visual_encoder(b['image']);gap.append(F.adaptive_avg_pool2d(feature,1).flatten(1).cpu())
        descriptors.append(torch.stack((feature.mean((2,3)),feature.amax((2,3))),-1).cpu())
        targets.append(select(b['target'],entry['indices']).cpu());structured=b['clinical_structured']
        nuisance.append(structured.cpu() if structured.shape[1] else torch.zeros(len(structured),1))
        patients.extend(batch['patient_id'])
    gap=torch.cat(gap);descriptors=torch.cat(descriptors);targets=torch.cat(targets);nuisance=torch.cat(nuisance)
    # Relation estimation is patient-level so repeated images do not multiply a
    # patient's clinical evidence; image-level samples remain unchanged in SGD.
    unique={name:i for i,name in enumerate(dict.fromkeys(patients))};group=torch.tensor([unique[x] for x in patients]);count=torch.bincount(group,minlength=len(unique)).float()
    def mean_by_patient(value):
        out=torch.zeros((len(unique),)+value.shape[1:],dtype=value.dtype);out.index_add_(0,group,value);return out/count.view((-1,)+(1,)*(value.ndim-1))
    gap=mean_by_patient(gap);descriptors=mean_by_patient(descriptors);nuisance=mean_by_patient(nuisance)
    targets=mean_by_patient(targets.float())
    if entry['task']['kind']=='classification': targets=targets.round().long()
    return gap,descriptors,targets,nuisance

def construct_priors(model,entries,device,cfg,seed,out):
    audit={}
    for entry in entries:
        identity=entry['identity'];gap,descriptors,target,nuisance=collect_statistics(model,entry,device)
        if nuisance.shape[1] == 0 or bool((nuisance.std(0) == 0).all()):
            audit[str(identity)]={'status':'not_available','reason':'no varying non-target clinical variables'}
            continue
        prior,relation_audit=build_relation_prior(gap,nuisance.numpy(),seed=seed+identity*1009,k=int(cfg['model']['nmi_neighbors']),initial=int(cfg['model']['relation_bootstrap_initial']),increment=int(cfg['model']['relation_bootstrap_increment']),maximum=int(cfg['model']['relation_bootstrap_max']),confidence=float(cfg['model']['relation_bootstrap_confidence']))
        model.set_prior(identity,prior)
        audit[str(identity)]=relation_audit
    (out/'relation_prior_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
def screen_all(model,entries,device,cfg,seed,out):
    audit={}
    for entry in entries:
        identity=entry['identity'];_,descriptors,target,nuisance=collect_statistics(model,entry,device)
        full_n=len(descriptors);maximum=int(cfg['model'].get('screening_max_patients',128))
        if full_n>maximum:
            generator=torch.Generator().manual_seed(seed+identity*3037)
            chosen=torch.randperm(full_n,generator=generator)[:maximum]
            descriptors,target,nuisance=descriptors[chosen],target[chosen],nuisance[chosen]
        selected,details=screen_channels(descriptors.to(device),target.to(device),nuisance.to(device),entry['task']['kind']=='classification',float(cfg['model']['retained_channel_ratio']),float(cfg['model']['screening_fdr']),seed+identity*2029,permutation_min=int(cfg['model']['permutation_min']),permutation_max=int(cfg['model']['permutation_max']),permutation_confidence=float(cfg['model']['permutation_confidence']),gcv_min=float(cfg['model']['kci_gcv_min']),gcv_max=float(cfg['model']['kci_gcv_max']),gcv_candidates=int(cfg['model']['kci_gcv_candidates']))
        model.set_channels(identity,selected);audit[str(identity)]={'screening':details,'selected_channels':selected.cpu().tolist(),'available_patients':full_n,'screening_patients':len(descriptors),'sampling':'all' if full_n<=maximum else 'deterministic_without_replacement'}
    (out/'channel_screening_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

def save(model,cfg,history,path,epoch):torch.save({'model_state':model.state_dict(),'config':cfg,'history':history,'global_epoch':epoch},path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'));parser.add_argument('--manifests-dir',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('checkpoints/unified_six_task'));parser.add_argument('--seed',type=int,default=3407);parser.add_argument('--resume-stage1',type=Path);args=parser.parse_args()
    cfg=load_config(args.config);set_seed(args.seed,True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for training')
    device=torch.device('cuda');tokenizer=AutoTokenizer.from_pretrained(resolve_cached_model(cfg['model']['clinical_encoder']));model=UnifiedCausalCCM(cfg['model']).to(device);args.output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for task_id,identity,indices in SPECS:
        task=get_task(cfg,task_id);entries.append({'task':task,'identity':identity,'indices':indices,'loaders':make_loaders(cfg,task,args.manifests_dir/f'{task_id}.csv',tokenizer)})
    history=[];epoch=0;optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
    warmup=int(cfg['training']['visual_warmup_epochs']);stage1=int(cfg['training']['stage1_epochs'])
    if args.resume_stage1:
        checkpoint=torch.load(args.resume_stage1,map_location=device);model.load_state_dict(checkpoint['model_state']);history=checkpoint.get('history',[]);epoch=int(checkpoint.get('global_epoch',stage1))
    for local in range(1,warmup+1) if not args.resume_stage1 else ():
        epoch+=1;tr=balanced_epoch(model,entries,optimizer,device,1,cfg);va=balanced_epoch(model,entries,None,device,1,cfg);history.append({'epoch':epoch,'stage':'warmup','train_loss':tr,'validation_loss':va})
        print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,args.output/f'epoch_{epoch:04d}.pt',epoch)
    if not args.resume_stage1:
        save(model,cfg,history,args.output/'stage1_warmup_complete.pt',epoch)
        construct_priors(model,entries,device,cfg,args.seed,args.output)
    for local in range(warmup+1,stage1+1) if not args.resume_stage1 else ():
        epoch+=1;tr=balanced_epoch(model,entries,optimizer,device,1,cfg);va=balanced_epoch(model,entries,None,device,1,cfg);history.append({'epoch':epoch,'stage':'prior_alignment','train_loss':tr,'validation_loss':va})
        print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,args.output/f'epoch_{epoch:04d}.pt',epoch)
    screen_all(model,entries,device,cfg,args.seed,args.output)
    for p in model.visual_encoder.parameters():p.requires_grad=False
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']));best=float('inf');stale=0
    for local in range(1,int(cfg['training']['max_epochs'])-stage1+1):
        epoch+=1;tr=balanced_epoch(model,entries,optimizer,device,2,cfg);va=balanced_epoch(model,entries,None,device,2,cfg);history.append({'epoch':epoch,'stage':'semantic_hypernetwork','train_loss':tr,'validation_loss':va});print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,args.output/f'epoch_{epoch:04d}.pt',epoch)
        if va<best-float(cfg['training']['early_stopping']['min_delta']):best=va;stale=0;save(model,cfg,history,args.output/'best_model.pt',epoch)
        else:stale+=1
        if local>=int(cfg['training']['early_stopping']['min_epochs']) and stale>=int(cfg['training']['early_stopping']['patience']):break
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
if __name__=='__main__':main()
