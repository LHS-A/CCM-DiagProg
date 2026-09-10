#!/usr/bin/env python
from __future__ import annotations
import argparse,json,math,hashlib
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
IDENTITY_NAMES=('ocular_diag','systemic_diag','ocular_reg','hba1c_reg','short_term','long_term')

def task_setting(value,identity):
    if isinstance(value,dict):
        name=IDENTITY_NAMES[identity]
        if name not in value:raise KeyError(f'missing task-specific setting for {name}')
        return value[name]
    return value

def task_exclusions(task,identity):
    """Resolve the paper's target/future/proxy mask for one task identity."""
    configured=task.get('clinical_exclude_by_identity',{})
    return set(configured.get(IDENTITY_NAMES[identity],task.get('clinical_exclude',[])))

def relation_fields(frame,task,identity):
    fields=[x for x in frame.columns if x.startswith('clinical_') and x!='clinical_text']
    if task['kind']=='classification':targets=['label']
    elif identity==2:targets=['CFS','TBUT','SIT','OSDI']
    elif identity==3:targets=['HbA1c']
    else:targets=list(task['targets'])
    return list(dict.fromkeys(fields+targets))

def validate_patient_partition(frame,path):
    groups={name:set(frame.loc[frame.split.eq(name),'patient_id'].astype(str)) for name in ('train','validation','test')}
    overlaps={(a,b):sorted(groups[a]&groups[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))}
    bad={f'{a}/{b}':ids[:10] for (a,b),ids in overlaps.items() if ids}
    if bad:raise ValueError(f'patient leakage in {path}: {bad}')

def make_loaders(cfg,task,path,tokenizer,identity):
    frame=pd.read_csv(path);allowed={'train','validation','test'};unknown=set(frame.split.dropna().unique())-allowed
    if unknown:raise ValueError(f'unsupported split values in {path}: {sorted(unknown)}')
    validate_patient_partition(frame,path)
    result={}
    for split in ('train','validation','test'):
        subset=frame[frame.split.eq(split)].reset_index(drop=True)
        if split=='train' and subset.empty:raise ValueError(f'{path} has no training partition')
        dataset=CCMManifestDataset(subset,task,tokenizer,int(cfg['data']['input_resolution']),max_length=int(cfg['model']['max_sequence_length']),clinical_missingness=None if split=='train' else float(cfg['model']['clinical_missingness_eval']),excluded_clinical_fields=task_exclusions(task,identity),partition=split,relation_clinical_fields=relation_fields(frame,task,identity))
        result[split]=DataLoader(dataset,batch_size=int(cfg['training']['batch_size']),shuffle=split=='train',num_workers=int(cfg['data']['num_workers']),pin_memory=True)
    return result
def select(tensor,indices): return tensor if indices is None else tensor[:,indices]
def prediction_loss(pred,target,mask,kind):
    if kind=='classification':
        return F.binary_cross_entropy_with_logits(pred[:,0],target.float()) if pred.shape[1]==1 else F.cross_entropy(pred,target)
    valid=mask.bool();return (pred[valid]-target[valid]).square().mean()
def to_device(batch,device): return {k:v.to(device) for k,v in batch.items() if torch.is_tensor(v)}

def balanced_epoch(model,entries,optimizer,device,stage,cfg):
    training=optimizer is not None;model.train(training)
    if stage==2:
        model.visual_encoder.eval();model.auxiliary.eval()
    iterators=[cycle(x['loaders']['train' if training else 'validation']) for x in entries]
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
    model.eval();gap=[];descriptors=[];targets=[];target_masks=[];nuisance=[];relations=[];patients=[]
    for batch in entry['loaders']['train']:
        if set(batch['data_partition'])!={'train'}:raise RuntimeError(f"non-training data reached statistics for {IDENTITY_NAMES[entry['identity']]}")
        b=to_device(batch,device);feature=model.visual_encoder(b['image']);gap.append(F.adaptive_avg_pool2d(feature,1).flatten(1).cpu())
        descriptors.append(torch.stack((feature.mean((2,3)),feature.amax((2,3))),-1).cpu())
        targets.append(select(b['target'],entry['indices']).cpu());target_masks.append(select(b['target_mask'],entry['indices']).cpu());structured=b['clinical_structured']
        nuisance.append(structured.cpu() if structured.shape[1] else torch.zeros(len(structured),1))
        relations.append(b['clinical_relation'].cpu())
        patients.extend(batch['patient_id'])
    gap=torch.cat(gap);descriptors=torch.cat(descriptors);targets=torch.cat(targets);target_masks=torch.cat(target_masks);nuisance=torch.cat(nuisance);relations=torch.cat(relations)
    # Relation estimation is patient-level so repeated images do not multiply a
    # patient's clinical evidence; image-level samples remain unchanged in SGD.
    unique={name:i for i,name in enumerate(dict.fromkeys(patients))};group=torch.tensor([unique[x] for x in patients]);count=torch.bincount(group,minlength=len(unique)).float()
    def mean_by_patient(value):
        out=torch.zeros((len(unique),)+value.shape[1:],dtype=value.dtype);out.index_add_(0,group,value);return out/count.view((-1,)+(1,)*(value.ndim-1))
    gap=mean_by_patient(gap);descriptors=mean_by_patient(descriptors)
    def masked_mean_by_patient(value,valid):
        safe=torch.where(valid,value,torch.zeros_like(value));total=torch.zeros((len(unique),)+value.shape[1:],dtype=value.dtype);number=torch.zeros_like(total)
        total.index_add_(0,group,safe);number.index_add_(0,group,valid.to(value.dtype));return total/number.clamp_min(1),number>0
    if nuisance.shape[1]:
        nuisance,nuisance_valid=masked_mean_by_patient(nuisance,torch.isfinite(nuisance));nuisance=torch.where(nuisance_valid,nuisance,torch.full_like(nuisance,float('nan')))
        column_mean=torch.nanmean(nuisance,0);column_mean=torch.nan_to_num(column_mean);nuisance=torch.where(torch.isfinite(nuisance),nuisance,column_mean)
    else:nuisance=mean_by_patient(nuisance)
    relations,relation_valid=masked_mean_by_patient(relations,torch.isfinite(relations));relations=torch.where(relation_valid,relations,torch.full_like(relations,float('nan')))
    relation_mean=torch.nanmean(relations,0);relation_mean=torch.nan_to_num(relation_mean);relations=torch.where(torch.isfinite(relations),relations,relation_mean)
    patient_targets=[]
    for group_id in range(len(unique)):
        values=targets[group.eq(group_id)]
        if entry['task']['kind']=='classification' and not bool((values==values[0]).all()):raise ValueError(f'inconsistent labels for patient {list(unique)[group_id]}')
        patient_targets.append(values.float().mean(0))
    if entry['task']['kind']=='classification':
        targets=torch.stack(patient_targets).round().long();valid_patients=torch.ones(len(targets),dtype=torch.bool)
    else:
        targets,target_valid=masked_mean_by_patient(targets.float(),target_masks.bool());valid_patients=target_valid.reshape(len(targets),-1).all(1)
    names=list(unique);keep=torch.where(valid_patients)[0];names=[names[i] for i in keep.tolist()]
    return gap[keep],descriptors[keep],targets[keep],nuisance[keep],relations[keep],names

def construct_priors(model,entries,device,cfg,seed,out,fold_id):
    audit={}
    for entry in entries:
        identity=entry['identity'];gap,descriptors,target,nuisance,relation,patients=collect_statistics(model,entry,device)
        if nuisance.shape[1] == 0 or bool((nuisance.std(0) == 0).all()):
            raise RuntimeError(f'{IDENTITY_NAMES[identity]} has no varying training-only non-target clinical variables')
        clinical_columns=list(entry['loaders']['train'].dataset.clinical_relation_columns)
        if relation.shape[1]<2:raise RuntimeError(f'{IDENTITY_NAMES[identity]} requires at least two task-specific clinical supervision variables for A_rel')
        prior,relation_audit=build_relation_prior(gap,relation.numpy(),seed=seed+identity*1009,k=int(cfg['model']['nmi_neighbors']),initial=int(cfg['model']['relation_bootstrap_initial']),increment=int(cfg['model']['relation_bootstrap_increment']),maximum=int(cfg['model']['relation_bootstrap_max']),confidence=float(cfg['model']['relation_bootstrap_confidence']),clinical_columns=clinical_columns)
        model.set_prior(identity,prior)
        manifest=Path(entry['manifest_path']);provenance={'task_id':identity,'task':IDENTITY_NAMES[identity],'fold_id':str(fold_id),'cohort':entry['task']['dataset_dir'],'partition':'train','source_manifest':str(manifest),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'patient_ids':patients}
        state={**provenance,**relation_audit};audit[str(identity)]=state
        task_dir=out/'relation_priors'/IDENTITY_NAMES[identity];task_dir.mkdir(parents=True,exist_ok=True)
        (task_dir/'C_clin_metadata.json').write_text(json.dumps({k:state[k] for k in ('task_id','task','fold_id','cohort','partition','source_manifest','source_manifest_sha256','patient_ids','clinical_columns','clinical_matrix_shape','clinical_matrix_sha256','clinical_normalization_mean','clinical_normalization_std')},indent=2)+'\n')
        (task_dir/'A_rel.json').write_text(json.dumps(relation_audit['A_rel'])+'\n')
        torch.save(torch.as_tensor(relation_audit['Pi'],dtype=torch.float32),task_dir/'Pi.pt')
        torch.save(prior.cpu(),task_dir/'M_prior.pt')
        (task_dir/'relation_view_weights.json').write_text(json.dumps({'clinical_weights':relation_audit['clinical_weights'],'projection_weights':relation_audit['projection_weights'],'clinical_bootstraps':relation_audit['clinical_bootstraps'],'projection_bootstraps':relation_audit['projection_bootstraps']},indent=2)+'\n')
    (out/'relation_prior_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
def screen_all(model,entries,device,cfg,seed,out):
    audit={}
    for entry in entries:
        identity=entry['identity'];_,descriptors,target,nuisance,_,patients=collect_statistics(model,entry,device)
        full_n=len(descriptors);configured=cfg['model'].get('screening_max_patients');maximum=full_n if configured is None else int(configured)
        if full_n>maximum:
            generator=torch.Generator().manual_seed(seed+identity*3037)
            chosen=torch.randperm(full_n,generator=generator)[:maximum]
            descriptors,target,nuisance=descriptors[chosen],target[chosen],nuisance[chosen];patients=[patients[i] for i in chosen.tolist()]
        rho=float(task_setting(cfg['model']['retained_channel_ratio'],identity))
        selected,details=screen_channels(descriptors.to(device),target.to(device),nuisance.to(device),entry['task']['kind']=='classification',rho,float(cfg['model']['screening_fdr']),seed+identity*2029,permutation_min=int(cfg['model']['permutation_min']),permutation_max=int(cfg['model']['permutation_max']),permutation_confidence=float(cfg['model']['permutation_confidence']),gcv_min=float(cfg['model']['kci_gcv_min']),gcv_max=float(cfg['model']['kci_gcv_max']),gcv_candidates=int(cfg['model']['kci_gcv_candidates']))
        model.set_channels(identity,selected);audit[str(identity)]={'task':IDENTITY_NAMES[identity],'partition':'train','target_representation_shape':list(target.shape),'nuisance_shape':list(nuisance.shape),'rho_t':rho,'selected_channels':selected.cpu().tolist(),'selected_feature_dim':len(selected),'available_patients':full_n,'screening_patients':len(descriptors),'patient_ids':patients,'sampling':'all' if full_n<=maximum else 'deterministic_without_replacement','screening':details}
        task_dir=out/'screening'/IDENTITY_NAMES[identity];task_dir.mkdir(parents=True,exist_ok=True)
        files={'hsic_statistics':details['hsic_statistics'],'hsic_pvalues':details['hsic_p'],'hsic_rejected':details['hsic_rejected'],'kci_statistics':details['kci_statistics'],'kci_pvalues':details['kci_p'],'kci_rejected':details['kci_rejected'],'retention_ratio':rho,'selected_channels':selected.cpu().tolist()}
        for name,value in files.items():(task_dir/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    (out/'channel_screening_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

def save(model,cfg,history,path,epoch,fold_id):torch.save({'model_state':model.state_dict(),'config':cfg,'history':history,'global_epoch':epoch,'fold_id':str(fold_id),'task_identities':IDENTITY_NAMES},path)
def learning_rate(cfg,epoch):
    base=float(cfg['training']['learning_rate']);minimum=float(cfg['training']['min_learning_rate']);maximum=int(cfg['training']['max_epochs'])
    return minimum+(base-minimum)*.5*(1+math.cos(math.pi*min(epoch,maximum)/maximum))
def set_learning_rate(optimizer,value):
    for group in optimizer.param_groups:group['lr']=value
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'));parser.add_argument('--manifests-dir',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('checkpoints/unified_six_task'));parser.add_argument('--seed',type=int,default=3407);parser.add_argument('--fold-id',default='single_fold');parser.add_argument('--resume-stage1',type=Path);args=parser.parse_args()
    cfg=load_config(args.config);set_seed(args.seed,True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for training')
    device=torch.device('cuda');tokenizer=AutoTokenizer.from_pretrained(resolve_cached_model(cfg['model']['clinical_encoder']));model=UnifiedCausalCCM(cfg['model']).to(device);args.output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for task_id,identity,indices in SPECS:
        task=get_task(cfg,task_id);manifest=args.manifests_dir/f'{task_id}.csv';entries.append({'task':task,'identity':identity,'indices':indices,'manifest_path':manifest,'loaders':make_loaders(cfg,task,manifest,tokenizer,identity)})
    history=[];epoch=0;model.set_stage_trainability(1);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
    warmup=int(cfg['training']['visual_warmup_epochs']);stage1=int(cfg['training']['stage1_epochs'])
    if args.resume_stage1:
        checkpoint=torch.load(args.resume_stage1,map_location=device);model.load_state_dict(checkpoint['model_state']);model.set_stage_trainability(1);history=checkpoint.get('history',[]);epoch=int(checkpoint.get('global_epoch',0))
    for target_epoch in range(epoch+1,warmup+1):
        epoch=target_epoch;lr=learning_rate(cfg,epoch-1);set_learning_rate(optimizer,lr);tr=balanced_epoch(model,entries,optimizer,device,1,cfg);va=balanced_epoch(model,entries,None,device,1,cfg);history.append({'epoch':epoch,'stage':'warmup','learning_rate':lr,'train_loss':tr,'validation_loss':va})
        print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,args.output/f'epoch_{epoch:04d}.pt',epoch,args.fold_id)
    if epoch==warmup:save(model,cfg,history,args.output/'stage1_warmup_complete.pt',epoch,args.fold_id)
    if not all(bool(prior.prior_ready) for prior in model.structural_priors):
        if epoch<warmup:raise RuntimeError('structural prior construction requires completed visual warm-up')
        construct_priors(model,entries,device,cfg,args.seed,args.output,args.fold_id)
    for target_epoch in range(max(epoch+1,warmup+1),stage1+1):
        epoch=target_epoch;lr=learning_rate(cfg,epoch-1);set_learning_rate(optimizer,lr);tr=balanced_epoch(model,entries,optimizer,device,1,cfg);va=balanced_epoch(model,entries,None,device,1,cfg);history.append({'epoch':epoch,'stage':'prior_alignment','learning_rate':lr,'train_loss':tr,'validation_loss':va})
        print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,args.output/f'epoch_{epoch:04d}.pt',epoch,args.fold_id)
    screen_all(model,entries,device,cfg,args.seed,args.output)
    model.set_stage_trainability(2)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']));best=float('inf');stale=0
    for local in range(1,int(cfg['training']['max_epochs'])-stage1+1):
        epoch+=1;lr=learning_rate(cfg,epoch-1);set_learning_rate(optimizer,lr);tr=balanced_epoch(model,entries,optimizer,device,2,cfg);va=balanced_epoch(model,entries,None,device,2,cfg);history.append({'epoch':epoch,'stage':'semantic_hypernetwork','learning_rate':lr,'train_loss':tr,'validation_loss':va});print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,args.output/f'epoch_{epoch:04d}.pt',epoch,args.fold_id)
        if va<best-float(cfg['training']['early_stopping']['min_delta']):best=va;stale=0;save(model,cfg,history,args.output/'best_model.pt',epoch,args.fold_id)
        else:stale+=1
        if local>=int(cfg['training']['early_stopping']['min_epochs']) and stale>=int(cfg['training']['early_stopping']['patience']):break
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
if __name__=='__main__':main()
