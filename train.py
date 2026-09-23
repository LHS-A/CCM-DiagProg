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
from src.data.patient import (PatientBalancedBatchSampler, patient_mean,
                              patient_targets, regression_normalizer)
from src.models import UnifiedCausalCCM
from src.models.relations import build_relation_prior
from src.models.filtering import filter_channels
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
def task_available_fields(task,identity):
    configured=task.get('clinical_available_by_identity',{})
    return set(configured.get(IDENTITY_NAMES[identity],task.get('clinical_available_fields',[])))

def relation_fields(frame,task,identity):
    if task['kind']=='classification':targets=['label']
    elif identity==2:targets=['TBUT','CFS','SIT','OSDI']
    elif identity==3:targets=['HbA1c']
    else:targets=list(task['targets'])
    target_keys={x.casefold() for x in targets};allowed={x.casefold() for x in task_available_fields(task,identity)}
    fields=[]
    for column in frame.columns:
        if not column.startswith('clinical_') or column=='clinical_text':continue
        short=column[len('clinical_'):].casefold()
        if short in target_keys:continue
        if allowed and short not in allowed and column.casefold() not in allowed:continue
        fields.append(column)
    return list(dict.fromkeys(fields+targets))

def validate_patient_partition(frame,path):
    groups={name:set(frame.loc[frame.split.eq(name),'patient_id'].astype(str)) for name in ('train','validation','test')}
    overlaps={(a,b):sorted(groups[a]&groups[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))}
    bad={f'{a}/{b}':ids[:10] for (a,b),ids in overlaps.items() if ids}
    if bad:raise ValueError(f'patient leakage in {path}: {bad}')

def validate_cross_task_partitions(frames):
    """The same patient may never occupy different partitions across tasks."""
    assigned={}
    for path,frame in frames:
        for patient,partitions in frame.groupby(frame.patient_id.astype(str)).split:
            values=set(partitions.astype(str))
            if len(values)!=1:raise ValueError(f'patient leakage in {path}: {patient} has {sorted(values)}')
            split=next(iter(values))
            if patient in assigned and assigned[patient][0]!=split:
                raise ValueError(f'cross-task patient leakage: {patient} is {assigned[patient][0]} in {assigned[patient][1]} but {split} in {path}')
            assigned[patient]=(split,str(path))

def identity_batch_size(total,identity):
    """Allocate the paper's total image batch across six represented tasks."""
    if total < len(IDENTITY_NAMES):raise ValueError('total batch size must represent all six tasks')
    base,remainder=divmod(int(total),len(IDENTITY_NAMES))
    return base+(identity<remainder)

def make_loaders(cfg,task,path,tokenizer,identity,seed=0):
    frame=pd.read_csv(path);allowed={'train','validation','test'};unknown=set(frame.split.dropna().unique())-allowed
    if unknown:raise ValueError(f'unsupported split values in {path}: {sorted(unknown)}')
    validate_patient_partition(frame,path)
    result={}
    for split in ('train','validation','test'):
        subset=frame[frame.split.eq(split)].reset_index(drop=True)
        if split=='train' and subset.empty:raise ValueError(f'{path} has no training partition')
        dataset=CCMManifestDataset(subset,task,tokenizer,int(cfg['data']['input_resolution']),max_length=int(cfg['model']['max_sequence_length']),clinical_missingness=None if split=='train' else float(cfg['model']['clinical_missingness_eval']),excluded_clinical_fields=task_exclusions(task,identity),allowed_clinical_fields=task_available_fields(task,identity),partition=split,relation_clinical_fields=relation_fields(frame,task,identity))
        batch_size=identity_batch_size(int(cfg['training']['batch_size']),identity)
        options={'num_workers':int(cfg['data']['num_workers']),'pin_memory':True}
        if split=='train':
            sampler=PatientBalancedBatchSampler(subset.patient_id.astype(str).tolist(),batch_size,seed+identity*1009)
            result[split]=DataLoader(dataset,batch_sampler=sampler,**options)
            result['statistics']=DataLoader(dataset,batch_size=batch_size,shuffle=False,**options)
        else:result[split]=DataLoader(dataset,batch_size=batch_size,shuffle=False,**options)
    return result
def select(tensor,indices): return tensor if indices is None else tensor[:,indices]
def sample_prediction_loss(pred,target,mask,kind):
    if kind=='classification':
        return F.binary_cross_entropy_with_logits(pred[:,0],target.float(),reduction='none') if pred.shape[1]==1 else F.cross_entropy(pred,target,reduction='none')
    valid=mask.bool();squared=(pred-target).square()*valid
    return squared.sum(1)/valid.sum(1).clamp_min(1)

def patient_average(values,patient_ids):
    groups={}
    for index,patient in enumerate(patient_ids):groups.setdefault(str(patient),[]).append(index)
    return torch.stack([values[torch.as_tensor(index,device=values.device)].mean() for index in groups.values()]).mean()

def normalize_target(target,mask,entry):
    if entry['task']['kind']=='classification':return target
    mean=entry['target_mean'].to(target.device);std=entry['target_std'].to(target.device)
    return torch.where(mask.bool(),(target-mean)/std,target)
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
                batch=next(it);b=to_device(batch,device);target=select(b['target'],entry['indices']);mask=select(b['target_mask'],entry['indices'])
                target=normalize_target(target,mask,entry)
                out=model(b['image'],b['input_ids'],b['attention_mask'],entry['identity'],b['clinical_available'],stage)
                per_image=sample_prediction_loss(out['prediction'],target,mask,entry['task']['kind'])
                value=patient_average(per_image,batch['patient_id'])
                if stage==1:value=value+float(cfg['model']['structural_loss_weight'])*out['prior_loss']
                else:value=value+float(cfg['model']['hyper_loss_weight'])*patient_average(out['dynamic_penalty'],batch['patient_id'])
                task_losses.append(float(value.detach()))
                if training: (value/len(entries)).backward()
            if training: optimizer.step()
            losses.append(float(np.mean(task_losses))) # exactly 1/6 for every task identity
    return float(np.mean(losses))

@torch.no_grad()
def collect_statistics(model,entry,device):
    model.eval();gap=[];descriptors=[];targets=[];target_masks=[];nuisance=[];relations=[];patients=[]
    statistics_loader=entry['loaders']['statistics'] if 'statistics' in entry['loaders'] else entry['loaders']['train']
    for batch in statistics_loader:
        if set(batch['data_partition'])!={'train'}:raise RuntimeError(f"non-training data reached statistics for {IDENTITY_NAMES[entry['identity']]}")
        b=to_device(batch,device);feature=model.visual_encoder(b['image']);gap.append(F.adaptive_avg_pool2d(feature,1).flatten(1).cpu())
        descriptors.append(torch.stack((feature.mean((2,3)),feature.amax((2,3))),-1).cpu())
        targets.append(select(b['target'],entry['indices']).cpu());target_masks.append(select(b['target_mask'],entry['indices']).cpu());structured=b['clinical_structured']
        nuisance.append(structured.cpu() if structured.shape[1] else torch.zeros(len(structured),1))
        relations.append(b['clinical_relation'].cpu())
        patients.extend(batch['patient_id'])
    gap=torch.cat(gap);descriptors=torch.cat(descriptors);targets=torch.cat(targets);target_masks=torch.cat(target_masks);nuisance=torch.cat(nuisance);relations=torch.cat(relations)
    # Clinical relations, channel responses, GAP/GMP descriptors, targets and
    # nuisance variables all use exactly one observation per training patient.
    gap,names=patient_mean(gap,patients);descriptors,descriptor_names=patient_mean(descriptors,patients)
    nuisance,nuisance_names=patient_mean(nuisance,patients);relations,relation_names=patient_mean(relations,patients)
    targets,patient_masks,target_names=patient_targets(targets,target_masks,patients,entry['task']['kind']=='classification')
    if not (names==descriptor_names==nuisance_names==relation_names==target_names):raise RuntimeError('patient aggregation order mismatch')
    valid_samples=torch.ones(len(targets),dtype=torch.bool) if entry['task']['kind']=='classification' else patient_masks.reshape(len(targets),-1).all(1)
    if nuisance.shape[1]:
        column_mean=torch.nanmean(nuisance,0);column_mean=torch.nan_to_num(column_mean);nuisance=torch.where(torch.isfinite(nuisance),nuisance,column_mean)
        nuisance=nuisance[:,nuisance.std(0)>0]
    keep=torch.where(valid_samples)[0];kept_names=[names[i] for i in keep.tolist()]
    return gap[keep],descriptors[keep],targets[keep],nuisance[keep],relations[keep],kept_names

def construct_priors(model,entries,device,cfg,seed,out,fold_id):
    audit={}
    for entry in entries:
        identity=entry['identity'];gap,descriptors,target,nuisance,relation,patients=collect_statistics(model,entry,device)
        if nuisance.shape[1] == 0 or bool((nuisance.std(0) == 0).all()):
            raise RuntimeError(f'{IDENTITY_NAMES[identity]} has no varying training-only non-target clinical variables')
        clinical_columns=list(entry['loaders']['train'].dataset.clinical_relation_columns);clinical_types=list(entry['loaders']['train'].dataset.clinical_relation_types)
        keep=[]
        for column in range(relation.shape[1]):
            observed=relation[:,column][torch.isfinite(relation[:,column])]
            keep.append(len(observed)>1 and int(observed.unique().numel())>1)
        keep_tensor=torch.tensor(keep,dtype=torch.bool);relation=relation[:,keep_tensor]
        clinical_columns=[x for x,chosen in zip(clinical_columns,keep) if chosen];clinical_types=[x for x,chosen in zip(clinical_types,keep) if chosen]
        if relation.shape[1]<2:raise RuntimeError(f'{IDENTITY_NAMES[identity]} requires at least two task-specific clinical supervision variables for A_rel')
        prior,relation_audit=build_relation_prior(gap,relation.numpy(),seed=seed+identity*1009,k=int(cfg['model']['nmi_neighbors']),resamples=int(cfg['model']['relation_bootstrap_resamples']),stability_temperature=float(cfg['model']['relation_stability_temperature']),projection_temperature=float(cfg['model']['relation_projection_temperature']),clinical_columns=clinical_columns,clinical_types=clinical_types)
        model.set_prior(identity,prior)
        manifest=Path(entry['manifest_path']);provenance={'task_id':identity,'task':IDENTITY_NAMES[identity],'fold_id':str(fold_id),'cohort':entry['task']['dataset_dir'],'partition':'train','source_manifest':str(manifest),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'sample_patient_ids':patients,'patient_ids':list(dict.fromkeys(patients))}
        state={**provenance,**relation_audit};audit[str(identity)]=state
        task_dir=out/'relation_priors'/IDENTITY_NAMES[identity];task_dir.mkdir(parents=True,exist_ok=True)
        (task_dir/'C_clin_metadata.json').write_text(json.dumps({k:state[k] for k in ('task_id','task','fold_id','cohort','partition','source_manifest','source_manifest_sha256','sample_patient_ids','patient_ids','clinical_columns','clinical_types','clinical_matrix_shape','clinical_matrix_sha256','clinical_normalization_mean','clinical_normalization_std')},indent=2)+'\n')
        (task_dir/'A_rel.json').write_text(json.dumps(relation_audit['A_rel'])+'\n')
        torch.save(torch.as_tensor(relation_audit['Pi'],dtype=torch.float32),task_dir/'Pi.pt')
        torch.save(prior.cpu(),task_dir/'M_prior.pt')
        (task_dir/'relation_view_weights.json').write_text(json.dumps({'clinical_weights':relation_audit['clinical_weights'],'projection_weights':relation_audit['projection_weights'],'clinical_bootstraps':relation_audit['clinical_bootstraps'],'projection_bootstraps':relation_audit['projection_bootstraps']},indent=2)+'\n')
    (out/'relation_prior_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
def filter_all(model,entries,device,cfg,seed,out):
    audit={}
    for entry in entries:
        identity=entry['identity'];_,descriptors,target,nuisance,_,patients=collect_statistics(model,entry,device)
        if entry['task']['kind']!='classification':target=normalize_target(target,torch.ones_like(target,dtype=torch.bool),entry)
        full_n=len(descriptors);configured=cfg['model'].get('filtering_max_samples');maximum=full_n if configured is None else int(configured)
        if full_n>maximum:
            generator=torch.Generator().manual_seed(seed+identity*3037)
            chosen=torch.randperm(full_n,generator=generator)[:maximum]
            descriptors,target,nuisance=descriptors[chosen],target[chosen],nuisance[chosen];patients=[patients[i] for i in chosen.tolist()]
        rho=float(task_setting(cfg['model']['retained_channel_ratio'],identity))
        selected,details=filter_channels(descriptors.to(device),target.to(device),nuisance.to(device),entry['task']['kind']=='classification',rho,float(cfg['model']['filtering_fdr']),seed+identity*2029,permutation_resamples=int(cfg['model']['filtering_permutation_resamples']),gcv_min=float(cfg['model']['kci_gcv_min']),gcv_max=float(cfg['model']['kci_gcv_max']),gcv_candidates=int(cfg['model']['kci_gcv_candidates']))
        model.set_channels(identity,selected);audit[str(identity)]={'task':IDENTITY_NAMES[identity],'partition':'train','target_representation_shape':list(target.shape),'nuisance_shape':list(nuisance.shape),'rho_t':rho,'selected_channels':selected.cpu().tolist(),'selected_feature_dim':len(selected),'available_samples':full_n,'filtering_samples':len(descriptors),'patient_ids':patients,'sampling':'all' if full_n<=maximum else 'deterministic_without_replacement','filtering':details}
        task_dir=out/'feature_filtering'/IDENTITY_NAMES[identity];task_dir.mkdir(parents=True,exist_ok=True)
        files={'hsic_statistics':details['hsic_statistics'],'hsic_pvalues':details['hsic_p'],'hsic_rejected':details['hsic_rejected'],'kci_statistics':details['kci_statistics'],'kci_pvalues':details['kci_p'],'kci_rejected':details['kci_rejected'],'retention_ratio':rho,'selected_channels':selected.cpu().tolist()}
        for name,value in files.items():(task_dir/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    (out/'channel_filtering_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

def attach_target_normalizers(entries):
    for entry in entries:
        if entry['task']['kind']=='classification':
            entry['target_mean']=torch.zeros(1);entry['target_std']=torch.ones(1);continue
        frame=entry['loaders']['train'].dataset.frame
        values=torch.as_tensor(frame[entry['task']['targets']].to_numpy(float),dtype=torch.float32)
        masks=torch.isfinite(values);values=torch.nan_to_num(values)
        values=select(values,entry['indices']);masks=select(masks,entry['indices'])
        mean,std=regression_normalizer(values,masks,frame.patient_id.astype(str).tolist())
        entry['target_mean']=mean;entry['target_std']=std

def normalizer_state(entries):
    return {IDENTITY_NAMES[x['identity']]:{'mean':x['target_mean'].tolist(),'std':x['target_std'].tolist()} for x in entries}

def save(model,cfg,history,path,epoch,fold_id,entries):
    torch.save({'model_state':model.state_dict(),'config':cfg,'history':history,'global_epoch':epoch,'fold_id':str(fold_id),'task_identities':IDENTITY_NAMES,'target_normalizers':normalizer_state(entries)},path)
def learning_rate(cfg,epoch):
    base=float(cfg['training']['learning_rate']);minimum=float(cfg['training']['min_learning_rate']);maximum=int(cfg['training']['max_epochs'])
    return minimum+(base-minimum)*.5*(1+math.cos(math.pi*min(epoch,maximum)/maximum))
def set_learning_rate(optimizer,value):
    for group in optimizer.param_groups:group['lr']=value
def optimize_phase(model,entries,optimizer,device,stage,cfg,history,epoch,stop_epoch,name,out,fold_id):
    """Optimize one paper stage to a validation plateau and restore its best state."""
    best=float('inf');stale=0;patience=int(cfg['training']['early_stopping']['patience']);delta=float(cfg['training']['early_stopping']['min_delta'])
    best_path=out/f'best_{name}.pt'
    while epoch<stop_epoch:
        epoch+=1;lr=learning_rate(cfg,epoch-1);set_learning_rate(optimizer,lr)
        tr=balanced_epoch(model,entries,optimizer,device,stage,cfg);va=balanced_epoch(model,entries,None,device,stage,cfg)
        history.append({'epoch':epoch,'stage':name,'learning_rate':lr,'train_loss':tr,'validation_loss':va});print(history[-1],flush=True)
        if epoch%int(cfg['training']['checkpoint_interval_epochs'])==0:save(model,cfg,history,out/f'epoch_{epoch:04d}.pt',epoch,fold_id,entries)
        if va<best-delta:best=va;stale=0;save(model,cfg,history,best_path,epoch,fold_id,entries)
        else:stale+=1
        if stale>=patience:break
    if not best_path.exists():raise RuntimeError(f'{name} produced no finite validation checkpoint')
    payload=torch.load(best_path,map_location=device);model.load_state_dict(payload['model_state'])
    return epoch
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'));parser.add_argument('--manifests-dir',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('checkpoints/unified_six_task'));parser.add_argument('--seed',type=int,default=3407);parser.add_argument('--fold-id',default='single_fold');args=parser.parse_args()
    cfg=load_config(args.config);set_seed(args.seed,True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for training')
    device=torch.device('cuda');tokenizer=AutoTokenizer.from_pretrained(resolve_cached_model(cfg['model']['clinical_encoder']));model=UnifiedCausalCCM(cfg['model']).to(device);args.output.mkdir(parents=True,exist_ok=True)
    manifest_frames=[]
    for task_id in sorted({x[0] for x in SPECS}):
        path=args.manifests_dir/f'{task_id}.csv';manifest_frames.append((path,pd.read_csv(path)))
    validate_cross_task_partitions(manifest_frames)
    entries=[]
    for task_id,identity,indices in SPECS:
        task=get_task(cfg,task_id);manifest=args.manifests_dir/f'{task_id}.csv';entries.append({'task':task,'identity':identity,'indices':indices,'manifest_path':manifest,'loaders':make_loaders(cfg,task,manifest,tokenizer,identity,args.seed)})
    attach_target_normalizers(entries)
    history=[];epoch=0;maximum=int(cfg['training']['max_epochs']);model.set_stage_trainability(1);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
    epoch=optimize_phase(model,entries,optimizer,device,1,cfg,history,epoch,min(maximum,int(cfg['training']['warmup_max_epochs'])),'warmup',args.output,args.fold_id)
    construct_priors(model,entries,device,cfg,args.seed,args.output,args.fold_id)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
    relation_stop=min(maximum,epoch+int(cfg['training']['relation_alignment_max_epochs']))
    epoch=optimize_phase(model,entries,optimizer,device,1,cfg,history,epoch,relation_stop,'prior_alignment',args.output,args.fold_id)
    filter_all(model,entries,device,cfg,args.seed,args.output)
    model.set_stage_trainability(2)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
    if epoch>=maximum:raise RuntimeError('Stage 1 consumed the complete 500-epoch budget before Stage 2')
    epoch=optimize_phase(model,entries,optimizer,device,2,cfg,history,epoch,maximum,'model',args.output,args.fold_id)
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
if __name__=='__main__':main()
