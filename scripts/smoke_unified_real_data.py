#!/usr/bin/env python3
"""Reduced-cost, real-image end-to-end smoke test of the paper pipeline."""
from __future__ import annotations
import argparse,tempfile,sys,copy
from pathlib import Path
import pandas as pd,torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import get_task,load_config
from src.data.dataset import CCMManifestDataset
from src.models import UnifiedCausalCCM
from src.models.screening import screen_channels
from src.utils.huggingface import resolve_cached_model
from train import collect_statistics,construct_priors,prediction_loss,relation_fields,to_device

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=Path('artifacts/one_fold_splits/task3.csv'));parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device=torch.device(args.device);cfg=load_config();task=get_task(cfg,'task3');frame=pd.read_csv(args.manifest)
    frame=frame.loc[frame.split.eq('train')].groupby('patient_id',as_index=False).head(1).head(8).copy();frame['split']='train'
    if frame.patient_id.nunique()<6:raise RuntimeError('at least six real training patients are required')
    tokenizer=AutoTokenizer.from_pretrained(resolve_cached_model(cfg['model']['clinical_encoder']))
    ds=CCMManifestDataset(frame,task,tokenizer,384,max_length=128,clinical_missingness=.5,excluded_clinical_fields={'CFS','TBUT','SIT','OSDI'},partition='train',relation_clinical_fields=relation_fields(frame,task,2))
    loader=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0);entry={'identity':2,'indices':[1,2,3,4],'task':task,'manifest_path':args.manifest,'loaders':{'train':loader}}
    model=UnifiedCausalCCM({**cfg['model'],'pretrained_visual':False}).to(device);model.set_stage_trainability(1);batch=to_device(next(iter(loader)),device)
    out=model(batch['image'],batch['input_ids'],batch['attention_mask'],2,batch['clinical_available'],1);loss=prediction_loss(out['prediction'],batch['target'][:,1:],batch['target_mask'][:,1:],'regression');loss.backward()
    gap,descriptors,target,nuisance,_,_=collect_statistics(model,entry,device)
    smoke_cfg=copy.deepcopy(cfg);smoke_cfg['model'].update({'nmi_neighbors':2,'relation_bootstrap_initial':2,'relation_bootstrap_increment':1,'relation_bootstrap_max':2})
    with tempfile.TemporaryDirectory() as prior_directory:
        construct_priors(model,[entry],device,smoke_cfg,7,Path(prior_directory),'smoke_fold')
        required=Path(prior_directory)/'relation_priors'/'ocular_reg'
        assert all((required/name).is_file() for name in ('C_clin_metadata.json','A_rel.json','Pi.pt','M_prior.pt','relation_view_weights.json'))
    model.zero_grad(set_to_none=True);out=model(batch['image'],batch['input_ids'],batch['attention_mask'],2,batch['clinical_available'],1);(out['prior_loss']+prediction_loss(out['prediction'],batch['target'][:,1:],batch['target_mask'][:,1:],'regression')).backward()
    selected,audit=screen_channels(descriptors.to(device),target.to(device),nuisance.to(device),False,.01,1.0,11,permutation_min=5,permutation_max=5,permutation_confidence=.90,gcv_candidates=3);model.set_channels(2,selected);model.set_stage_trainability(2)
    model.zero_grad(set_to_none=True);out=model(batch['image'],batch['input_ids'],batch['attention_mask'],2,batch['clinical_available'],2);(prediction_loss(out['prediction'],batch['target'][:,1:],batch['target_mask'][:,1:],'regression')+float(cfg['model']['hyper_loss_weight'])*out['hyper_loss']).backward()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'smoke.pt';torch.save({'model_state':model.state_dict(),'config':cfg},path);clone=UnifiedCausalCCM({**cfg['model'],'pretrained_visual':False}).to(device);clone.load_state_dict(torch.load(path,map_location=device)['model_state']);clone.eval()
        with torch.inference_mode():prediction=clone(batch['image'],batch['input_ids'],batch['attention_mask'],2,batch['clinical_available'],2)['prediction']
    assert prediction.shape==(len(batch['image']),4) and torch.isfinite(prediction).all() and audit['kci_candidate_count']>0
    print('REAL_DATA_PIPELINE_SMOKE_PASS',len(frame),len(selected))

if __name__=='__main__':main()
