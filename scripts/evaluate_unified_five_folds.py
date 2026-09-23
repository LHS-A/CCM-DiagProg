#!/usr/bin/env python3
"""Evaluate all six identities with five fold-specific unified checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from infer import (IDENTITY_TARGETS, classification_metric_rows,
                   regression_metric_rows)
from src.config import get_task,load_config
from src.data.patient import patient_targets

SPECS=(("task1",0),("task2",1),("task3",2),("task3",3),("task4",4),("task5",5))


def point_metrics(config,task_id,identity,manifest,predictions):
    task=get_task(config,task_id);frame=pd.read_csv(manifest)
    if 'split' in frame and frame.split.eq('test').any():frame=frame.loc[frame.split.eq('test')]
    patient_ids=frame.patient_id.astype(str).tolist();predictions=predictions.copy();predictions['patient_id']=predictions.patient_id.astype(str)
    names=IDENTITY_TARGETS.get(identity,task.get('classes',task.get('targets')))
    if task['kind']=='classification':
        values=torch.as_tensor(frame.label.to_numpy(int));mask=torch.ones(len(frame),dtype=torch.bool)
        truth,_,ids=patient_targets(values,mask,patient_ids,True)
        columns=[f'probability_{name}' for name in names]
        aligned=predictions.set_index('patient_id').loc[ids,columns].to_numpy(float)
        return pd.DataFrame(classification_metric_rows(truth.numpy().astype(int),aligned,names))
    target_names=IDENTITY_TARGETS.get(identity,task['targets']);indices=[task['targets'].index(x) for x in target_names]
    values=torch.as_tensor(frame[task['targets']].to_numpy(float)[:,indices],dtype=torch.float64);mask=torch.isfinite(values)
    truth,valid,ids=patient_targets(torch.nan_to_num(values),mask,patient_ids,False)
    aligned=predictions.set_index('patient_id').loc[ids,[f'prediction_{name}' for name in target_names]].to_numpy(float)
    reference=np.where(valid.numpy(),truth.numpy(),np.nan)
    return pd.DataFrame(regression_metric_rows(reference,aligned,target_names))


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold-manifests",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--external-manifests",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/unified_five_folds"))
    parser.add_argument("--config",type=Path,default=Path("configs/default.yaml"))
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];records=[];prediction_files={};cfg=load_config(args.config)
    cohorts=(("internal",args.fold_manifests),)+( (("external",args.external_manifests),) if args.external_manifests else () )
    for fold in range(1,6):
        checkpoint=args.checkpoints/f"fold_{fold}"/"best_model.pt"
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        for cohort,manifest_root in cohorts:
            for task,identity in SPECS:
                manifest=(manifest_root/f"fold_{fold}"/f"{task}.csv") if cohort=="internal" else (manifest_root/f"{task}.csv")
                if not manifest.is_file():raise FileNotFoundError(manifest)
                destination=args.output/cohort/f"fold_{fold}"/f"identity_{identity}"
                prediction=destination/"predictions.csv";metrics=destination/"metrics.csv"
                command=[sys.executable,str(root/"infer.py"),"--config",str(args.config),"--manifest",str(manifest),"--checkpoint",str(checkpoint),"--identity",str(identity),"--output",str(prediction),"--metrics",str(metrics),"--device",args.device]
                subprocess.run(command,cwd=root,check=True)
                prediction_files[(cohort,task,identity,fold)]=(prediction,manifest)
                frame=pd.read_csv(metrics);frame.insert(0,"identity",identity);frame.insert(0,"task",task);frame.insert(0,"fold",fold);frame.insert(0,"cohort",cohort);records.append(frame)
    all_metrics=pd.concat(records,ignore_index=True);args.output.mkdir(parents=True,exist_ok=True);all_metrics.to_csv(args.output/"fold_metrics.csv",index=False)
    keys=[x for x in ("cohort","task","identity","class","target") if x in all_metrics]
    numeric=[x for x in all_metrics.select_dtypes("number").columns if x not in {"fold","identity"}]
    summary=all_metrics.groupby(keys,dropna=False)[numeric].agg(["mean","std"]).reset_index()
    summary.columns=["_".join(str(x) for x in column if x).rstrip("_") if isinstance(column,tuple) else column for column in summary.columns]
    summary.to_csv(args.output/"mean_std_metrics.csv",index=False)
    point=[]
    for cohort,_ in cohorts:
        for task,identity in SPECS:
            items=[prediction_files[(cohort,task,identity,fold)] for fold in range(1,6)]
            if cohort=='internal':
                combined=pd.concat([pd.read_csv(path) for path,_ in items],ignore_index=True)
                if combined.patient_id.astype(str).duplicated().any():raise RuntimeError(f'duplicate pooled OOF patient for {task}/{identity}')
                # Truth comes from all five held-out manifests; concatenate their
                # test partitions to form the pooled OOF reference.
                frames=[]
                for _,manifest in items:
                    frame=pd.read_csv(manifest);frames.append(frame.loc[frame.split.eq('test')])
                reference=args.output/'_pooled_reference.csv';pd.concat(frames,ignore_index=True).to_csv(reference,index=False)
            else:
                folds=[]
                for path,_ in items:
                    current=pd.read_csv(path);current['patient_id']=current.patient_id.astype(str);folds.append(current.set_index('patient_id'))
                if any(not folds[0].index.equals(frame.index) for frame in folds[1:]):raise RuntimeError(f'external patient order differs across folds for {task}/{identity}')
                combined=folds[0].copy();numeric=[column for column in combined if column.startswith(('probability_','prediction_'))]
                combined[numeric]=sum(frame[numeric] for frame in folds)/len(folds)
                probability=[column for column in numeric if column.startswith('probability_')]
                if probability:
                    combined['predicted_label']=combined[probability].to_numpy().argmax(1)
                    combined['predicted_class']=[probability[index][len('probability_'):] for index in combined.predicted_label]
                combined=combined.reset_index()
                reference=items[0][1]
            metric=point_metrics(cfg,task,identity,reference,combined)
            metric.insert(0,'identity',identity);metric.insert(0,'task',task);metric.insert(0,'cohort',cohort);point.append(metric)
            target=args.output/cohort/'point_estimates';target.mkdir(parents=True,exist_ok=True);combined.to_csv(target/f'identity_{identity}_predictions.csv',index=False)
    pd.concat(point,ignore_index=True).to_csv(args.output/'patient_level_point_estimates.csv',index=False)
    reference=args.output/'_pooled_reference.csv'
    if reference.exists():reference.unlink()
    return 0


if __name__=="__main__":raise SystemExit(main())
