#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (accuracy_score,cohen_kappa_score,confusion_matrix,f1_score,
                             mean_absolute_error,mean_squared_error,precision_score,r2_score,
                             recall_score,roc_auc_score)
from scipy.stats import pearsonr,spearmanr

from src.config import get_task, load_config
from src.data.dataset import CCMManifestDataset
from src.models import UnifiedCausalCCM
from src.utils.huggingface import resolve_cached_model

IDENTITY_TASK = {0: "task1", 1: "task2", 2: "task3", 3: "task3", 4: "task4", 5: "task5"}
IDENTITY_TARGETS = {2: ["CFS", "TBUT", "SIT", "OSDI"], 3: ["HbA1c"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity", type=int, choices=range(6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-missingness", type=float, choices=(0.0, 0.5, 1.0), default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config); task = get_task(cfg, IDENTITY_TASK[args.identity])
    frame = pd.read_csv(args.manifest).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve_cached_model(cfg["model"]["clinical_encoder"]))
    dataset = CCMManifestDataset(frame, task, tokenizer, int(cfg["data"]["input_resolution"]),
                                 max_length=int(cfg["model"]["max_sequence_length"]),
                                 clinical_missingness=args.text_missingness)
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = UnifiedCausalCCM({**cfg["model"], "pretrained_visual": False})
    model.load_state_dict(payload["model_state"]); model.eval().to(args.device)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(args.device), batch["input_ids"].to(args.device),
                           batch["attention_mask"].to(args.device), args.identity,
                           batch["clinical_available"].to(args.device), stage=2)["prediction"]
            predictions.append(output.cpu().numpy())
    prediction = np.concatenate(predictions)
    names = IDENTITY_TARGETS.get(args.identity, task.get("classes", task.get("targets")))
    output = frame[[c for c in ("image_path", "patient_id", "split") if c in frame]].copy()
    if task['kind']=='classification':
        if prediction.shape[1]==1:
            positive=1/(1+np.exp(-prediction[:,0]));probabilities=np.column_stack((1-positive,positive))
        else:
            probabilities=np.exp(prediction-prediction.max(1,keepdims=True));probabilities/=probabilities.sum(1,keepdims=True)
        output['predicted_label']=probabilities.argmax(1);output['predicted_class']=[names[x] for x in output.predicted_label]
        for index,name in enumerate(names):output[f'probability_{name}']=probabilities[:,index]
    else:
        for index, name in enumerate(names): output[f"prediction_{name}"] = prediction[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output, index=False)
    rows=[]
    if task['kind']=='classification':
        y=frame.label.to_numpy(int);prob=np.exp(prediction-prediction.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);pred=prob.argmax(1);cm=confusion_matrix(y,pred,labels=range(len(names)))
        for i,name in enumerate(names):
            tp=cm[i,i];fn=cm[i].sum()-tp;fp=cm[:,i].sum()-tp;tn=cm.sum()-tp-fn-fp
            rows.append({'class':name,'ACC':tp/max(tp+fn,1),'AUC':roc_auc_score(y==i,prob[:,i]),'SEN':tp/max(tp+fn,1),'SPE':tn/max(tn+fp,1),'PRE':precision_score(y,pred,labels=[i],average='macro',zero_division=0),'F1':f1_score(y,pred,labels=[i],average='macro',zero_division=0),'Kappa':cohen_kappa_score(y,pred)})
        rows.append({'class':'macro/overall','ACC':accuracy_score(y,pred),'AUC':roc_auc_score(y,prob,multi_class='ovr',average='macro') if len(names)>2 else roc_auc_score(y,prob[:,1]),'SEN':recall_score(y,pred,average='macro'),'SPE':np.mean([x['SPE'] for x in rows]),'PRE':precision_score(y,pred,average='macro',zero_division=0),'F1':f1_score(y,pred,average='macro',zero_division=0),'Kappa':cohen_kappa_score(y,pred)})
    else:
        target_names=IDENTITY_TARGETS.get(args.identity,task['targets']);indices=[task['targets'].index(x) for x in target_names]
        truth=frame[task['targets']].to_numpy(float)[:,indices]
        for i,name in enumerate(target_names):
            valid=np.isfinite(truth[:,i]);a=truth[valid,i];b=prediction[valid,i]
            rows.append({'target':name,'RMSE':mean_squared_error(a,b)**.5,'MAE':mean_absolute_error(a,b),'Pearson_r':pearsonr(a,b)[0],'Spearman_rho':spearmanr(a,b)[0],'R2':r2_score(a,b)})
    if args.metrics:
        args.metrics.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.metrics,index=False)
    print(args.output)


if __name__ == "__main__": main()
