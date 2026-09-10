#!/usr/bin/env python3
"""Patient-level 5-fold orchestration for the unified six-task model.

For outer fold k, fold k is test, fold k+1 is validation, and the
remaining three folds are training (approximately 6:2:2).  ``train.py``
then recomputes every relation prior and every task-specific HSIC/KCI
selection from that fold's training rows only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

TASKS=("task1","task2","task3","task4","task5")


def patient_folds(frame:pd.DataFrame,seed:int)->dict[str,int]:
    if frame.patient_id.isna().any() or (frame.patient_id.astype(str).str.strip()=="").any():
        raise ValueError("patient_id is mandatory for five-fold partitioning")
    table=frame[["patient_id"]].drop_duplicates().reset_index(drop=True)
    if "label" in frame:
        counts=frame.groupby("patient_id").label.nunique()
        if (counts>1).any():raise ValueError("a patient has inconsistent class labels")
        table["label"]=table.patient_id.map(frame.groupby("patient_id").label.first())
        if table.label.value_counts().min()>=5:
            splitter=StratifiedKFold(5,shuffle=True,random_state=seed);iterator=splitter.split(table,table.label)
        else:splitter=KFold(5,shuffle=True,random_state=seed);iterator=splitter.split(table)
    else:splitter=KFold(5,shuffle=True,random_state=seed);iterator=splitter.split(table)
    assignment={}
    for fold,(_,held) in enumerate(iterator):
        for patient in table.iloc[held].patient_id.astype(str):assignment[patient]=fold
    return assignment


def create_manifests(source:Path,destination:Path,outer:int,seed:int)->None:
    destination.mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        frame=pd.read_csv(source/f"{task}.csv");assignment=patient_folds(frame,seed)
        patient_fold=frame.patient_id.astype(str).map(assignment)
        frame["split"]=np.where(patient_fold.eq(outer),"test",np.where(patient_fold.eq((outer+1)%5),"validation","train"))
        sets={name:set(frame.loc[frame.split.eq(name),"patient_id"].astype(str)) for name in ("train","validation","test")}
        assert not sets["train"]&sets["validation"] and not sets["train"]&sets["test"] and not sets["validation"]&sets["test"]
        frame.to_csv(destination/f"{task}.csv",index=False)


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-manifests",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,default=Path("artifacts/unified_five_folds"))
    parser.add_argument("--output",type=Path,default=Path("checkpoints/unified_five_folds"))
    parser.add_argument("--config",type=Path,default=Path("configs/default.yaml"))
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--fold",type=int,choices=range(1,6),help="Run one fold; omit to run all five")
    args=parser.parse_args();folds=[args.fold-1] if args.fold else range(5)
    root=Path(__file__).resolve().parents[1]
    for fold in folds:
        manifests=args.work_dir/f"fold_{fold+1}";create_manifests(args.source_manifests,manifests,fold,args.seed)
        command=[sys.executable,str(root/"train.py"),"--config",str(args.config),"--manifests-dir",str(manifests),"--output",str(args.output/f"fold_{fold+1}"),"--seed",str(args.seed+fold)]
        subprocess.run(command,check=True,cwd=root)
    return 0


if __name__=="__main__":raise SystemExit(main())
