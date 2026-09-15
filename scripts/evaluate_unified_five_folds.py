#!/usr/bin/env python3
"""Evaluate all six identities with five fold-specific unified checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SPECS=(("task1",0),("task2",1),("task3",2),("task3",3),("task4",4),("task5",5))


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold-manifests",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--external-manifests",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/unified_five_folds"))
    parser.add_argument("--config",type=Path,default=Path("configs/default.yaml"))
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];records=[]
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
                frame=pd.read_csv(metrics);frame.insert(0,"identity",identity);frame.insert(0,"task",task);frame.insert(0,"fold",fold);frame.insert(0,"cohort",cohort);records.append(frame)
    all_metrics=pd.concat(records,ignore_index=True);args.output.mkdir(parents=True,exist_ok=True);all_metrics.to_csv(args.output/"fold_metrics.csv",index=False)
    keys=[x for x in ("cohort","task","identity","class","target") if x in all_metrics]
    numeric=[x for x in all_metrics.select_dtypes("number").columns if x not in {"fold","identity"}]
    summary=all_metrics.groupby(keys,dropna=False)[numeric].agg(["mean","std"]).reset_index()
    summary.columns=["_".join(str(x) for x in column if x).rstrip("_") if isinstance(column,tuple) else column for column in summary.columns]
    summary.to_csv(args.output/"mean_std_metrics.csv",index=False)
    return 0


if __name__=="__main__":raise SystemExit(main())
