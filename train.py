#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd, torch
from src.config import load_config, get_task
from src.models import build_model
from src.training.trainer import Trainer
from scripts._common import make_loaders

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
 c=load_config(a.config); tid=c['active_task']; task=get_task(c,tid); manifest=Path(a.manifest or c['data']['manifest'])
 loaders,_=make_loaders(c,task,manifest)
 model=build_model(task,c['model'])
 Trainer(model,task,c,torch.device(a.device)).fit(loaders['train'],loaders['validation'],Path(c['project']['checkpoints_root'])/tid)
if __name__=='__main__': main()
