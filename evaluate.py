#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd, torch
from scripts._common import load_final_model, make_loader_for_frame
from src.config import load_config, get_task
from src.training.trainer import evaluate

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--manifest'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
 c=load_config(a.config); task=get_task(c,c['active_task']); frame=pd.read_csv(a.manifest or c['data']['manifest']); loader=make_loader_for_frame(c,task,frame[frame.split.eq('test')]); model=load_final_model(Path(a.checkpoint),task,c,torch.device(a.device)); print(evaluate(model,loader,task,torch.device(a.device)))
if __name__=='__main__': main()
