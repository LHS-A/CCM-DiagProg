from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoModel
from src.models.causal_ccm import ResNet50Features, StructuralPrior
from src.utils.huggingface import resolve_cached_model

TASKS=("oc_diag","sys_diag","oc_reg","hba1c","short","long")
OUTPUT_DIMS=(3,3,4,1,4,2)

class SemanticContext(nn.Module):
    def __init__(self,c:int):
        super().__init__(); self.visual=nn.ModuleList(nn.Linear(c,256) for _ in TASKS)
        self.q=nn.Linear(256,256,bias=False); self.k=nn.Linear(768,256,bias=False); self.v=nn.Linear(768,256,bias=False)
        self.norm=nn.ModuleList(nn.LayerNorm(256) for _ in TASKS)
    def forward(self,f,text,mask,available,task,channels):
        x=f.flatten(2).transpose(1,2)*channels[None,None].to(f.dtype); x=self.visual[task](x)
        score=(self.q(x)@self.k(text).transpose(1,2))/(256**.5)
        score=score.masked_fill(~mask.bool()[:,None],torch.finfo(score.dtype).min)
        semantic=(torch.softmax(score,-1)@self.v(text))*available[:,None,None].to(x.dtype)
        return self.norm[task](x+semantic).mean(1)

class UnifiedCausalCCM(nn.Module):
    def __init__(self,cfg:dict[str,Any]):
        super().__init__(); self.visual_encoder=ResNet50Features(bool(cfg["pretrained_visual"])); c=self.visual_encoder.out_channels
        self.text_encoder=AutoModel.from_pretrained(resolve_cached_model(cfg["clinical_encoder"]))
        if self.text_encoder.config.hidden_size!=768: raise ValueError("ClinicalBERT hidden size must be 768")
        self.structural_prior=StructuralPrior(c); self.register_buffer("channel_masks",torch.ones(6,c,dtype=torch.bool))
        self.context=SemanticContext(c); self.task_embedding=nn.Embedding(6,32)
        self.hyper=nn.Sequential(nn.Linear(800,512),nn.GELU(),nn.Linear(512,256),nn.GELU())
        self.generators=nn.ModuleList(nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256*o+o)) for o in OUTPUT_DIMS)
        self.auxiliary=nn.ModuleList(nn.Linear(c,o) for o in OUTPUT_DIMS)
    def set_channels(self,task:int,indices:Tensor):
        m=torch.zeros(2048,dtype=torch.bool,device=self.channel_masks.device); m[indices.long()]=True; self.channel_masks[task]=m
    def forward(self,image,ids,attention,task:int,available,stage:int=2):
        f=self.visual_encoder(image)
        if stage==1:
            return {"prediction":self.auxiliary[task](F.adaptive_avg_pool2d(f,1).flatten(1)),"prior_loss":self.structural_prior.loss(f),"hyper_loss":f.new_zeros(())}
        text=self.text_encoder(input_ids=ids,attention_mask=attention).last_hidden_state*available[:,None,None].to(f.dtype)
        aware=self.context(f,text,attention,available,task,self.channel_masks[task]); tid=torch.full((len(image),),task,dtype=torch.long,device=image.device)
        code=self.hyper(torch.cat([self.task_embedding(tid),text[:,0]],-1)); dyn=self.generators[task](code); o=OUTPUT_DIMS[task]
        w=dyn[:,:256*o].view(-1,256,o); b=dyn[:,256*o:]; pred=torch.bmm(aware[:,None],w).squeeze(1)+b
        return {"prediction":pred,"prior_loss":f.new_zeros(()),"hyper_loss":w.square().mean()+b.square().mean()}
