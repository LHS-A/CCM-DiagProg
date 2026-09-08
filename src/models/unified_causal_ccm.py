from __future__ import annotations
from typing import Any
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from transformers import AutoModel
from src.models.causal_ccm import ResNet50Features,StructuralPrior
from src.utils.huggingface import resolve_cached_model

TASKS=('oc_diag','sys_diag','oc_reg','hba1c','short','long');OUTPUT_DIMS=(3,3,4,1,4,1)

class SemanticContext(nn.Module):
    def __init__(self,channels:int,attention_dim:int=256,heads:int=8):
        super().__init__();self.visual=nn.ModuleList(nn.Linear(channels,attention_dim) for _ in TASKS)
        self.attention=nn.MultiheadAttention(attention_dim,heads,kdim=768,vdim=768,batch_first=True)
        self.norm=nn.ModuleList(nn.LayerNorm(attention_dim) for _ in TASKS)
    def forward(self,feature,text,mask,available,task,channels):
        visual=feature.flatten(2).transpose(1,2)*channels[None,None].to(feature.dtype);visual=self.visual[task](visual)
        if text is None or not bool(available.any()): return self.norm[task](visual).mean(1)
        semantic,_=self.attention(visual,text,text,key_padding_mask=~mask.bool(),need_weights=False)
        semantic=semantic*available[:,None,None].to(semantic.dtype)
        return self.norm[task](visual+semantic).mean(1)

class UnifiedCausalCCM(nn.Module):
    def __init__(self,cfg:dict[str,Any]):
        super().__init__();self.visual_encoder=ResNet50Features(bool(cfg['pretrained_visual']));channels=self.visual_encoder.out_channels
        self.text_encoder=AutoModel.from_pretrained(resolve_cached_model(cfg['clinical_encoder']))
        if self.text_encoder.config.hidden_size!=768: raise ValueError('ClinicalBERT hidden size must be 768')
        self.structural_priors=nn.ModuleList(StructuralPrior(channels) for _ in TASKS)
        self.register_buffer('channel_masks',torch.ones(6,channels,dtype=torch.bool))
        self.context=SemanticContext(channels,int(cfg['attention_dim']),8);self.task_embedding=nn.Embedding(6,int(cfg['task_embedding_dim']))
        self.hyper=nn.Sequential(nn.Linear(800,512),nn.GELU(),nn.Linear(512,256),nn.GELU())
        self.generators=nn.ModuleList(nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256*out+out)) for out in OUTPUT_DIMS)
        self.auxiliary=nn.ModuleList(nn.Linear(channels,out) for out in OUTPUT_DIMS)
    def set_channels(self,task:int,indices:Tensor):
        mask=torch.zeros(self.channel_masks.shape[1],dtype=torch.bool,device=self.channel_masks.device);mask[indices.long()]=True;self.channel_masks[task]=mask
    def set_prior(self,task:int,prior:Tensor): self.structural_priors[task].set_prior(prior.to(self.channel_masks.device))
    def forward(self,image,ids,attention,task:int,available,stage:int=2):
        feature=self.visual_encoder(image)
        if stage==1:
            return {'prediction':self.auxiliary[task](F.adaptive_avg_pool2d(feature,1).flatten(1)),
                    'prior_loss':self.structural_priors[task].loss(feature),'hyper_loss':feature.new_zeros(())}
        if bool(available.any()):
            text=self.text_encoder(input_ids=ids,attention_mask=attention).last_hidden_state
            text=text*available[:,None,None].to(text.dtype);patient=text[:,0]
        else:
            text=None;patient=feature.new_zeros(len(image),768)
        aware=self.context(feature,text,attention,available,task,self.channel_masks[task]);identity=torch.full((len(image),),task,dtype=torch.long,device=image.device)
        code=self.hyper(torch.cat((self.task_embedding(identity),patient),-1));dynamic=self.generators[task](code);out=OUTPUT_DIMS[task]
        weight=dynamic[:,:256*out].view(-1,256,out);bias=dynamic[:,256*out:];prediction=torch.bmm(aware[:,None],weight).squeeze(1)+bias
        return {'prediction':prediction,'prior_loss':feature.new_zeros(()),'hyper_loss':weight.square().mean()+bias.square().mean()}
