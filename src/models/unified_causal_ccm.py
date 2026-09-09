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
    def set_input_dim(self,task:int,input_dim:int,device:torch.device):
        current=self.visual[task]
        if current.in_features!=input_dim:
            self.visual[task]=nn.Linear(input_dim,current.out_features).to(device=device,dtype=current.weight.dtype)
    def forward(self,feature,text,mask,available,task,indices):
        selected=feature.index_select(1,indices)
        visual=self.visual[task](selected.flatten(2).transpose(1,2))
        if text is None or not bool(available.any()): return self.norm[task](visual).mean(1)
        semantic,_=self.attention(visual,text,text,key_padding_mask=~mask.bool(),need_weights=False)
        semantic=semantic*available[:,None,None].to(semantic.dtype)
        return self.norm[task](visual+semantic).mean(1)

class UnifiedCausalCCM(nn.Module):
    def __init__(self,cfg:dict[str,Any]):
        super().__init__();self.visual_encoder=ResNet50Features(bool(cfg['pretrained_visual']));channels=self.visual_encoder.out_channels
        self.text_encoder=AutoModel.from_pretrained(resolve_cached_model(cfg['clinical_encoder']))
        text_hidden=int(self.text_encoder.config.hidden_size)
        semantic_dim=int(cfg.get('patient_semantic_dim',768))
        self.text_projection=nn.Identity() if text_hidden==semantic_dim else nn.Linear(text_hidden,semantic_dim)
        self.structural_priors=nn.ModuleList(StructuralPrior(channels) for _ in TASKS)
        self.register_buffer('channel_indices',torch.arange(channels).repeat(len(TASKS),1),persistent=True)
        self.register_buffer('channel_counts',torch.full((len(TASKS),),channels,dtype=torch.long),persistent=True)
        self.context=SemanticContext(channels,int(cfg['attention_dim']),8);self.task_embedding=nn.Embedding(6,int(cfg['task_embedding_dim']))
        self.hyper=nn.Sequential(nn.Linear(800,512),nn.GELU(),nn.Linear(512,256),nn.GELU())
        self.generators=nn.ModuleList(nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256*out+out)) for out in OUTPUT_DIMS)
        self.auxiliary=nn.ModuleList(nn.Linear(channels,out) for out in OUTPUT_DIMS)
    def set_channels(self,task:int,indices:Tensor):
        if task not in range(len(TASKS)):raise IndexError(f'invalid task identity {task}')
        indices=torch.as_tensor(indices,dtype=torch.long,device=self.channel_indices.device).flatten().unique(sorted=True)
        if not len(indices):raise ValueError('task-specific channel subset cannot be empty')
        if int(indices.min())<0 or int(indices.max())>=self.channel_indices.shape[1]:raise ValueError('channel index out of range')
        self.channel_indices[task].fill_(-1);self.channel_indices[task,:len(indices)]=indices;self.channel_counts[task]=len(indices)
        self.context.set_input_dim(task,len(indices),self.channel_indices.device)
    def selected_channels(self,task:int)->Tensor:return self.channel_indices[task,:int(self.channel_counts[task])]
    def set_prior(self,task:int,prior:Tensor): self.structural_priors[task].set_prior(prior.to(self.channel_indices.device))
    def set_stage_trainability(self,stage:int)->None:
        if stage not in (1,2):raise ValueError('stage must be 1 or 2')
        groups=(self.visual_encoder,self.auxiliary) if stage==1 else (self.text_encoder,self.text_projection,self.context,self.task_embedding,self.hyper,self.generators)
        for parameter in self.parameters():parameter.requires_grad=False
        for module in groups:
            for parameter in module.parameters():parameter.requires_grad=True
    def load_state_dict(self,state_dict,strict:bool=True):
        state=dict(state_dict)
        if 'channel_counts' in state and 'channel_indices' in state:
            for task,count in enumerate(state['channel_counts'].tolist()):self.set_channels(task,state['channel_indices'][task,:int(count)])
        elif 'channel_masks' in state:
            raise RuntimeError('legacy shared-width masking checkpoint is incompatible with exact task-specific projections; retrain with the current implementation')
        return super().load_state_dict(state,strict=strict)
    def forward(self,image,ids,attention,task:int,available,stage:int=2):
        feature=self.visual_encoder(image)
        if stage==1:
            return {'prediction':self.auxiliary[task](F.adaptive_avg_pool2d(feature,1).flatten(1)),
                    'prior_loss':self.structural_priors[task].loss(feature),'hyper_loss':feature.new_zeros(())}
        if bool(available.any()):
            text=self.text_projection(self.text_encoder(input_ids=ids,attention_mask=attention).last_hidden_state)
            text=text*available[:,None,None].to(text.dtype);patient=text[:,0]
        else:
            text=None;patient=feature.new_zeros(len(image),768)
        aware=self.context(feature,text,attention,available,task,self.selected_channels(task));identity=torch.full((len(image),),task,dtype=torch.long,device=image.device)
        code=self.hyper(torch.cat((self.task_embedding(identity),patient),-1));dynamic=self.generators[task](code);out=OUTPUT_DIMS[task]
        weight=dynamic[:,:256*out].view(-1,256,out);bias=dynamic[:,256*out:];prediction=torch.bmm(aware[:,None],weight).squeeze(1)+bias
        return {'prediction':prediction,'prior_loss':feature.new_zeros(()),'hyper_loss':weight.square().mean()+bias.square().mean()}
