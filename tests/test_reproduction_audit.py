from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from PIL import Image

from src.models.screening import screen_channels
from src.models.causal_ccm import StructuralPrior
from src.data.dataset import CCMManifestDataset
from train import collect_statistics,learning_rate,task_exclusions,validate_patient_partition
from scripts.train_unified_five_folds import patient_folds
import src.models.unified_causal_ccm as unified


class DummyVisual(nn.Module):
    def __init__(self,pretrained=False):super().__init__();self.out_channels=2048;self.scale=nn.Parameter(torch.ones(()))
    def forward(self,image):return nn.functional.adaptive_avg_pool2d(image.mean(1,keepdim=True),(2,2)).expand(-1,2048,-1,-1)*self.scale


class DummyText(nn.Module):
    def __init__(self):super().__init__();self.config=SimpleNamespace(hidden_size=312);self.embedding=nn.Embedding(100,312)
    def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def model(monkeypatch):
    monkeypatch.setattr(unified,'ResNet50Features',DummyVisual)
    monkeypatch.setattr(unified.AutoModel,'from_pretrained',lambda *_args,**_kwargs:DummyText())
    return unified.UnifiedCausalCCM({'pretrained_visual':False,'clinical_encoder':'unused','patient_semantic_dim':768,'attention_dim':256,'task_embedding_dim':32})


def test_six_task_subsets_are_independent_and_used(monkeypatch):
    net=model(monkeypatch);sets=[]
    for task in range(6):
        indices=torch.arange(task,task+8)
        net.set_channels(task,indices);sets.append(net.selected_channels(task).clone())
        assert net.context.visual[task].in_features==8
    assert all(not torch.equal(sets[0],other) for other in sets[1:])
    image=torch.ones(2,3,8,8);ids=torch.tensor([[1,2,3],[3,2,1]]);mask=torch.ones_like(ids);available=torch.ones(2,dtype=torch.bool)
    for task,out_dim in enumerate(unified.OUTPUT_DIMS):assert net(image,ids,mask,task,available)['prediction'].shape==(2,out_dim)


def test_residual_path_patient_conditioning_and_freezing(monkeypatch):
    net=model(monkeypatch);net.set_channels(2,torch.arange(10));net.eval()
    image=torch.ones(2,3,8,8);mask=torch.ones(2,3,dtype=torch.long);available=torch.ones(2,dtype=torch.bool)
    a=net(image,torch.ones(2,3,dtype=torch.long),mask,2,available)['prediction']
    b=net(image,torch.full((2,3),2,dtype=torch.long),mask,2,available)['prediction']
    assert not torch.allclose(a,b)
    # The shared trunk is conditioned by task identity as well as the patient.
    patient=net.text_projection(net.text_encoder(input_ids=torch.ones(1,3,dtype=torch.long),attention_mask=torch.ones(1,3,dtype=torch.long)).last_hidden_state)[:,0]
    codes=[net.hyper(torch.cat((net.task_embedding(torch.tensor([task])),patient),-1)) for task in range(6)]
    assert all(not torch.allclose(codes[0],code) for code in codes[1:])
    missing=net(image,torch.ones(2,3,dtype=torch.long),mask,2,torch.zeros(2,dtype=torch.bool))['prediction']
    assert torch.isfinite(missing).all()
    net.set_stage_trainability(1);assert any(p.requires_grad for p in net.visual_encoder.parameters());assert not any(p.requires_grad for p in net.text_encoder.parameters())
    net.set_stage_trainability(2);assert not any(p.requires_grad for p in net.visual_encoder.parameters());assert any(p.requires_grad for p in net.text_encoder.parameters())


def test_checkpoint_roundtrip_restores_task_specific_subsets(monkeypatch):
    first=model(monkeypatch)
    for task in range(6):first.set_channels(task,torch.arange(task+2))
    first.eval();state=first.state_dict();second=model(monkeypatch);second.load_state_dict(state);second.eval()
    for task in range(6):assert torch.equal(first.selected_channels(task),second.selected_channels(task))
    args=(torch.ones(1,3,8,8),torch.ones(1,3,dtype=torch.long),torch.ones(1,3,dtype=torch.long),3,torch.ones(1,dtype=torch.bool))
    assert torch.allclose(first(*args)['prediction'],second(*args)['prediction'])


def test_hsic_kci_top_rho_audit_is_complete_and_task_local():
    rng=np.random.default_rng(4);n=36;c=8;y=torch.tensor(rng.normal(size=n),dtype=torch.float32);z=torch.tensor(rng.normal(size=(n,2)),dtype=torch.float32)
    signal=y[:,None]+.02*torch.tensor(rng.normal(size=(n,c)),dtype=torch.float32);descriptors=torch.stack((signal,signal.square()),-1)
    outputs=[]
    for seed,rho in ((11,.25),(23,.30),(37,.35),(41,.40),(53,.45),(67,.50)):
        selected,audit=screen_channels(descriptors,y,z,False,rho,1.0,seed,permutation_min=25,permutation_max=50,permutation_confidence=.90,gcv_candidates=5)
        assert len(selected)<=round(c*rho)
        assert audit['selected_channels']==selected.tolist()
        assert len(audit['hsic_statistics'])==c and len(audit['kci_statistics'])==c
        assert sum(np.isfinite(audit['kci_p']))==audit['kci_candidate_count']==len(audit['hsic_candidates'])
        assert all(a==b for a,b in zip(audit['significant_intersection'],[x and y for x,y in zip(audit['hsic_rejected'],audit['kci_rejected'])]))
        outputs.append(audit)
    assert len({id(x) for x in outputs})==6 and len({x['retention_ratio'] for x in outputs})==6


def test_target_masking_structured_leakage_and_half_up_dropout(tmp_path):
    image=tmp_path/'x.png';Image.new('RGB',(4,4)).save(image)
    class Tokenizer:
        def __init__(self):self.sentences=[]
        def __call__(self,sentence,**_kwargs):
            self.sentences.append(sentence);return {'input_ids':torch.ones(1,4,dtype=torch.long),'attention_mask':torch.ones(1,4,dtype=torch.long)}
    tokenizer=Tokenizer();frame=pd.DataFrame([{'image_path':str(image),'patient_id':'p1','label':0,'clinical_text':'target: 9; age: 40; sex: 1','clinical_target':9.,'clinical_age':40.,'clinical_sex':1.}])
    task={'kind':'classification'};dataset=CCMManifestDataset(frame,task,tokenizer,4,clinical_missingness=.5,excluded_clinical_fields={'target'})
    sample=dataset[0]
    assert dataset.clinical_structured_columns==('clinical_age','clinical_sex')
    assert 'target' not in tokenizer.sentences[-1].casefold()
    assert len([x for x in tokenizer.sentences[-1].split(';') if x.strip()])==1
    assert sample['clinical_structured'].shape==(2,)
    assert sample['data_partition']=='unspecified'
    complete=CCMManifestDataset(frame,task,tokenizer,4,clinical_missingness=0.0,excluded_clinical_fields={'target'})[0]
    missing=CCMManifestDataset(frame,task,tokenizer,4,clinical_missingness=1.0,excluded_clinical_fields={'target'})[0]
    assert bool(complete['clinical_available']) and not bool(missing['clinical_available'])


def test_identity_specific_safe_fields_and_provenance_guards():
    task={'clinical_exclude':['all'],'clinical_exclude_by_identity':{'ocular_reg':['CFS'],'hba1c_reg':['HbA1c','glucose']}}
    assert task_exclusions(task,2)=={'CFS'} and task_exclusions(task,3)=={'HbA1c','glucose'}
    leaking=pd.DataFrame({'patient_id':['p1','p1'],'split':['train','test']})
    with pytest.raises(ValueError,match='patient leakage'):validate_patient_partition(leaking,'fixture.csv')
    class NeverCalled(nn.Module):
        def forward(self,_):raise AssertionError('partition guard must run before feature extraction')
    class FakeModel(nn.Module):
        def __init__(self):super().__init__();self.visual_encoder=NeverCalled()
    fake=FakeModel()
    batch={'data_partition':['validation'],'image':torch.zeros(1,3,2,2)}
    entry={'identity':0,'indices':None,'task':{'kind':'classification'},'loaders':{'train':[batch]}}
    with pytest.raises(RuntimeError,match='non-training data'):collect_statistics(fake,entry,torch.device('cpu'))


def test_paper_cosine_schedule_and_patient_folds():
    cfg={'training':{'learning_rate':1e-4,'min_learning_rate':1e-6,'max_epochs':500}}
    assert learning_rate(cfg,0)==pytest.approx(1e-4)
    assert learning_rate(cfg,500)==pytest.approx(1e-6)
    frame=pd.DataFrame({'patient_id':[f'p{i}' for i in range(20) for _ in range(2)],'label':[i%2 for i in range(20) for _ in range(2)]})
    assignment=patient_folds(frame,3407)
    assert len(assignment)==20 and set(assignment.values())==set(range(5))


def test_eq1_eq3_visual_relation_and_stop_gradient():
    feature=torch.tensor([[[[0.,1.],[2.,3.]],[[3.,1.],[2.,0.]]]],requires_grad=True)
    module=StructuralPrior(2);prior=torch.tensor([[1.,.25],[.25,1.]])
    module.set_prior(prior);loss=module.loss(feature)
    flat=feature.flatten(2);minimum=flat.amin(-1,keepdim=True)
    normalized=(flat-minimum)/(flat.amax(-1,keepdim=True)-minimum+1e-6)
    normalized=normalized/(normalized.square().sum(-1,keepdim=True).sqrt()+1e-6)
    relation=torch.einsum('bcp,bdp->cd',normalized,normalized)
    eye=torch.eye(2);relation=relation*(1-eye);reference=prior*(1-eye)
    expected=(relation/(relation.norm()+1e-6)-reference/(reference.norm()+1e-6)).square().mean()
    assert torch.allclose(loss,expected);loss.backward()
    assert feature.grad is not None and module.prior.grad is None
