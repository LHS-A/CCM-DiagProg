import torch
from src.config import load_config, get_task
from src.models import build_model

def test_all_tasks_construct_and_forward():
 c=load_config()
 for tid in c['tasks']:
  task=get_task(c,tid); mc=dict(c['model']); mc['pretrained_visual']=False; mc['pretrained_text']=False; mc['text_encoder_config']={'hidden_size':64,'num_hidden_layers':1,'num_attention_heads':4,'intermediate_size':128,'vocab_size':100}
  model=build_model(task,mc).eval(); image=torch.randn(1,3,64,64)
  if task.get('image_only'): out=model(image,torch.empty(1,0,dtype=torch.long),torch.empty(1,0,dtype=torch.long))
  else: out=model(image,torch.ones(1,8,dtype=torch.long),torch.ones(1,8,dtype=torch.long))
  assert out['prediction'].shape==(1,task['output_dim'])
