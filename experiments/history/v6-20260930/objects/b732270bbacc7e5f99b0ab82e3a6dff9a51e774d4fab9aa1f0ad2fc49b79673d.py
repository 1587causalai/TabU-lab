import json,sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from dual import DualModel,prepare,score,V53LossConfig
from training_support import episode_seeds
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.artifacts import atomic_json,finite_state
from tabu_lab.models.restoration._dtype import execution_dtype
q=json.loads((B/'queue.json').read_text());device=q['device'];configure_runtime(device);cap=.75 if q['host']=='gongqian-mini' else .5
if device=='mps':torch.mps.set_per_process_memory_fraction(cap)
else:torch.cuda.set_per_process_memory_fraction(cap)
plan=load_v55_plan(B/'manifest.json');init=json.loads((B/'initial.json').read_text());payload=torch.load(init['parent'],map_location='cpu',weights_only=False)
m=DualModel(plan.config).to(device,dtype=execution_dtype(device));m.load_state_dict(payload['model'],strict=True);del payload
config=V53LossConfig(**plan.spec['stages'][0]['loss']);seeds=episode_seeds(plan);tests=[]
selected=[next(t for t in plan.tables if t.schema[t.target_column].kind==k) for k in ['numeric','nominal','ordinal']]+[next(t for t in plan.tables if t.name.startswith('sparse_'))]
for index,t in enumerate(selected):
 inputs,request,truth,info=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],init['offsets'][t.name],seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
 prepared=prepare(m,inputs,request,truth,config);m.zero_grad(set_to_none=True);values={}
 for b in ['v6','v55']:
  s=score(m,prepared,config,b);values[b]=float(s.loss.detach());(.5*s.loss).backward();del s
 assert finite_state([p.grad for p in m.parameters() if p.grad is not None])
 delta=None
 if index==0:
  seq={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None};m.zero_grad(set_to_none=True)
  s6=score(m,prepared,config,'v6');s55=score(m,prepared,config,'v55');(.5*(s6.loss+s55.loss)).backward()
  delta=max(float((p.grad-seq[n]).abs().max()) for n,p in m.named_parameters() if n in seq)
  for n,p in m.named_parameters():
   if n in seq:assert torch.allclose(p.grad,seq[n],rtol=1e-4,atol=1e-5)
  del s6,s55,seq
 tests.append(dict(table=t.name,rows=len(inputs.query),query=int(inputs.query.sum()),losses=values,averaged_gradient_max_difference=delta))
 del inputs,request,truth,prepared
atomic_json(B/'smoke.json',dict(outcome='passed',optimizer_steps=0,tests=tests,parameter_set='single',memory_fraction=cap));print(json.dumps(tests))
