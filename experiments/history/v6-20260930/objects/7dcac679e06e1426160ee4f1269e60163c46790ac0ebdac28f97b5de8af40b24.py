"""Disposable correctness/update checks; parent files and formal optimizer are untouched."""
import sys,json,time,hashlib,dataclasses
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import torch
from mix_model import MixedV6Model,encode_pair,donor_map,MIX_CONFIG
from tabu_lab.models.restoration_v6 import V6Model,score_training_episode
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.artifacts import finite_state,sha256
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.restoration_optimizers import adamw
from training_support import episode_seeds

def main():
 q=json.loads((B/'queue.json').read_text());device=q['device'];host=q['host'];runtime=configure_runtime(device)
 cap=.75 if host=='gongqian-mini' else .5
 if device=='mps':torch.mps.set_per_process_memory_fraction(cap)
 else:torch.cuda.set_per_process_memory_fraction(cap)
 plan=load_v55_plan(q['parent']['manifest']);donor=torch.load(q['parent']['checkpoint'],map_location='cpu',weights_only=False)
 assert sha256(q['parent']['checkpoint'])==q['parent']['checkpoint_sha256']
 model=MixedV6Model(plan.config).to(device,dtype=execution_dtype(device));model.load_state_dict(donor['model'],strict=True)
 legacy=V6Model(plan.config,supervision='target_only').to(device,dtype=execution_dtype(device));legacy.load_state_dict(donor['model'],strict=True)
 opt=adamw(model,plan.optimizer);opt.load_state_dict(donor['optimizer']);del donor
 config=V53LossConfig(**plan.spec['stages'][0]['loss']);seeds=episode_seeds(plan)
 tables=[next(t for t in plan.tables if t.schema[t.target_column].kind==k) for k in ['numeric','nominal','ordinal']]
 tables += [next(t for t in plan.tables if t.name=='pumadyn32nh'),next(t for t in plan.tables if t.name.startswith('sparse_'))]
 report=dict(outcome='checking',runtime=runtime,config=MIX_CONFIG,tests=[])
 for index,t in enumerate(tables):
  inputs,request,truth,info=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],17,seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
  tick=time.monotonic();prepared=model.prepare(inputs,request)
  rawbase,mixed,mapping=encode_pair(model.encoder,inputs,prepared.features)
  oldbase=model.encoder.forward_prepared(inputs,prepared.features)
  assert torch.allclose(rawbase,oldbase,atol=2e-6,rtol=2e-6)
  target=mapping['target_column'];n,m=inputs.visible.shape
  for j,donors in enumerate(mapping['donors']):
   assert len(set(donors))==len(donors)==min(2,len([k for k in range(m) if k not in (j,target)]))
   assert j not in donors and target not in donors
  reference=rawbase.clone()
  extras=torch.stack([sum((torch.where(inputs.visible[:,k,None],rawbase[:n,k],0) for k in ds),torch.zeros_like(rawbase[:n,j])) for j,ds in enumerate(mapping['donors'])],1)
  reference[:n,:m]=torch.where((inputs.visible|inputs.query)[...,None],rawbase[:n,:m]+extras,0)
  assert torch.allclose(mixed,reference,atol=1e-5,rtol=1e-5)
  # Raw-before-W and projected-after-W must have matching weight gradients.
  probe=torch.randn_like(mixed);g1=torch.autograd.grad((mixed*probe).sum(),model.encoder.projection.weight,retain_graph=True)[0]
  g2=torch.autograd.grad((reference*probe).sum(),model.encoder.projection.weight)[0]
  assert torch.allclose(g1,g2,atol=2e-3,rtol=2e-4)
  gs=torch.get_rng_state().clone();assert donor_map(inputs)==donor_map(inputs);assert torch.equal(gs,torch.get_rng_state())
  twin=dataclasses.replace(inputs,values=tuple(torch.where(inputs.query[:,j],torch.ones_like(v),v) for j,v in enumerate(inputs.values)))
  assert all(torch.equal(a,b) for a,b in zip(inputs.values,twin.values))
  # Column-key mapping equivariance including small-width fallback.
  perm=list(reversed(range(m)));rev={old:new for new,old in enumerate(perm)}
  ip=dataclasses.replace(inputs,schema=tuple(inputs.schema[k] for k in perm),values=tuple(inputs.values[k] for k in perm),visible=inputs.visible[:,perm],query=inputs.query[:,perm])
  pm=donor_map(ip)
  assert all(pm[new]==[rev[k] for k in mapping['donors'][old]] for new,old in enumerate(perm))
  if index==0:
   model.mix_enabled=False
   a=score_training_episode(model,inputs,truth,request=request,loss_config=config)
   b=score_training_episode(legacy,inputs,truth,request=request,loss_config=config)
   assert torch.allclose(a.loss,b.loss,atol=1e-5,rtol=1e-5)
   report['disabled_matches_legacy_loss']=float(abs(a.loss-b.loss).detach());del a,b
   model.mix_enabled=True
  del prepared,rawbase,mixed,oldbase,reference,extras,g1,g2
  opt.zero_grad(set_to_none=True);s=score_training_episode(model,inputs,truth,request=request,loss_config=config)
  assert s.scored_cells==int(inputs.query.sum());s.loss.backward();grads=[p for p in model.parameters() if p.grad is not None]
  assert finite_state([p.grad for p in grads]);torch.nn.utils.clip_grad_norm_(grads,plan.optimizer.grad_clip,error_if_nonfinite=True);opt.step();assert finite_state(model.state_dict())
  torch.mps.synchronize() if device=='mps' else torch.cuda.synchronize()
  report['tests'].append(dict(table=t.name,kind=t.kind,rows=n,columns=m,loss=float(s.loss.detach()),seconds=time.monotonic()-tick,mapping=mapping,allocated=(torch.mps.current_allocated_memory() if device=='mps' else torch.cuda.memory_allocated()),reserved=(torch.mps.driver_allocated_memory() if device=='mps' else torch.cuda.memory_reserved())))
  del s,inputs,request,truth
 assert sha256(q['parent']['checkpoint'])==q['parent']['checkpoint_sha256']
 report.update(outcome='passed',successful_disposable_updates=len(tables),parent_unchanged=True)
 (B/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'outcome':'passed','updates':len(tables),'max_reserved_bytes':max(x['reserved'] for x in report['tests'])}))
if __name__=='__main__':main()
