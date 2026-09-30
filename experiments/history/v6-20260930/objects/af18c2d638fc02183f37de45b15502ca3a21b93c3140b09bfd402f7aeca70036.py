"""Disposable branch parity, RNG continuation, real-device update and sampling tests."""
import sys,json,time,random
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import torch
from dual import DualModel,prepare,score,V53LossConfig
from branch_sampling import BranchSampler,BRANCH_CONFIG
from train import validate_plan
from tabu_lab.models.restoration_v6 import V6Model
from tabu_lab.models.restoration_v55 import V55Model
from tabu_lab.models.restoration_v53.training import score_prepared_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.artifacts import finite_state,sha256
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.restoration_optimizers import adamw
from training_support import episode_seeds
class CountModel(DualModel):
 def forward_prepared(self,*args,**kwargs):
  self.calls+=1
  return super().forward_prepared(*args,**kwargs)
def main():
 q=json.loads((B/'queue.json').read_text());device=q['device'];runtime=configure_runtime(device);cap=.75 if q['host']=='gongqian-mini' else .5
 if device=='mps':torch.mps.set_per_process_memory_fraction(cap)
 else:torch.cuda.set_per_process_memory_fraction(cap)
 plan=load_v55_plan(q['parent']['manifest']);validate_plan(plan);donor=torch.load(q['parent']['checkpoint'],map_location='cpu',weights_only=False);assert sha256(q['parent']['checkpoint'])==q['parent']['checkpoint_sha256']
 model=CountModel(plan.config).to(device,dtype=execution_dtype(device));model.load_state_dict(donor['model'],strict=True);opt=adamw(model,plan.optimizer);opt.load_state_dict(donor['optimizer'])
 config=V53LossConfig(**plan.spec['stages'][0]['loss']);seeds=episode_seeds(plan);r=BranchSampler();global_before=random.getstate();seq=[r.draw() for _ in range(20000)];assert .48<seq.count('v6')/len(seq)<.52 and random.getstate()==global_before
 saved=r.state()
 r1=BranchSampler(saved);r2=BranchSampler(saved);assert [r1.draw() for _ in range(100)]==[r2.draw() for _ in range(100)]
 tables=[next(t for t in plan.tables if t.schema[t.target_column].kind==k) for k in ['numeric','nominal','ordinal']]+[next(t for t in plan.tables if t.name=='pumadyn32nh'),next(t for t in plan.tables if t.name.startswith('sparse_'))]
 report=dict(outcome='checking',runtime=runtime,branch_sampling=BRANCH_CONFIG,tests=[],parity={},sampler_20000_v6=seq.count('v6'),sampler_resume_identical=True,global_rng_untouched=True)
 # Match the established direct V6 and V55 scorer from identical parent weights and inputs.
 t=tables[0];inputs,request,truth,_=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],17,seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
 for branch,cls in [('v6',V6Model),('v55',V55Model)]:
  legacy=(cls(plan.config,supervision='target_only') if branch=='v6' else cls(plan.config)).to(device,dtype=execution_dtype(device));legacy.load_state_dict(donor['model'],strict=True)
  prepared=prepare(model,inputs,request,truth,config);model.calls=0;model.zero_grad(set_to_none=True);a=score(model,prepared,config,branch);a.loss.backward();ga={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None};assert model.calls==1
  lp=prepare(legacy,inputs,request,truth,config);z=score_prepared_episode(legacy,lp,config,decode=False);z.loss.backward();assert torch.allclose(a.loss,z.loss,atol=1e-5,rtol=1e-5)
  maxdiff=0.
  for n,param in legacy.named_parameters():
   if param.grad is not None:maxdiff=max(maxdiff,float((ga[n]-param.grad).abs().max()));assert torch.allclose(ga[n],param.grad,atol=2e-5,rtol=2e-4)
  report['parity'][branch]=dict(loss_diff=float(abs(a.loss-z.loss).detach()),max_gradient_diff=maxdiff)
  del a,z,ga,lp,prepared,legacy
 del donor
 for index,t in enumerate(tables):
  inputs,request,truth,_=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],19,seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
  for branch in ['v6','v55']:
   tick=time.monotonic();model.calls=0;opt.zero_grad(set_to_none=True);prepared=prepare(model,inputs,request,truth,config);loss=score(model,prepared,config,branch);assert model.calls==1 and len(prepared.visible.request.targets)==int(inputs.query.sum());loss.loss.backward();grads=[p for p in model.parameters() if p.grad is not None];assert finite_state([p.grad for p in grads]);torch.nn.utils.clip_grad_norm_(grads,plan.optimizer.grad_clip,error_if_nonfinite=True);opt.step();assert finite_state(model.state_dict())
   torch.mps.synchronize() if device=='mps' else torch.cuda.synchronize()
   report['tests'].append(dict(table=t.name,branch=branch,rows=inputs.query.shape[0],scored_cells=len(prepared.visible.request.targets),forward_passes=model.calls,optimizer_steps=1,loss=float(loss.loss.detach()),seconds=time.monotonic()-tick,reserved_bytes=torch.mps.driver_allocated_memory() if device=='mps' else torch.cuda.memory_reserved()))
   del loss,prepared
 assert sha256(q['parent']['checkpoint'])==q['parent']['checkpoint_sha256']
 report.update(outcome='passed',disposable_updates=len(report['tests']),parent_unchanged=True)
 (B/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(outcome='passed',updates=len(report['tests']),parity=report['parity'],max_reserved_bytes=max(x['reserved_bytes'] for x in report['tests']))))
if __name__=='__main__':main()
