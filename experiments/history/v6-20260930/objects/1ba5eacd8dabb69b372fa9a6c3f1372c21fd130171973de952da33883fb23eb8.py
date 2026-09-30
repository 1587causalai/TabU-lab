import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from dual import DualModel,prepare,score,V53LossConfig
from training_support import checkpoint,restore_rng,episode_seeds
from tabu_lab.curriculum_v53.artifacts import atomic_json,append_event,finite_state,sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.restoration_optimizers import adamw

def main(a):
 root=B/a.segment/'run';root.mkdir(parents=True,exist_ok=False)
 receipt=dict(outcome='starting',supervision='target_only',objective='0.5*v6_target_only+0.5*v55_query_squared',weights=[.5,.5],forward_passes_per_update=2,optimizer_steps_per_update=1,successful_update_seconds_target=a.seconds,parent_sha256=a.parent_sha)
 atomic_json(root/'campaign.json',receipt)
 try:
  runtime=configure_runtime(a.device);cap=.75 if a.host=='gongqian-mini' else .5
  if a.device=='mps':torch.mps.set_per_process_memory_fraction(cap)
  else:torch.cuda.set_per_process_memory_fraction(cap)
  plan=load_v55_plan(B/'manifest.json');stage=plan.spec['stages'][0];config=V53LossConfig(**stage['loss'])
  assert sha256(a.parent)==a.parent_sha
  donor=torch.load(a.parent,map_location='cpu',weights_only=False);assert donor['model_config']==plan.config.as_dict()
  model=DualModel(plan.config).to(a.device,dtype=execution_dtype(a.device));model.load_state_dict(donor['model'],strict=True)
  opt=adamw(model,plan.optimizer);opt.load_state_dict(donor['optimizer']);restore_rng(donor['rng'])
  initial=json.loads((B/'initial.json').read_text())
  state=dict(update=donor['state']['update'],parent_update=donor['state']['update'],cursor=0 if a.segment=='half1' else donor['state']['cursor'],successful_update_seconds=0.,table_episode_offsets=initial['offsets'] if a.segment=='half1' else donor['state']['table_episode_offsets'],table_updates={t.name:0 for t in plan.tables},exposure={'old618':0,'sparse100':0})
  identity=dict(schema='tabu.dual718.v1',supervision='target_only',objective=receipt['objective'],weights=[.5,.5],model_config=plan.config.as_dict(),parent_sha256=a.parent_sha,parent_identity_sha256=donor['identity']['sha256'],parent_manifest_sha256=sha256(B/'manifest.json'),runtime=runtime,sampling=stage['sampling'],code={n:sha256(B/n) for n in ['dual.py','train.py','training_support.py']})
  identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();del donor
  receipt.update(outcome='admitted',parent_update=state['parent_update'],runtime=runtime,identity=identity,allocator_fraction_cap=cap,optimizer_restored=True,rng_restored=True,balanced_manifest=str(B/'manifest.json'),sampling_cursor_reset=a.segment=='half1');atomic_json(root/'campaign.json',receipt)
  seeds=episode_seeds(plan);last_save=0
  while state['successful_update_seconds']<a.seconds:
   t,rel=schedule_entry(plan,0,state['cursor']);episode=state['table_episode_offsets'][t.name]+rel;start=time.monotonic()
   inputs,request,truth,info=build_episode(t,stage['recipe'][t.kind],episode,seeds,a.device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
   model.train();opt.zero_grad(set_to_none=True);prepared=prepare(model,inputs,request,truth,config);losses={}
   # Sequential backward frees each graph. Parameters are unchanged until BOTH finish.
   for branch in ['v6','v55']:
    s=score(model,prepared,config,branch);losses[branch]=float(s.loss.detach());(.5*s.loss).backward();del s
   grads=[p for p in model.parameters() if p.grad is not None]
   if not grads or not finite_state([p.grad for p in grads]):raise FloatingPointError('nonfinite gradient')
   norm=torch.nn.utils.clip_grad_norm_(grads,plan.optimizer.grad_clip,error_if_nonfinite=True);opt.step()
   if not finite_state(model.state_dict()) or not finite_state(opt.state_dict()):raise FloatingPointError('nonfinite update')
   torch.mps.synchronize() if a.device=='mps' else torch.cuda.synchronize()
   seconds=time.monotonic()-start;state['successful_update_seconds']+=seconds;state['update']+=1;state['cursor']+=1;state['table_updates'][t.name]+=1
   family='sparse100' if t.name.startswith('sparse_') else 'old618';state['exposure'][family]+=1
   append_event(root/'updates.jsonl',dict(update=state['update'],table=t.name,family=family,episode_index=episode,loss=.5*(losses['v6']+losses['v55']),loss_v6=losses['v6'],loss_v55=losses['v55'],gradient_norm=float(norm),seconds=seconds,successful_update_seconds=state['successful_update_seconds'],train_rows=len(inputs.query),query_rows=int(inputs.query.sum()),scored_cells=len(prepared.visible.request.targets),forward_passes=2,optimizer_steps=1,accelerator_allocated_bytes=torch.mps.current_allocated_memory() if a.device=='mps' else torch.cuda.memory_allocated()))
   del inputs,request,truth,prepared
   if state['successful_update_seconds']-last_save>=300 or state['successful_update_seconds']>=a.seconds:
    path,digest=checkpoint(root,model,opt,state,identity)
    receipt.update(outcome='training_completed' if state['successful_update_seconds']>=a.seconds else 'running',checkpoint=str(path),checkpoint_sha256=digest,checkpoint_update=state['update'],successful_update_seconds=state['successful_update_seconds'],cohort_updates=state['exposure'],table_updates=state['table_updates'],table_coverage=sum(v>0 for v in state['table_updates'].values()))
    atomic_json(root/'campaign.json',receipt);last_save=state['successful_update_seconds'];print(json.dumps(dict(outcome=receipt['outcome'],seconds=last_save,updates=state['update']-state['parent_update'])),flush=True)
 except Exception as e:
  receipt.update(outcome='failed',error_type=type(e).__name__,error=str(e));atomic_json(root/'campaign.json',receipt);raise
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['host','device','segment','parent','parent-sha']:p.add_argument('--'+k,required=True)
 p.add_argument('--seconds',type=float,required=True);main(p.parse_args())
