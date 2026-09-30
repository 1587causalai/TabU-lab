import os,sys,json,time,copy,threading,resource,subprocess,re
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=Path.home()/'experiments/v6-window1024-continue60m-20260928'
sys.path[:0]=[str(OLD),str(OLD/'source/src')]
import torch
from training_support import restore_rng,episode_seeds
from tabu_lab.curriculum_v53.artifacts import atomic_json,append_event,finite_state,sha256
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration_v6 import V6Model,score_training_episode
from tabu_lab.restoration_optimizers import adamw
from tabu_lab.models.restoration_v53 import V53LossConfig
receipt={'outcome':'starting','window':2048,'purpose':'capacity_probe_not_production_continuation','updates_target':24}
atomic_json(BASE/'status.json',receipt)
try:
 runtime=configure_runtime('mps');torch.mps.set_per_process_memory_fraction(0.75)
 c=json.loads((OLD/'half2/run/campaign.json').read_text())
 assert c['outcome']=='training_completed' and sha256(c['checkpoint'])==c['checkpoint_sha256']
 donor=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
 assert donor['identity']['supervision']=='target_only'
 oldplan=load_v55_plan(c['balanced_manifest']);spec=copy.deepcopy(oldplan.spec)
 for t in spec['tables']:
  t['path']=os.path.relpath((Path(c['balanced_manifest']).parent/t['path']).resolve(),BASE)
  if t['cohort']!='history618':t['window_rows']=2048
 spec['experiment_id']='v6-window2048-capacity-probe-mini-20260928'
 atomic_json(BASE/'manifest.json',spec);plan=load_v55_plan(BASE/'manifest.json');stage=plan.spec['stages'][0]
 model=V6Model(plan.config,supervision='target_only').to('mps',dtype=torch.float32)
 model.load_state_dict(donor['model'],strict=True);optimizer=adamw(model,plan.optimizer);optimizer.load_state_dict(donor['optimizer']);restore_rng(donor['rng'])
 cursor=donor['state']['cursor'];offsets=donor['state']['table_episode_offsets'];del donor
 seeds=episode_seeds(plan);samples=[];stop=threading.Event()
 def sample():
  while not stop.is_set():
   try:samples.append({'time':time.time(),'allocated':torch.mps.current_allocated_memory(),'driver':torch.mps.driver_allocated_memory(),'rss_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
   except Exception:pass
   stop.wait(.1)
 thread=threading.Thread(target=sample,daemon=True);thread.start()
 receipt.update(outcome='running',runtime=runtime,parent_sha256=c['checkpoint_sha256'],supervision='target_only',recommended_max_memory=torch.mps.recommended_max_memory(),allocator_fraction_cap=.75,updates=[])
 atomic_json(BASE/'status.json',receipt)
 for i in range(24):
  start=time.monotonic();table,ep=schedule_entry(plan,0,cursor+i);assert table.cohort!='history618'
  sample_start=len(samples)
  inputs,request,truth,info=build_episode(table,stage['recipe'][table.kind],offsets[table.name]+ep,seeds,'mps',epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
  assert inputs.query.shape[0]==min(2048,table.train_rows)
  model.train();optimizer.zero_grad(set_to_none=True)
  score=score_training_episode(model,inputs,truth,request=request,loss_config=V53LossConfig(**stage["loss"]));score.loss.backward()
  params=[p for p in model.parameters() if p.grad is not None];assert params and finite_state([p.grad for p in params])
  norm=torch.nn.utils.clip_grad_norm_(params,plan.optimizer.grad_clip,error_if_nonfinite=True);optimizer.step();torch.mps.synchronize()
  row={'step':i+1,'table':table.name,'train_rows':inputs.query.shape[0],'total_train_rows':table.train_rows,'query_rows':score.query_rows,'seconds':time.monotonic()-start,'loss':float(score.loss.detach()),'gradient_norm':float(norm),'allocated':torch.mps.current_allocated_memory(),'driver':torch.mps.driver_allocated_memory(),'rss_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  ss=samples[sample_start:]
  row['sampled_peak_allocated']=max([x['allocated'] for x in ss]+[row['allocated']]);row['sampled_peak_driver']=max([x['driver'] for x in ss]+[row['driver']])
  append_event(BASE/'updates.jsonl',row);receipt['updates'].append(row);atomic_json(BASE/'status.json',receipt);print(json.dumps(row),flush=True)
  del score,inputs,request,truth
 stop.set();thread.join(2)
 assert len({r['table'] for r in receipt['updates']})==12
 receipt.update(outcome='completed',sampled_peak_allocated=max(x['allocated'] for x in samples),sampled_peak_driver=max(x['driver'] for x in samples),rss_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,total_seconds=sum(x['seconds'] for x in receipt['updates']),production_checkpoint_written=False)
 atomic_json(BASE/'memory-samples.json',samples);atomic_json(BASE/'status.json',receipt)
except Exception as e:
 receipt.update(outcome='failed',error_type=type(e).__name__,error=str(e));atomic_json(BASE/'status.json',receipt);raise
