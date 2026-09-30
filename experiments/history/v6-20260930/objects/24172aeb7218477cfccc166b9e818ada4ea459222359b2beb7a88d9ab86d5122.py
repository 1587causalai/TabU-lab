"""Validate the selected loss, full sampling cycle, and finite gradients; no updates."""
import sys,json
from pathlib import Path
import torch
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from train import rewrite_spec,validate_plan
from training_support import episode_seeds
from tabu_lab.curriculum_v53.artifacts import atomic_json,sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v6 import V6Model,score_training_episode
from tabu_lab.models.restoration_v53 import V53LossConfig
q=json.loads((B/'queue.json').read_text());p=q['parent'];device=q['device'];configure_runtime(device)
cap=.75 if q['host']=='gongqian-mini' else .5
if device=='mps':torch.mps.set_per_process_memory_fraction(cap)
else:torch.cuda.set_per_process_memory_fraction(cap)
old=load_v55_plan(p['manifest']);spec=rewrite_spec(old.spec,Path(p['manifest']).parent,B,q['host'])
atomic_json(B/'smoke-manifest.json',spec);plan=load_v55_plan(B/'smoke-manifest.json');validate_plan(plan)
assert sha256(p['checkpoint'])==p['checkpoint_sha256']
payload=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
model=V6Model(plan.config,supervision=q['supervision']).to(device,dtype=execution_dtype(device));model.load_state_dict(payload['model'],strict=True)
tables=[next(t for t in plan.tables if t.name=='pumadyn32nh')]
for kind in ['nominal','ordinal']:tables.append(next(t for t in plan.tables if t.schema[t.target_column].kind==kind))
results=[]
for t in tables:
    model.zero_grad(set_to_none=True)
    inputs,request,truth,info=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],0,episode_seeds(plan),device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
    score=score_training_episode(model,inputs,truth) if q['supervision']=='joint_all' else score_training_episode(model,inputs,truth,request=request,loss_config=V53LossConfig())
    mask=inputs.query.any(1)
    expected=int(((truth.states>=0)&mask[:,None]).sum()) if q['supervision']=='joint_all' else int(inputs.query.sum())
    assert score.scored_cells==expected
    score.loss.backward();gs=[p.grad for p in model.parameters() if p.grad is not None]
    assert gs and all(bool(torch.isfinite(g).all()) for g in gs) and any(bool((g!=0).any()) for g in gs)
    results.append(dict(table=t.name,kind=t.schema[t.target_column].kind,rows=len(inputs.visible),query_rows=score.query_rows,scored_cells=score.scored_cells,loss=float(score.loss.detach())))
atomic_json(B/'smoke.json',dict(outcome='passed',optimizer_updates=0,supervision=q['supervision'],equal_cycle_tables=730,tests=results,parent_sha256=p['checkpoint_sha256']))
print((B/'smoke.json').read_text())
