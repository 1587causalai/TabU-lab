"""Validate weighted sampling, donor episode continuity, RNG and disposable updates."""
import json, sys, copy, time
from pathlib import Path
from collections import Counter
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
import torch
from train import rewrite_spec, validate_plan, settled_offsets
from dual import DualModel, prepare, score, V53LossConfig
from branch_sampling import BranchSampler
from training_support import episode_seeds, restore_rng
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry
from tabu_lab.curriculum_v53.artifacts import sha256, atomic_json, finite_state
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.restoration_optimizers import adamw

class CountModel(DualModel):
    def forward_prepared(self,*args,**kwargs):
        self.calls+=1
        return super().forward_prepared(*args,**kwargs)

def main():
    q=json.loads((B/'queue.json').read_text()); parent=q['parent']; device=q['device']
    runtime=configure_runtime(device); cap=.75 if q['host']=='gongqian-mini' else .5
    if device=='mps': torch.mps.set_per_process_memory_fraction(cap)
    else: torch.cuda.set_per_process_memory_fraction(cap)
    assert sha256(parent['checkpoint'])==parent['checkpoint_sha256']
    donor=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
    old=load_v55_plan(parent['manifest'])
    assert donor['identity']['parent_manifest_sha256']==sha256(parent['manifest'])
    assert donor['identity']['objective']=='v6_v55_bernoulli50_all730'
    spec=rewrite_spec(old.spec,Path(parent['manifest']).parent,B,q['host'])
    atomic_json(B/'preflight-manifest.json',spec); plan=load_v55_plan(B/'preflight-manifest.json')
    validate_plan(plan)
    assert plan.config.as_dict()==old.config.as_dict() and plan.optimizer==old.optimizer
    offsets=settled_offsets(old,donor)
    # Verify the optimized settlement against the original algorithm across two cycles and a tail.
    probe=copy.deepcopy(donor['state']); probe['cursor']=1463
    fast=settled_offsets(old,{'state':probe}); slow=dict(probe['table_episode_offsets'])
    for i in range(1463):
        t,episode=schedule_entry(old,0,i)
        slow[t.name]=probe['table_episode_offsets'][t.name]+episode+1
    assert fast==slow
    # Three cycles preserve table sequence; no duplicated or skipped per-table episodes.
    seen={t.name:[] for t in plan.tables}; cohort=Counter()
    for i in range(2700):
        t,ep=schedule_entry(plan,0,i); seen[t.name].append(ep); cohort[t.cohort]+=1
    assert cohort['sparse100']==810 and sum(cohort.values())==2700
    assert all(sorted(episodes)==list(range(len(episodes))) for episodes in seen.values())
    assert all(len(v) in (8,9) if n.startswith('sparse_') else len(v)==3 for n,v in seen.items())
    selector=BranchSampler(donor['state']['branch_rng_state']); seq=[selector.draw() for _ in range(200)]
    again=BranchSampler(donor['state']['branch_rng_state']); assert seq==[again.draw() for _ in range(200)]
    saved=selector.state(); expected=[selector.draw() for _ in range(100)]
    restored=BranchSampler(saved); assert expected==[restored.draw() for _ in range(100)]
    model=CountModel(plan.config).to(device,dtype=execution_dtype(device));model.load_state_dict(donor['model'],strict=True)
    opt=adamw(model,plan.optimizer);opt.load_state_dict(donor['optimizer']);restore_rng(donor['rng'])
    config=V53LossConfig(**plan.spec['stages'][0]['loss']); seeds=episode_seeds(plan); tests=[]
    tables=[next(t for t in plan.tables if t.name.startswith('sparse_')),next(t for t in plan.tables if t.name=='pumadyn32nh')]
    for t in tables:
        inputs,request,truth,_=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],offsets[t.name],seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
        for branch in ['v6','v55']:
            tick=time.monotonic();model.calls=0;opt.zero_grad(set_to_none=True)
            prepared=prepare(model,inputs,request,truth,config); s=score(model,prepared,config,branch)
            assert model.calls==1 and len(prepared.visible.request.targets)==int(inputs.query.sum())
            s.loss.backward(); grads=[p for p in model.parameters() if p.grad is not None]
            assert grads and finite_state([p.grad for p in grads]);torch.nn.utils.clip_grad_norm_(grads,plan.optimizer.grad_clip,error_if_nonfinite=True);opt.step()
            assert finite_state(model.state_dict()) and finite_state(opt.state_dict())
            torch.mps.synchronize() if device=='mps' else torch.cuda.synchronize()
            tests.append(dict(table=t.name,branch=branch,rows=inputs.query.shape[0],query_cells=int(inputs.query.sum()),forward_passes=model.calls,optimizer_steps=1,loss=float(s.loss.detach()),seconds=time.monotonic()-tick))
            del s,prepared
    assert sha256(parent['checkpoint'])==parent['checkpoint_sha256']
    previous=B.parent/q['parent']['source_run']
    unchanged=['dual.py','branch_sampling.py','evaluate.py','eval_fit.py']
    assert all(sha256(B/f)==sha256(previous/f) for f in unchanged)
    report=dict(outcome='passed',runtime=runtime,parent_sha256=parent['checkpoint_sha256'],parent_unchanged=True,tests=tests,disposable_updates=len(tests),three_cycle_cohort_counts=dict(cohort),episode_settlement_verified=True,parent_cursor=donor['state']['cursor'],initial_table_episode_offsets=offsets,branch_rng_restored=True,branch_rng_roundtrip=True,unchanged_code_hashes={f:sha256(B/f) for f in unchanged},prior_parity_receipt=str(previous/'preflight.json'))
    atomic_json(B/'preflight.json',report)
    print(json.dumps({k:report[k] for k in ['outcome','disposable_updates','three_cycle_cohort_counts','episode_settlement_verified','branch_rng_restored']}))

if __name__=='__main__':main()
