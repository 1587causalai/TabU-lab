"""Read-only parameter test of exact-window all-cell scoring and backward."""
import json, sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from training_support import episode_seeds
from restoration_objective import score_training_episode
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.artifacts import sha256, atomic_json
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v6 import V6Model
from tabu_lab.models.restoration_v53.training import V53LossConfig

q=json.loads((B/'queue.json').read_text());device=q['device'];parent=q['parent']
configure_runtime(device)
cap=.75 if q['host']=='gongqian-mini' else .5
if device=='mps':torch.mps.set_per_process_memory_fraction(cap)
else:torch.cuda.set_per_process_memory_fraction(cap)
assert sha256(parent['checkpoint'])==parent['checkpoint_sha256']
donor=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
plan=load_v55_plan(parent['manifest'])
model=V6Model(plan.config,supervision='target_only').to(device,dtype=execution_dtype(device))
model.load_state_dict(donor['model'],strict=True)
table=next(t for t in plan.tables if t.name=='pumadyn32nh');episode=0
inputs,request,truth,info=build_episode(table,plan.spec['stages'][0]['recipe'][table.kind],episode,
    episode_seeds(plan),device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
score=score_training_episode(model,inputs,truth,request=request,loss_config=V53LossConfig(state_weights=None))
assert score.scored_cells==int((truth.states>=0).sum())>score.query_rows
positions=score.output.request.targets
numeric=torch.tensor([s.kind=='numeric' for s in inputs.schema],device=device)[positions[:,1]]
expected=score.per_cell.new_zeros(())
for mask in (numeric,~numeric):
    if bool(mask.any()):expected=expected+(score.per_cell[mask]/mask.sum()).sum()
torch.testing.assert_close(score.loss,expected)
# Scorer targets are each column's own encoding, never own+target encoding.
for column in score.output.columns:
    rows=positions[column.target_indices,0]
    own=score.output.facts[column.column].answers.encode_targets(truth.values[column.column][rows])
    per=(column.result.encoding-own).square().mean(-1)
    if inputs.schema[column.column].kind=='numeric':per=per*128
    torch.testing.assert_close(score.per_cell[column.target_indices],per)
score.loss.backward()
grads=[p.grad for p in model.parameters() if p.grad is not None]
assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
assert any(bool((g!=0).any()) for g in grads)
if device=='mps':torch.mps.synchronize()
else:torch.cuda.synchronize()
result=dict(outcome='passed',optimizer_updates=0,table=table.name,rows=len(inputs.visible),
    columns=len(inputs.schema),query_rows=score.query_rows,scored_cells=score.scored_cells,
    loss=float(score.loss.detach()),finite_nonzero_gradients=True,own_cell_encoding_verified=True,
    original_parent_sha256=parent['checkpoint_sha256'])
atomic_json(B/'smoke.json',result)
print(json.dumps(result))
