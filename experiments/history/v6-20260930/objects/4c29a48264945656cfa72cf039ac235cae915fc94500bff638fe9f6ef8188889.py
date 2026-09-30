import copy,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.artifacts import atomic_json,sha256
B=Path(__file__).resolve().parent
q=json.loads((B/'queue.json').read_text());oldroot=Path(q['parent_root'])
c=json.loads((oldroot/'half2/run/campaign.json').read_text())
assert c['outcome']=='training_completed' and c['checkpoint_sha256']==q['expected_parent_sha']
assert sha256(c['checkpoint'])==q['expected_parent_sha']
old=load_v55_plan(c['balanced_manifest']);payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
assert payload['model_config']==old.config.as_dict()
assert payload['identity']['parent_manifest_sha256']==sha256(old.path)
spec=copy.deepcopy(old.spec);spec['tables']=[t for t in spec['tables'] if t['cohort']=='history718']
assert len(spec['tables'])==718
for t in spec['tables']:t['path']=os.path.relpath((old.path.parent/t['path']).resolve(),B)
spec['experiment_id']='dual718-'+q['host']+'-20260928';spec['description']='Equal-table 718; shared parameters; 0.5 V6 target-only broadcast + 0.5 V55 original Query squared restoration; two forwards one optimizer step'
spec['stages'][0]['sampling']=[dict(cohort='history718',episodes=718)]
spec['stages'][0]['question']='Does dual-path training improve fixed-Query training-row fit across 718 tables?'
spec['probes']=[];spec['stages'][0]['probes']=[]
atomic_json(B/'manifest.json',spec);plan=load_v55_plan(B/'manifest.json')
consumed={t.name:0 for t in old.tables}
for cursor in range(payload['state']['cursor']):
 t,i=schedule_entry(old,0,cursor);consumed[t.name]=max(consumed[t.name],i+1)
offsets={t.name:payload['state']['table_episode_offsets'][t.name]+consumed[t.name] for t in plan.tables}
atomic_json(B/'initial.json',dict(parent=c['checkpoint'],parent_sha256=q['expected_parent_sha'],parent_supervision=payload['identity']['supervision'],offsets=offsets,optimizer_preserved=True,rng_preserved=True,sampling_cursor_reset=True,table_count=718,windows=sorted(set(t.window_rows for t in plan.tables)),target_kinds={k:sum(t.schema[t.target_column].kind==k for t in plan.tables) for k in ['numeric','nominal','ordinal']}))
print(json.dumps(dict(outcome='prepared',tables=718,windows=sorted(set(t.window_rows for t in plan.tables)))))
