"""Read-only CUDA/model/data admission and exact fixed-Query bank export."""
from pathlib import Path
import collections, datetime, hashlib, json, sys
from common import ROOT,PARENT,PARENT_SHA,PARENT_ROOT,PARENT_EVAL,MANIFEST,atomic,sha
sys.path.insert(0,str(ROOT/'source/src'))
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.artifacts import load_checkpoint
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.models.restoration._dtype import execution_dtype

plan=load_v55_plan(MANIFEST);old=load_v55_plan(PARENT_ROOT/'manifests/numeric.json')
runtime=configure_runtime('cuda:0');assert runtime['dtype']=='float64'
payload,digest=load_checkpoint(PARENT);assert digest==PARENT_SHA and payload['state']['update']==6656
assert plan.config.as_dict()==payload['model_config']
assert plan.identity['source']==old.identity['source']==payload['identity']['source']
for key in ('model','optimizer','seeds'):assert plan.spec[key]==old.spec[key]
assert plan.spec['tables'][:498]==old.spec['tables']
assert plan.spec['probes'][:5]==old.spec['probes']
for key in ('objective','loss','optimizer','recipe'):assert plan.spec['stages'][0][key]==old.spec['stages'][0][key]
parent_eval=json.loads(PARENT_EVAL.read_text());assert parent_eval['checkpoint_sha256']==digest and parent_eval['outcome']=='completed'
assert all(v['complete'] for v in parent_eval['probes'].values())
model=make_model(plan).to(device='cuda:0',dtype=execution_dtype('cuda:0'));model.load_state_dict(payload['model'],strict=True);del model
names=[schedule_entry(plan,0,i)[0].name for i in range(618)]
assert len(names)==len(set(names))==618 and set(names)=={t.name for t in plan.tables}
bank=[]
for probe in plan.spec['probes']:
    for table in plan.tables:
        if table.cohort not in probe['cohorts']:continue
        masks=[]
        for i in range(probe['masks']):
            seeds=dict(plan.spec['seeds']);seeds['evaluation']=int.from_bytes(hashlib.sha256(f"{seeds['evaluation']}/{probe['name']}".encode()).digest()[:8],'little')
            _,_,_,info=build_episode(table,probe['recipe'],i,seeds,'cpu',evaluation=True,partition=probe['partition'],epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
            query=sorted(info['query_row_ids']);support=sorted(set(info['row_ids'])-set(query))
            assert len(query)==51 and len(support)==153
            masks.append({'mask_index':i,'query_addresses':info['query_addresses'],'query_rows':query,'support_rows':support,'mask_seed':info['mask_seed'],'code_seed':info['code_seed'],'window_seed':info['window_seed']})
        bank.append({'table':table.name,'probe':probe['name'],'masks':masks})
assert len(bank)==618 and sum(len(x['masks']) for x in bank)==1238
atomic(ROOT/'evidence/qualification/fixed-query-bank.json',{'tables':bank,'table_count':618,'masks':1238})
receipt={'outcome':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_checkpoint_sha256':digest,'parent_update':6656,'runtime':runtime,'identity':plan.identity,'model':plan.config.as_dict(),'optimizer':plan.spec['optimizer'],'old_probes_and_seeds_unchanged':True,'cycle_unique_tables':618,'fixed_bank_sha256':sha(ROOT/'evidence/qualification/fixed-query-bank.json'),'table_count':618,'mask_count':1238,'cohort_counts':dict(collections.Counter(t.cohort for t in plan.tables)),'source_manifest_sha256':sha(MANIFEST)}
atomic(ROOT/'evidence/qualification/preflight.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ('identity','model','optimizer')}))
