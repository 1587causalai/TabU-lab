"""Prepare only a new isolated single-table manifest and frozen evaluation bank."""
import copy,json,os,random,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
q=json.loads((B/'queue.json').read_text());p=q['parent']
assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest()==p['checkpoint_sha256']
spec=json.loads(Path(p['manifest']).read_text())
table=next(t for t in spec['tables'] if t['id']=='pumadyn32nh')
table['path']=os.path.relpath((Path(p['manifest']).parent/table['path']).resolve(),B)
table['cohort']='puma';table['window_rows']=512
spec['tables']=[table];spec['probes']=[]
spec['experiment_id']='v6-puma-single-hostloss-15m-'+q['host']+'-20260929'
spec['description']='Puma single-table fit diagnostic; preserved parent weights/AdamW/RNG; new schedule; no replay.'
s=spec['stages'][0];s['sampling']=[dict(cohort='puma',episodes=1)]
s['name']='puma_single';s['max_seconds']=900;s['question']='Fit Puma without inter-table interference'
s['loss']=dict(state_weights=[0.,1.,0.,0.],discrete_weight=1.)
(B/'single-manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
bank_root=B.parent/'openml12-frozen-icl-20260927'
entry=next(e for e in json.loads((bank_root/'bank.json').read_text())['tables'] if e['name']=='pumadyn32nh')
data=json.loads((bank_root/entry['path']).read_text())
tr=data['splits']['train'];te=data['splits']['test'];perm=list(tr)
random.Random(20260929).shuffle(perm)
traces=[]
for k in range(3):
    query=perm[k::3];qs=set(query)
    traces.append(dict(context_row_ids=[i for i in tr if i not in qs],query_row_ids=query,codebook_seed=entry['traces'][0]['codebook_seed']))
bank=dict(data_sha256=entry['sha256'],train_traces=traces,test_trace=dict(context_row_ids=tr,query_row_ids=te,codebook_seed=entry['traces'][0]['codebook_seed']))
(B/'fit-bank.json').write_text(json.dumps(bank,sort_keys=True)+'\n')
print(json.dumps(dict(host=q['host'],parent_verified=True,train_rows=len(tr),test_rows=len(te),features=len(data['values'][0])-1,bank_sha256=hashlib.sha256((B/'fit-bank.json').read_bytes()).hexdigest())))
