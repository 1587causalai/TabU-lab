"""Build matched train/monitor/test banks without changing original data."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
OLD=B.parent/'v6-puma-single-hostloss-15m-20260929'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
q=json.loads((B/'queue.json').read_text());cfg=json.loads((B/'experiment.json').read_text())
assert digest(Path(q['parent']['checkpoint']))==q['parent']['checkpoint_sha256']
template=json.loads((OLD/'single-manifest.json').read_text())
broot=B.parent/'openml12-frozen-icl-20260927'
bank=json.loads((broot/'bank.json').read_text())
entries={e['name']:e for e in bank['tables']}
(B/'tables').mkdir(exist_ok=False)
receipt={'host':q['host'],'parent_sha256':q['parent']['checkpoint_sha256'],'tables':{},'source_helpers':{f:digest(OLD/f) for f in ['common.py','dual.py','training_support.py','frozen_evaluate.py']}}
for name in cfg['tables']:
 root=B/'tables'/name;root.mkdir();e=entries[name]
 frozen=broot/e['path'];assert digest(frozen)==e['sha256']
 data=json.loads(frozen.read_text());src=B.parent/'openml12-finetune-20260927/data'/f'{name}.json';train_data=json.loads(src.read_text())
 assert train_data['values']==data['values'] and train_data['splits']==data['splits']
 assert all(f['kind']=='numeric' for f in train_data['features'])
 raw=np.asarray(data['values'],dtype=np.float64);assert np.isfinite(raw).all()
 train=data['splits']['train'];test=data['splits']['test'];perm=np.random.default_rng(cfg['seed']).permutation(train).tolist()
 nval=round(.2*len(train));valid,fit=perm[:nval],perm[nval:]
 split=dict(seed=cfg['seed'],fit_row_ids=fit,validation_row_ids=valid,train_row_ids=train,test_row_ids=test)
 if name=='pumadyn32nh':
  prior=json.loads((B.parent/'v6-puma-lrval-grid-20260929/split.json').read_text());assert split==prior
 assert not set(fit)&set(valid) and not set(train)&set(test) and len(fit)>3
 save(root/'split.json',split)
 train_data['splits']={'train':fit,'validation':valid,'test':test};save(root/'fit-data.json',train_data)
 spec=copy.deepcopy(template);spec['experiment_id']=f'openml12-single-{q["host"]}-{name}-20260929';spec['description']='Single-table exploratory adaptation; fit-only update and support; validation-selected checkpoint, no full-train refit.'
 ent=spec['tables'][0];ent.update(id=name,path='fit-data.json',sha256=digest(root/'fit-data.json'),target_column=raw.shape[1]-1,window_rows=min(512,len(fit)))
 save(root/'single-manifest.json',spec);save(root/'queue.json',q)
 seed=e['traces'][0]['codebook_seed'];traces=[]
 for fold in range(3):
  queries=fit[fold::3];queryset=set(queries)
  traces.append(dict(context_row_ids=[i for i in fit if i not in queryset],query_row_ids=queries,codebook_seed=seed))
 fb=dict(table=name,data_sha256=e['sha256'],split_sha256=digest(root/'split.json'),train_traces=traces,test_trace=dict(context_row_ids=fit,query_row_ids=test,codebook_seed=seed),scope='fit-only support; train score covers new-update rows; validation labels excluded')
 save(root/'fit-bank.json',fb)
 receipt['tables'][name]=dict(original_data_sha256=e['sha256'],training_source_sha256=digest(src),split_sha256=digest(root/'split.json'),bank_sha256=digest(root/'fit-bank.json'),fit_rows=len(fit),monitor_rows=len(valid),test_rows=len(test),columns=raw.shape[1],window_rows=min(512,len(fit)))
save(B/'setup-receipt.json',receipt)
print(json.dumps(receipt))
