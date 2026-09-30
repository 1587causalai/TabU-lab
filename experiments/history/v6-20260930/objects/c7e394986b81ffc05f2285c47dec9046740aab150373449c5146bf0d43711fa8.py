"""Freeze the authorized joint618 manifest and exact data inventory."""
from pathlib import Path
import collections, copy, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'tri100-r2-90min-dgx2-20260926/manifests/numeric.json'
NEW = Path('/Users/cms/.openclaw/workspace/projects/causal-superintelligence/TabU/tabu-lab/experiments/local/v55-new120-oldreplay-balanced-dustin-20260926/manifests/new120-oldreplay-balanced-v55.json')
for folder in ('manifests','data','evidence/execution','evidence/qualification','monitoring/evidence'):
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
(ROOT/'source').symlink_to('../ordinal100-20260926/source')
spec = json.loads(OLD.read_text())
for cohort in ('ordinal100','nominal100','tanh100','old120','real78'):
    (ROOT/'data'/cohort).symlink_to('../../ordinal100-20260926/data/'+cohort)
(ROOT/'data/new120').mkdir()
entries = [x for x in json.loads(NEW.read_text())['tables'] if x['cohort']=='new120']
assert len(entries)==120 and all(x['id'].startswith('new120__') for x in entries)
for table in entries:
    src = (NEW.parent/table['path']).resolve()
    assert hashlib.sha256(src.read_bytes()).hexdigest()==table['sha256']
    shutil.copy2(src,ROOT/'data/new120'/src.name)
spec['tables'].extend(copy.deepcopy(entries))
spec['experiment_id']='v55-joint618-120min-dgx2-20260926'
spec['description']='Two hours of successful updates on 618 equally weighted tables; six-cohort mixed shuffle; amended before launch.'
spec['probes'].append({'name':'new120_fit','cohorts':['new120'],'masks':2,'partition':'train','purpose':'fit','recipe':{'kind':'supervised_row','fraction':.25}})
stage=spec['stages'][0]
stage.update(name='joint618_uniform',question='Do all six cohorts improve under equal per-table joint training?',sampling=[{'cohort':c,'episodes':n} for c,n in [('ordinal100',100),('nominal100',100),('tanh100',100),('old120',120),('new120',120),('real78',78)]],checkpoint_every=618)
stage['probes'].append('new120_fit')
out=ROOT/'manifests/joint618.json'
out.write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
inventory=[];counts=collections.defaultdict(collections.Counter)
for entry in spec['tables']:
    path=(out.parent/entry['path']).resolve();raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert sha==entry['sha256'];data=json.loads(raw)
    row={'id':entry['id'],'cohort':entry['cohort'],'target_kind':data['target_kind'],'sha256':sha,'raw_rows':len(data['values']),'train_rows':len(data['splits']['train'])}
    inventory.append(row)
    counts[entry['cohort']]['tables']+=1;counts[entry['cohort']][data['target_kind']]+=1
assert len(inventory)==len({x['id'] for x in inventory})==len({x['sha256'] for x in inventory})==618
assert sum(x['raw_rows'] for x in inventory)==164136
assert sum(x['train_rows'] for x in inventory)==126072
receipt={'schema':'tabu.joint618.inventory.v1','tables':inventory,'cohorts':counts,'table_count':618,'raw_rows':164136,'train_rows':126072,'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_manifests':[str(OLD),str(NEW)],'scientific_change':'sampling and data roster only; preserve model, optimizer, loss, runtime, old probes and all seeds'}
(ROOT/'evidence/inventory.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='tables'}))
