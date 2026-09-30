"""Prepare an isolated update/monitor split; retain the original test bank."""
import copy,json,hashlib,sys
from pathlib import Path
B=Path(__file__).resolve().parent
OLD=B.parent/'v6-puma-single-hostloss-15m-20260929'
sys.path[:0]=[str(OLD),str(OLD/'source/src')]
from common import load_data
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.artifacts import sha256,atomic_json
q=json.loads((OLD/'queue.json').read_text())
atomic_json(B/'queue.json',q)
assert sha256(q['parent']['checkpoint'])==q['parent']['checkpoint_sha256']
data,bank=load_data()
sp=json.loads((B/'split.json').read_text())
assert sp['train_row_ids']==data['splits']['train'] and sp['test_row_ids']==data['splits']['test']
assert set(sp['fit_row_ids']).isdisjoint(sp['validation_row_ids'])
assert set(sp['fit_row_ids'])|set(sp['validation_row_ids'])==set(sp['train_row_ids'])
spec=json.loads((OLD/'single-manifest.json').read_text())
original_training_path=(OLD/spec['tables'][0]['path']).resolve()
assert sha256(original_training_path)==spec['tables'][0]['sha256']
new=json.loads(original_training_path.read_text())
assert new['values']==data['values'] and new['splits']['train']==data['splits']['train']
new['splits']={'train':sp['fit_row_ids'],'validation':sp['validation_row_ids'],'test':sp['test_row_ids']}
atomic_json(B/'fit-data.json',new)
spec['experiment_id']='puma-lrval-grid-'+q['host']+'-20260929'
spec['description']='5242 update rows; 1311 monitor rows excluded from new updates; pretrained parent saw them historically.'
spec['tables'][0].update(path='fit-data.json',sha256=sha256(B/'fit-data.json'))
atomic_json(B/'single-manifest.json',spec)
plan=load_v55_plan(B/'single-manifest.json')
assert set(plan.tables[0].row_ids)==set(sp['fit_row_ids'])
(B/'fit-bank.json').write_bytes((OLD/'fit-bank.json').read_bytes())
atomic_json(B/'setup-receipt.json',dict(host=q['host'],parent_sha256=q['parent']['checkpoint_sha256'],
    split_sha256=sha256(B/'split.json'),fit_bank_sha256=sha256(B/'fit-bank.json'),
    update_rows=5242,monitor_rows=1311,test_rows=1639,monitor_exposure='seen by parent; excluded from this run updates',
    old_source=str(OLD/'source'),old_helpers={f:sha256(OLD/f) for f in ['common.py','dual.py','training_support.py','frozen_evaluate.py']}))
print((B/'setup-receipt.json').read_text())
