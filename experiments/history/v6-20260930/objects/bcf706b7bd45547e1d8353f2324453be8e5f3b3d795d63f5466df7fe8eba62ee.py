"""Read-only recomputation of stored predictions and validation stopping choices."""
import json
import sys
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parent
OLD = B.parent/'v6-puma-single-hostloss-15m-20260929'
sys.path[:0] = [str(OLD), str(OLD/'source/src')]
from common import load_data
from tabu_lab.curriculum_v53.artifacts import sha256

s = json.loads((B/'terminal.json').read_text())
sp = json.loads((B/'split.json').read_text())
assert s['outcome']=='completed' and s['selection_completed_utc'] < s['test_started_utc']
assert s['script_sha256']==sha256(B/'run.py') and s['split_sha256']==sha256(B/'split.json')
assert s['bank_sha256']==sha256(OLD/'fit-bank.json')
data, _ = load_data()
assert sp['train_row_ids']==data['splits']['train'] and sp['test_row_ids']==data['splits']['test']
fit, valid, train, test = (set(sp[k+'_row_ids']) for k in ['fit','validation','train','test'])
assert len(fit)==5242 and len(valid)==1311 and fit.isdisjoint(valid)
assert fit|valid==train and train.isdisjoint(test)
raw=np.asarray(data['values'],dtype=np.float64);y=raw[:,-1]
train_y=y[data['splits']['train']];med=np.median(train_y);mad=np.median(np.abs(train_y-med))
trace=[json.loads(l) for l in (B/'trace.jsonl').read_text().splitlines() if l.strip()]
for name, model in s['models'].items():
    choice=s['selected'][name]
    assert model['refit_rounds']==model['selected_rounds']==choice['best_rounds']
    assert choice['stopped_by_patience']
    if name.startswith('mlp'):
        rows=[r for r in trace if r['model']==name and r['phase']=='selection']
        best=min(rows,key=lambda r:r['validation_mse_standardized'])
        assert best['epoch']==choice['best_rounds']
        assert len(rows)==choice['searched_rounds']==choice['best_rounds']+50
    else:
        h=json.loads((B/'artifacts'/f'{name}-validation-history.json').read_text())['validation']['rmse']
        assert int(np.argmin(h))+1==choice['best_rounds']
        assert len(h)==choice['searched_rounds']==choice['best_rounds']+100
    pred=np.load(B/'artifacts'/f'{name}-predictions.npz')
    for part in ['train','test']:
        ids=pred[part+'_row_ids'];pr=pred[part+'_predictions'];truth=y[ids]
        assert ids.tolist()==data['splits'][part] and np.isfinite(pr).all()
        mse=np.mean((truth-pr)**2);r2=1-mse/np.var(truth)
        slog=1-np.mean(np.log1p(((truth-pr)/mad)**2))/np.mean(np.log1p(((truth-med)/mad)**2))
        for key,value in [('mse',mse),('r2',r2),('slog',slog)]:
            assert abs(value-model['metrics'][part][key])<1e-10,(name,part,key,value)
print(json.dumps(dict(outcome='passed',source_hash_matched=True,split_disjoint=True,
    selection_before_test=True,best_validation_rounds_verified=True,refit_rounds_completed=True,
    prediction_metrics_independently_recomputed=True,models=list(s['models']),
    predictions_per_model=8192,bank_sha256=s['bank_sha256'])))
