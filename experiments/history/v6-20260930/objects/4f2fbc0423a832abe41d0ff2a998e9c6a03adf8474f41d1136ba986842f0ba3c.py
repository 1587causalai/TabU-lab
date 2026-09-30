"""Fixed Puma data and masked-train/test evaluation. No test-label training."""
import hashlib
import json
import math
import random
from pathlib import Path
import torch
from tabu_lab.curriculum_v53.artifacts import atomic_json, sha256
from tabu_lab.models.restoration._dtype import execution_dtype
from frozen_evaluate import make_input, metrics, state_hash
from evaluate import score_log

B = Path(__file__).resolve().parent
NAME = 'pumadyn32nh'

def load_data():
    spec = json.loads((B/'single-manifest.json').read_text())
    ent = spec['tables'][0]
    train_path = (B/ent['path']).resolve()
    assert sha256(train_path) == ent['sha256']
    train_data = json.loads(train_path.read_text())
    bank_root = B.parent/'openml12-frozen-icl-20260927'
    ent_bank = next(e for e in json.loads((bank_root/'bank.json').read_text())['tables'] if e['name']==NAME)
    data_path = bank_root/ent_bank['path']
    assert sha256(data_path)==ent_bank['sha256']
    data = json.loads(data_path.read_text())
    assert train_data['values']==data['values'], 'training/evaluation values differ'
    assert train_data['splits']['train']==data['splits']['train']
    assert train_data['splits']['test']==data['splits']['test']
    train, test = data['splits']['train'], data['splits']['test']
    assert len(train)==6553 and len(test)==1639 and not set(train)&set(test)
    bank=json.loads((B/'fit-bank.json').read_text())
    assert bank['data_sha256']==ent_bank['sha256']
    queries=[i for e in bank['train_traces'] for i in e['query_row_ids']]
    assert len(queries)==len(set(queries))==6553 and set(queries)==set(train)
    for e in bank['train_traces']:
        assert not set(e['context_row_ids'])&set(e['query_row_ids'])
        assert set(e['context_row_ids'])|set(e['query_row_ids'])==set(train)
    assert bank['test_trace']['context_row_ids']==train
    assert bank['test_trace']['query_row_ids']==test
    return data,bank

def metric_rows(rows, data):
    m=metrics(rows)
    m['mse']=m['rmse']**2
    m['slog']=score_log(rows,[float(data['values'][i][-1]) for i in data['splits']['train']])
    targets=torch.tensor([r['target'] for r in rows],dtype=torch.float64)
    preds=torch.tensor([r['prediction'] for r in rows],dtype=torch.float64)
    m['prediction_std']=float(preds.std(unbiased=False))
    m['target_std']=float(targets.std(unbiased=False))
    return m

def evaluate_model(model,device,label,checkpoint_sha,data,bank):
    out=B/'evaluations'/label
    out.mkdir(parents=True,exist_ok=False)
    initial=state_hash(model)
    report=dict(outcome='running',checkpoint_sha256=checkpoint_sha,branch=model.branch,
                bank_sha256=sha256(B/'fit-bank.json'),optimizer_updates=0,
                train_fit_scope='all training rows; target hidden in each query; previously seen in parameter training',
                test_scope='original heldout test rows; all original train support',metrics={})
    atomic_json(out/'started.json',report)
    try:
        model.eval()
        for group,traces in [('train',bank['train_traces']),('test',[bank['test_trace']])]:
            rows=[]
            for trace in traces:
                inputs,request=make_input(data,trace,NAME,device,execution_dtype(device))
                q=len(trace['context_row_ids'])
                assert not bool(inputs.visible[q:,-1].any())
                assert bool((inputs.values[-1][q:]==0).all())
                with torch.no_grad():
                    result=model(inputs,request)
                assert len(result.columns)==1 and result.columns[0].result.status=='ok'
                col=result.columns[0]
                pred=col.decoded.detach().cpu().tolist()
                address=request.targets.detach().cpu().tolist()
                pos=col.target_indices.detach().cpu().tolist()
                for v,p in zip(pred,pos,strict=True):
                    row,c=address[p]
                    assert c==32 and row>=q and math.isfinite(v)
                    rid=trace['query_row_ids'][row-q]
                    rows.append(dict(row_id=rid,target=float(data['values'][rid][-1]),prediction=float(v)))
                del inputs,request,result,col
                if device=='mps': torch.mps.synchronize(); torch.mps.empty_cache()
                else: torch.cuda.synchronize(); torch.cuda.empty_cache()
            expected=data['splits'][group]
            assert len(rows)==len(expected) and {r['row_id'] for r in rows}==set(expected)
            rows.sort(key=lambda r:r['row_id'])
            atomic_json(out/(group+'-predictions.json'),rows)
            report['metrics'][group]=metric_rows(rows,data)
            atomic_json(out/'progress.json',report)
        assert state_hash(model)==initial
        report.update(outcome='completed',model_state_unchanged=True,total_predictions=8192)
    except Exception as ex:
        report.update(outcome='failed',error_type=type(ex).__name__,error=str(ex))
        atomic_json(out/'terminal.json',report)
        raise
    atomic_json(out/'terminal.json',report)
    return report
