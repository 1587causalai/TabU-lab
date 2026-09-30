from pathlib import Path
import copy, hashlib, json, math, os, statistics
ROOT=Path(__file__).resolve().parent
PARENT_ROOT=Path('/home/cms/experiments/tabu-v55-tri100-r2-90min-dgx2-20260926')
PARENT=PARENT_ROOT/'runs/stage-03-numeric/checkpoint-progress.pt'
PARENT_SHA='1c909b761323c24b7f770964958350e388c573f67a349e2df914013eb9dfed26'
PARENT_EVAL=PARENT_ROOT/'evaluations/stage-03-numeric-final/terminal.json'
MANIFEST=ROOT/'manifests/joint618.json'
URL='https://wandb.ai/zj3712/restoration-v55-single-dgp/runs/v55-joint618-120m-dgx2-1c909b76'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def probe_digest(value,names=None):
    p=copy.deepcopy(value['probes'])
    if names is not None:p={k:p[k] for k in names}
    for x in p.values():assert x['complete'];x.pop('seconds',None)
    return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rows(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n')]
def quantile(values,f):
    v=sorted(values);z=(len(v)-1)*f;i=math.floor(z)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(z-i)
def dist(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),p10=quantile(values,.1),p95=quantile(values,.95),min=min(values),max=max(values),table_count=len(values))
def typed(value):
    kinds={x['id']:x['target_kind'] for x in json.loads((ROOT/'evidence/inventory.json').read_text())['tables']}
    out={}
    for probe,block in value['probes'].items():
        assert block['complete']
        for row in block['by_table']:
            kind=kinds[row['table']]
            for metric in ('query_numeric_r2','query_discrete_accuracy','query_ordinal_rank_mae'):
                v=row['metrics'].get(metric)
                if isinstance(v,(int,float)) and math.isfinite(v):out.setdefault(f'{probe}/{kind}/{metric}',{})[row['table']]=v
    return out
def paired(initial,final):
    a,b=typed(initial),typed(final);assert a.keys()==b.keys();result={}
    for key,av in a.items():
        bv=b[key];assert av.keys()==bv.keys();direction=-1 if key.endswith('mae') else 1
        delta={t:bv[t]-v for t,v in av.items()};scores=[v*direction for v in delta.values()]
        result[key]={'start':dist(list(av.values())),'final':dist(list(bv.values())),'delta':dist(list(delta.values())),'better_tie_worse':[sum(v>1e-12 for v in scores),sum(abs(v)<=1e-12 for v in scores),sum(v< -1e-12 for v in scores)],'by_table':{t:{'start':av[t],'final':bv[t],'delta':delta[t]} for t in av}}
    return result
