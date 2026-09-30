"""Fresh fixed train-row Query bank, V6 default inference only; fit, not held-out generalization."""
import argparse,hashlib,json,math,statistics,sys,time
from pathlib import Path
import torch
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from dual import DualModel,prepare,score,V53LossConfig
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.curriculum_v53.artifacts import atomic_json,sha256
from tabu_lab.models.restoration._dtype import execution_dtype

def mean(x):return statistics.fmean(x) if x else None

def macro(rows):
 return {**{f'{branch}/{key}':mean([v[branch][key] for v in rows if v[branch].get(key) is not None]) for branch in ['v6'] for key in ['loss','r2','slog','accuracy','rank_mae']},'tables':len(rows),'numeric_tables':sum(v['kind']=='numeric' for v in rows),'nominal_tables':sum(v['kind']=='nominal' for v in rows),'ordinal_tables':sum(v['kind']=='ordinal' for v in rows),**{f'{branch}/{kind}_accuracy':mean([v[branch]['accuracy'] for v in rows if v['kind']==kind]) for branch in ['v6'] for kind in ['nominal','ordinal']}}

def main(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);report=dict(outcome='running',checkpoint_sha256=a.expected_sha,optimizer_updates=0,evaluation='718 known tables; 2 fixed training-row Query masks per table; single V6 target-only inference; fixed dual718 bank; not held-out generalization',tables={})
 atomic_json(out/'started.json',report)
 try:
  runtime=configure_runtime(a.device);cap=.75 if a.host=='gongqian-mini' else .5
  if a.device=='mps':torch.mps.set_per_process_memory_fraction(cap)
  else:torch.cuda.set_per_process_memory_fraction(cap)
  plan=load_v55_plan(B/'fit-manifest.json');assert sha256(a.checkpoint)==a.expected_sha
  payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=DualModel(plan.config).to(a.device,dtype=execution_dtype(a.device));model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
  report.update(runtime=runtime,checkpoint_update=payload['state']['update'],parent_supervision=payload['identity']['supervision']);del payload
  config=V53LossConfig(**plan.spec['stages'][0]['loss']);seeds=dict(plan.spec['seeds']);seeds['evaluation']=int.from_bytes(hashlib.sha256(b'dual718-fixed-fit-bank-20260928').digest()[:8],'little');traces=[];total=0
  with torch.no_grad():
   for t in plan.tables:
    target=t.target_column;schema=t.schema[target];family='sparse100' if t.name.startswith('sparse_') else 'old618';row=dict(kind=schema.kind,family=family);addresses={b:{} for b in ['v6']};losses={b:[] for b in addresses}
    train=t.values[target].tolist();med=statistics.median(train);mad=statistics.median(abs(x-med) for x in train) if schema.kind=='numeric' else 0
    ranks={int(v):i/max(schema.domain_size-1,1) for i,v in enumerate(schema.order or range(schema.domain_size))} if schema.kind=='ordinal' else None
    for index in range(2):
     inputs,request,truth,info=build_episode(t,plan.spec['stages'][0]['recipe'][t.kind],index,seeds,a.device,evaluation=True,partition='train',epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
     prepared=prepare(model,inputs,request,truth,config)
     traces.append(dict(table=t.name,index=index,query_addresses=info['query_addresses'],row_ids=info['row_ids'],mask_seed=info['mask_seed'],code_seed=info['code_seed'],window_seed=info['window_seed']))
     total+=len(prepared.visible.request.targets)
     for branch in addresses:
      s=score(model,prepared,config,branch,decode=True);losses[branch].append(float(s.loss));assert len(s.output.columns)==1
      c=s.output.columns[0];assert c.result.status=='ok' and c.decoded is not None
      local=s.output.request.targets[c.target_indices,0];ys=truth.values[target][local].cpu().tolist();pred=c.decoded.cpu().tolist()
      for i,y,v in zip(local.cpu().tolist(),ys,pred,strict=True):
       assert math.isfinite(v)
       addr=info['row_ids'][i];x=addresses[branch].setdefault(addr,dict(y=y,n=0,sse=0.,correct=0.,rank=0.,logerr=0.));x['n']+=1;x['sse']+=(v-y)**2;x['correct']+=int(v==y)
       if ranks is not None:x['rank']+=abs(ranks[int(v)]-ranks[int(y)])
       if mad>1e-12:x['logerr']+=math.log1p(((v-y)/mad)**2)
      del s,c
     del inputs,request,truth,prepared
    for branch,items in addresses.items():
     xs=list(items.values());m=dict(loss=mean(losses[branch]),unique_query_rows=len(xs),query_exposures=sum(x['n'] for x in xs))
     if schema.kind=='numeric':
      center=mean([x['y'] for x in xs]);var=mean([(x['y']-center)**2 for x in xs]);mse=mean([x['sse']/x['n'] for x in xs]);den=mean([math.log1p(((x['y']-med)/mad)**2) for x in xs]) if mad>1e-12 else 0
      m.update(r2=1-mse/var if var>0 else None,slog=1-mean([x['logerr']/x['n'] for x in xs])/den if den>0 else None)
      if m['slog'] is None:m['slog_undefined_reason']='zero training MAD or fixed-query baseline variance'
     else:
      m['accuracy']=mean([x['correct']/x['n'] for x in xs])
      if ranks is not None:m['rank_mae']=mean([x['rank']/x['n'] for x in xs])
     row[branch]=m
    report['tables'][t.name]=row
    atomic_json(out/'progress.json',dict(outcome='running',completed_tables=len(report['tables']),total_tables=718))
    if len(report['tables'])%100==0:print(json.dumps(dict(tables=len(report['tables']))),flush=True)
  bank_digest=hashlib.sha256(json.dumps(traces,sort_keys=True).encode()).hexdigest();bank=B/'fixed-fit-bank.json'
  if bank.exists():assert json.loads(bank.read_text())['sha256']==bank_digest
  else:atomic_json(bank,dict(sha256=bank_digest,tables=718,masks=1436,traces=traces))
  assert len(report['tables'])==718
  assert total==97648
  rows=list(report['tables'].values());report.update(outcome='completed',macro=macro(rows),by_family={f:macro([v for v in rows if v['family']==f]) for f in ['old618','sparse100']},bank_sha256=bank_digest,total_predictions_per_branch=total,total_masks=1436)
 except Exception as e:
  report.update(outcome='failed',error_type=type(e).__name__,error=str(e));atomic_json(out/'terminal.json',report);raise
 atomic_json(out/'terminal.json',report);print(json.dumps(dict(outcome='completed',macro=report['macro'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['host','device','checkpoint','expected-sha','output']:p.add_argument('--'+k,required=True)
 main(p.parse_args())
