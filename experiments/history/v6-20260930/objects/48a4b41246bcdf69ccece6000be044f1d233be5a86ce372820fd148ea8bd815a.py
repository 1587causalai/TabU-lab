"""Read-only paired checkpoint evaluation; no optimizer or training entrypoint."""
import pathlib,json,os,subprocess,time,hashlib
from datetime import datetime,timezone
B=pathlib.Path(__file__).resolve().parent;E=B.parent

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 q=json.loads((B/'queue.json').read_text());h=q['host'];device='cuda:0' if h=='dgx2' else 'mps';cur=E/'v6-v55-dual718-continue60m-20260928';old=E/'v6-openml12-recover30m-20260928';syn=E/'v6-sparse-relevance100-20260928'
 s=dict(outcome='preparing',host=h,optimizer_updates=0,started_utc=datetime.now(timezone.utc).isoformat(),completed={});save(B/'status.json',s)
 try:
  assert json.loads((cur/'status.json').read_text())['outcome']=='completed'
  ck={k:json.loads((r/phase/'run/campaign.json').read_text()) for k,r,phase in [('parent',old,'half2'),('middle',cur,'half1'),('final',cur,'half2')]}
  for c in ck.values():assert c['outcome']=='training_completed' and sha(pathlib.Path(c['checkpoint']))==c['checkpoint_sha256']
  source_checks={}
  for model in ['restoration_v6/model.py','restoration_v55/model.py','restoration_v53/model.py']:
   paths=[r/'source/src/tabu_lab/models'/model for r in [cur,old,syn]];hashes=[sha(p) for p in paths];assert len(set(hashes))==1,(model,hashes);source_checks[model]=hashes[0]
  p=json.loads((old/'evaluations/half2/terminal.json').read_text());assert p['outcome']=='completed' and p['checkpoint_sha256']==ck['parent']['checkpoint_sha256'] and p['total_predictions']==7894
  assert p['bank_sha256']==sha(E/'openml12-frozen-icl-20260927/bank.json')
  save(B/'historical-parent-openml12.json',p);save(B/'checkpoints.json',ck);save(B/'source-checks.json',source_checks)
  jobs=[('parent','openml12'),('final','openml12'),('middle','openml12'),('parent','sparse100'),('middle','sparse100'),('final','sparse100')]
  s['historical_parent_openml12']=dict(macro=p['macro'],checkpoint_sha256=p['checkpoint_sha256'],source=str(old/'evaluations/half2/terminal.json'))
  for node,dataset in jobs:
   c=ck[node];out=B/'evaluations'/node/dataset;s.update(outcome='evaluating',node=node,dataset=dataset);save(B/'status.json',s)
   script=B/'eval_openml.py' if dataset=='openml12' else B/'eval_sparse.py'
   args=['--manifest',c['balanced_manifest'],'--checkpoint',c['checkpoint'],'--expected-sha',c['checkpoint_sha256'],'--device',device,'--output',str(out)]
   if dataset=='openml12':args+=['--bank-root',str(E/'openml12-frozen-icl-20260927'),'--memory-fraction','0.75' if h=='gongqian-mini' else '0.5']
   else:args+=['--host',h]
   cmd=[str(pathlib.Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',f"import runpy;runpy.run_path({str(script)!r},run_name='__main__')",*args]
   env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
   start=time.monotonic()
   with (B/f'{node}-{dataset}.log').open('w') as f:subprocess.run(cmd,cwd=cur/'source/src',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
   ev=json.loads((out/'terminal.json').read_text());assert ev['outcome']=='completed' and ev['optimizer_updates']==0 and ev['checkpoint_sha256']==c['checkpoint_sha256']
   assert ev['total_predictions']==(7894 if dataset=='openml12' else 54400)
   assert sha(pathlib.Path(c['checkpoint']))==c['checkpoint_sha256']
   s['completed'][node+'/'+dataset]=dict(macro=ev['macro'],checkpoint_sha256=ev['checkpoint_sha256'],seconds=time.monotonic()-start,bank_sha256=ev['bank_sha256']);save(B/'status.json',s)
  s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(B/'status.json',s)
 except Exception as e:
  s.update(outcome='failed',error_type=type(e).__name__,error=str(e));save(B/'status.json',s);raise
if __name__=='__main__':main()
