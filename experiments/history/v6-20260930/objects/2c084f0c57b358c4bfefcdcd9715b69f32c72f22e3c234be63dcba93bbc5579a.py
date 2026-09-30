import pathlib,json,os,subprocess,time
from datetime import datetime,timezone
B=pathlib.Path(__file__).resolve().parent

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def execute(script,args,log):
 env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
 cmd=[str(pathlib.Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',f"import runpy;runpy.run_path({str(B/script)!r},run_name='__main__')",*map(str,args)]
 with (B/log).open('w') as f:subprocess.run(cmd,cwd=B/'source/src',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=14400)
def main():
 q=json.loads((B/'queue.json').read_text());h=q['host'];e=B.parent;prior=e/'v6-v55-dual718-continue60m-20260928';template_root=e/'v6-openml12-recover30m-20260928';check=e/'v6-dual718-checkpoint-test-20260928'
 s=dict(outcome='waiting_for_evaluation',started_utc=datetime.now(timezone.utc).isoformat(),completed_training_seconds=0,additional_seconds_target=3600,host=h,supervision=q['supervision']);save(B/'status.json',s)
 try:
  while True:
   c=json.loads((check/'status.json').read_text())
   if c['outcome']=='completed':break
   if c['outcome']=='failed':raise RuntimeError('preceding checkpoint test failed; training not started')
   time.sleep(20)
  donor=json.loads((prior/'half2/run/campaign.json').read_text());template=json.loads((template_root/'half2/run/campaign.json').read_text())
  assert donor['outcome']=='training_completed' and donor['checkpoint_sha256']==q['expected_parent_sha']
  assert template['identity']['supervision']==q['supervision']
  for filename,path in [('before-eval.json',check/'evaluations/final/openml12/terminal.json'),('original-parent-eval.json',check/'evaluations/parent/openml12/terminal.json'),('before-sparse-eval.json',check/'evaluations/final/sparse100/terminal.json'),('original-parent-sparse-eval.json',check/'evaluations/parent/sparse100/terminal.json')]:
   ev=json.loads(path.read_text());assert ev['outcome']=='completed';save(B/filename,ev)
  assert json.loads((B/'before-eval.json').read_text())['checkpoint_sha256']==donor['checkpoint_sha256']
  s.update(start_sha256=donor['checkpoint_sha256'],start_update=donor['checkpoint_update'],start_successful_seconds=0,sampling='12 OpenML per20 plus8 equal-table history718 replay; window512');save(B/'status.json',s)
  for phase in ['half1','half2']:
   budget=1800 if phase=='half1' else 3600-s['completed_training_seconds'];assert budget>0
   s.update(outcome='training',phase=phase,phase_budget=budget);save(B/'status.json',s)
   extra=(['--reset-sampling-cursor','--openml-template-manifest',template['balanced_manifest'],'--openml-template-checkpoint',template['checkpoint'],'--openml-template-sha',template['checkpoint_sha256']] if phase=='half1' else [])
   execute('train.py',['--host',h,'--device',q['device'],'--parent',donor['checkpoint'],'--parent-sha',donor['checkpoint_sha256'],'--manifest',donor['balanced_manifest'],'--supervision',q['supervision'],'--segment',phase,'--seconds',budget]+extra,phase+'-train.log')
   c=json.loads((B/phase/'run/campaign.json').read_text());assert c['outcome']=='training_completed'
   s['completed_training_seconds']+=c['successful_update_seconds'];s.update(outcome='evaluating',checkpoint_sha256=c['checkpoint_sha256'],checkpoint_update=c['checkpoint_update']);save(B/'status.json',s)
   execute('evaluate.py',['--manifest',c['balanced_manifest'],'--checkpoint',c['checkpoint'],'--expected-sha',c['checkpoint_sha256'],'--device',q['device'],'--bank-root',e/'openml12-frozen-icl-20260927','--output',B/'evaluations'/phase,'--memory-fraction',.75 if h=='gongqian-mini' else .5],phase+'-eval.log')
   ev=json.loads((B/'evaluations'/phase/'terminal.json').read_text());assert ev['outcome']=='completed' and ev['total_predictions']==7894 and ev['optimizer_updates']==0 and ev['checkpoint_sha256']==c['checkpoint_sha256']
   s.setdefault('evaluations',{})[phase]=dict(macro=ev['macro'],checkpoint_sha256=ev['checkpoint_sha256']);save(B/'status.json',s);donor=c
  assert 3600<=s['completed_training_seconds']<3650
  s.update(outcome='evaluating_sparse100',phase='terminal');save(B/'status.json',s)
  execute('eval_sparse.py',['--host',h,'--device',q['device'],'--manifest',donor['balanced_manifest'],'--checkpoint',donor['checkpoint'],'--expected-sha',donor['checkpoint_sha256'],'--output',B/'evaluations/sparse-final'],'sparse-final-eval.log')
  ev=json.loads((B/'evaluations/sparse-final/terminal.json').read_text());assert ev['outcome']=='completed' and ev['total_predictions']==54400 and ev['checkpoint_sha256']==donor['checkpoint_sha256']
  s['evaluations']['sparse-final']=dict(macro=ev['macro'],checkpoint_sha256=ev['checkpoint_sha256']);s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(B/'status.json',s)
 except Exception as ex:s.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save(B/'status.json',s);raise
if __name__=='__main__':main()
