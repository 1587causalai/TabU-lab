import json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parent

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def run(script,args,log):
 env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
 cmd=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',f"import runpy;runpy.run_path({str(B/script)!r},run_name='__main__')",*map(str,args)]
 with (B/log).open('w') as f:subprocess.run(cmd,cwd=B/'source/src',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=14400)
def main():
 q=json.loads((B/'queue.json').read_text());init=json.loads((B/'initial.json').read_text());s=dict(outcome='starting',phase='before',completed_training_seconds=0.,additional_seconds_target=1800,started_utc=datetime.now(timezone.utc).isoformat(),start_sha256=init['parent_sha256']);save(B/'status.json',s)
 try:
  parent=init['parent'];sha=init['parent_sha256']
  for phase in ['before','half1','half2']:
   if phase!='before':
    budget=900 if phase=='half1' else 1800-s['completed_training_seconds'];assert budget>0
    s.update(outcome='training',phase=phase);save(B/'status.json',s)
    run('train.py',['--host',q['host'],'--device',q['device'],'--segment',phase,'--parent',parent,'--parent-sha',sha,'--seconds',budget],phase+'-train.log')
    c=json.loads((B/phase/'run/campaign.json').read_text());assert c['outcome']=='training_completed'
    s['completed_training_seconds']+=c['successful_update_seconds'];parent=c['checkpoint'];sha=c['checkpoint_sha256']
   s.update(outcome='evaluating',phase=phase,checkpoint_sha256=sha);save(B/'status.json',s)
   run('evaluate.py',['--host',q['host'],'--device',q['device'],'--checkpoint',parent,'--expected-sha',sha,'--output',B/'evaluations'/phase],phase+'-eval.log')
   e=json.loads((B/'evaluations'/phase/'terminal.json').read_text());assert e['outcome']=='completed' and len(e['tables'])==718
   if phase=='before':save(B/'before-eval.json',e)
   s.setdefault('evaluations',{})[phase]=dict(macro=e['macro'],by_family=e['by_family'],bank_sha256=e['bank_sha256'],checkpoint_sha256=sha);save(B/'status.json',s)
  s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(B/'status.json',s)
 except Exception as e:
  s.update(outcome='failed',error=str(e),error_type=type(e).__name__);save(B/'status.json',s);raise
if __name__=='__main__':main()
