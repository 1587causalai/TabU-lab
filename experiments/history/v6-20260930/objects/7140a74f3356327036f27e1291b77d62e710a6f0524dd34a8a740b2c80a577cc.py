import json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent

def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)

def execute(script,args,log):
 env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
 cmd=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',f"import runpy; runpy.run_path({str(BASE/script)!r},run_name='__main__')",*map(str,args)]
 with (BASE/log).open('w') as stream:
  subprocess.run(cmd,cwd=BASE/'source/src',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=2400)

def main():
 q=json.loads((BASE/'queue.json').read_text());old=Path(q['parent_root']);donor=json.loads((old/q['parent_phase']/'run/campaign.json').read_text());ev=json.loads((old/'evaluations'/q['parent_phase']/'terminal.json').read_text())
 assert donor['outcome']=='training_completed' and ev['outcome']=='completed' and ev['total_predictions']==7894
 assert donor['checkpoint_sha256']==ev['checkpoint_sha256']==q['expected_parent_sha']
 assert donor['identity']['supervision']==q['supervision']
 save(BASE/'before-eval.json',ev)
 s=dict(outcome='starting',started_utc=datetime.now(timezone.utc).isoformat(),start_update=donor['checkpoint_update'],start_successful_seconds=0,additional_seconds_target=1800,start_sha256=donor['checkpoint_sha256'],sampling='OpenML12 once plus 8 old618 per20; OpenML window1024',supervision=q['supervision'],completed_training_seconds=0)
 save(BASE/'status.json',s)
 try:
  for phase in ['half1','half2']:
   budget=900 if phase=='half1' else max(0,1800-s['completed_training_seconds'])
   if budget<=0:raise RuntimeError('first segment consumed full budget; no automatic extra updates')
   s.update(outcome='training',phase=phase,phase_budget=budget);save(BASE/'status.json',s)
   execute('train_jointall.py',['--host',q['host'],'--device',q['device'],'--parent',donor['checkpoint'],'--parent-sha',donor['checkpoint_sha256'],'--manifest',donor['balanced_manifest'],'--supervision',q['supervision'],'--segment',phase,'--seconds',budget]+(['--reset-sampling-cursor'] if phase=='half1' else []),phase+'-train.log')
   campaign=json.loads((BASE/phase/'run/campaign.json').read_text());assert campaign['outcome']=='training_completed'
   s['completed_training_seconds']+=campaign['successful_update_seconds']
   s.update(outcome='evaluating',additional_successful_seconds=s['completed_training_seconds'],checkpoint_sha256=campaign['checkpoint_sha256'],checkpoint_update=campaign['checkpoint_update']);save(BASE/'status.json',s)
   execute('evaluate.py',['--manifest',campaign['balanced_manifest'],'--checkpoint',campaign['checkpoint'],'--expected-sha',campaign['checkpoint_sha256'],'--device',q['device'],'--bank-root',Path.home()/'experiments/openml12-frozen-icl-20260927','--output',BASE/'evaluations'/phase],phase+'-eval.log')
   ev=json.loads((BASE/'evaluations'/phase/'terminal.json').read_text());assert ev['outcome']=='completed' and ev['total_predictions']==7894
   s.setdefault('evaluations',{})[phase]={'macro':ev['macro'],'checkpoint_sha256':ev['checkpoint_sha256']};save(BASE/'status.json',s);donor=campaign
  s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(BASE/'status.json',s)
 except Exception as e:
  s.update(outcome='failed',error_type=type(e).__name__,error=str(e));save(BASE/'status.json',s);raise

if __name__=='__main__':main()
