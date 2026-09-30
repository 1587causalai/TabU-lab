import json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent

def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def execute(script,args,log):
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    cmd=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',f"import runpy; runpy.run_path({str(BASE/script)!r},run_name='__main__')",*map(str,args)]
    with (BASE/log).open('w') as stream:subprocess.run(cmd,cwd=BASE/'source/src',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
def main():
    q=json.loads((BASE/'queue.json').read_text());old=Path(q['parent_root']);donor=json.loads((old/'half2/run/campaign.json').read_text())
    s=dict(outcome='starting',started_utc=datetime.now(timezone.utc).isoformat(),additional_seconds_target=3600,completed_training_seconds=0,start_update=donor['checkpoint_update'],start_successful_seconds=0,start_sha256=donor['checkpoint_sha256'],supervision=q['supervision'])
    save(BASE/'status.json',s)
    try:
        assert donor['outcome']=='training_completed' and donor['checkpoint_sha256']==q['expected_parent_sha'] and donor['identity']['supervision']==q['supervision']
        save(BASE/'parent-receipt.json',donor)
        def evaluate(c,phase):
            execute('evaluate.py',['--host',q['host'],'--device',q['device'],'--manifest',c['balanced_manifest'],'--checkpoint',c['checkpoint'],'--expected-sha',c['checkpoint_sha256'],'--output',BASE/'evaluations'/phase],phase+'-eval.log')
            ev=json.loads((BASE/'evaluations'/phase/'terminal.json').read_text());assert ev['outcome']=='completed' and ev['total_predictions']==54400
            return ev
        s.update(outcome='evaluating',phase='before');save(BASE/'status.json',s)
        ev=evaluate(donor,'before');save(BASE/'before-eval.json',ev)
        for phase in ['half1','half2']:
            budget=1800 if phase=='half1' else 3600-s['completed_training_seconds']
            assert budget>0
            s.update(outcome='training',phase=phase,phase_budget=budget);save(BASE/'status.json',s)
            execute('train_jointall.py',['--host',q['host'],'--device',q['device'],'--parent',donor['checkpoint'],'--parent-sha',donor['checkpoint_sha256'],'--manifest',donor['balanced_manifest'],'--supervision',q['supervision'],'--segment',phase,'--seconds',budget]+(['--reset-sampling-cursor'] if phase=='half1' else []),phase+'-train.log')
            c=json.loads((BASE/phase/'run/campaign.json').read_text());assert c['outcome']=='training_completed'
            s['completed_training_seconds']+=c['successful_update_seconds'];s.update(outcome='evaluating',checkpoint_sha256=c['checkpoint_sha256'],checkpoint_update=c['checkpoint_update']);save(BASE/'status.json',s)
            ev=evaluate(c,phase);s.setdefault('evaluations',{})[phase]=dict(macro=ev['macro'],checkpoint_sha256=ev['checkpoint_sha256']);save(BASE/'status.json',s);donor=c
        s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(BASE/'status.json',s)
    except Exception as e:
        s.update(outcome='failed',error_type=type(e).__name__,error=str(e));save(BASE/'status.json',s);raise
if __name__=='__main__':main()
