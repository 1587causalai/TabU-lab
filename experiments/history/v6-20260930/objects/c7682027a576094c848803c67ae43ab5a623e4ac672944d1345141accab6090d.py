"""Run a balanced 30-minute stage, then evaluate all twelve complete test splits."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BASE=Path(__file__).resolve().parent


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def execute(script,args,log):
    config=json.loads((BASE/'host.json').read_text())
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    command=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',
             f"import runpy; runpy.run_path({str(BASE/script)!r},run_name='__main__')",*map(str,args)]
    with (BASE/log).open('w') as stream:
        subprocess.run(command,cwd=BASE/'source/src',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    queue=json.loads((BASE/'queue.json').read_text())
    prerequisite = Path.home()/'experiments/v6-broadcast-balanced30m-20260927'
    save(BASE/'status.json', dict(outcome='waiting_for_balanced_stage',
         prerequisite=str(prerequisite), queued_utc=datetime.now(timezone.utc).isoformat(),
         additional_seconds_target=1800, start_successful_seconds=0))
    while True:
        current=json.loads((prerequisite/'status.json').read_text())
        if current['outcome']=='failed':
            save(BASE/'status.json',dict(outcome='failed',error='prerequisite balanced stage failed; joint-all not started'))
            return
        if current['outcome']=='completed':
            break
        time.sleep(10)
    donor=json.loads((prerequisite/'main/run/campaign.json').read_text())
    evaluation=json.loads((prerequisite/'evaluations/main/terminal.json').read_text())
    assert donor['outcome']=='training_completed' and donor['successful_update_seconds']>=1800
    assert evaluation['outcome']=='completed' and evaluation['checkpoint_sha256']==donor['checkpoint_sha256']
    config=dict(host=queue['host'],device=queue['device'],supervision=queue['supervision'],parent=donor['checkpoint'],
                parent_sha=donor['checkpoint_sha256'],parent_update=donor['checkpoint_update'],
                manifest=donor['balanced_manifest'])
    save(BASE/'host.json',config)
    save(BASE/'before-eval.json',evaluation)
    status=dict(outcome='training',phase='main',started_utc=datetime.now(timezone.utc).isoformat(),
                start_update=config['parent_update'],start_successful_seconds=0,
                additional_seconds_target=1800,start_sha256=config['parent_sha'],
                sampling='OpenML12 equal:12 + history618:3 per 15 updates',supervision=config['supervision'])
    save(BASE/'status.json',status)
    try:
        execute('train_jointall.py',['--host',config['host'],'--device',config['device'],
            '--parent',config['parent'],'--parent-sha',config['parent_sha'],
            '--manifest',config['manifest'],'--supervision',config['supervision']],'train.log')
        campaign=json.loads((BASE/'main/run/campaign.json').read_text())
        assert campaign['outcome']=='training_completed' and campaign['successful_update_seconds']>=1800
        status.update(outcome='evaluating',additional_successful_seconds=campaign['successful_update_seconds'],
                      checkpoint_sha256=campaign['checkpoint_sha256'],checkpoint_update=campaign['checkpoint_update'])
        save(BASE/'status.json',status)
        execute('evaluate.py',['--manifest',campaign['balanced_manifest'],'--checkpoint',campaign['checkpoint'],
            '--expected-sha',campaign['checkpoint_sha256'],'--device',config['device'],
            '--bank-root',Path.home()/'experiments/openml12-frozen-icl-20260927',
            '--output',BASE/'evaluations/main'],'eval.log')
        evaluation=json.loads((BASE/'evaluations/main/terminal.json').read_text())
        assert evaluation['outcome']=='completed' and evaluation['total_predictions']==7894
        status.update(outcome='completed',macro=evaluation['macro'],completed_utc=datetime.now(timezone.utc).isoformat())
        save(BASE/'status.json',status)
    except Exception as error:
        status.update(outcome='failed',error_type=type(error).__name__,error=str(error))
        save(BASE/'status.json',status)
        raise


if __name__=='__main__':
    main()
