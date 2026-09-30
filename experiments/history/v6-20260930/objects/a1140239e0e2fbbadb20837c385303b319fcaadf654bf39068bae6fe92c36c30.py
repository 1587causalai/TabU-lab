"""Continue two hosts for one hour with 1024-row windows and no replay."""
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
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    command=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',
             f"import runpy; runpy.run_path({str(BASE/script)!r},run_name='__main__')",*map(str,args)]
    with (BASE/log).open('w') as stream:
        subprocess.run(command,cwd=BASE/'source/src',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    config=json.loads((BASE/'queue.json').read_text())
    old=Path.home()/'experiments/v6-window1024-noreplay60m-20260928'
    parent_half='half2'
    previous=json.loads((old/'status.json').read_text())
    donor=json.loads((old/parent_half/'run/campaign.json').read_text())
    evaluation=json.loads((old/'evaluations'/parent_half/'terminal.json').read_text())
    assert previous['outcome']=='completed' and donor['outcome']=='training_completed'
    assert evaluation['outcome']=='completed' and evaluation['checkpoint_sha256']==donor['checkpoint_sha256']
    assert donor['identity']['supervision']==config['supervision']
    save(BASE/'before-eval.json',evaluation)
    status=dict(outcome='starting',started_utc=datetime.now(timezone.utc).isoformat(),
                start_update=donor['checkpoint_update'],start_successful_seconds=0,
                additional_seconds_target=3600,start_sha256=donor['checkpoint_sha256'],
                sampling='OpenML12 equal; at most 1024 training rows; no replay',supervision=config['supervision'],
                completed_training_seconds=0)
    save(BASE/'status.json',status)
    try:
        for half in ('half1','half2'):
            status.update(outcome='training',phase=half)
            save(BASE/'status.json',status)
            segment_seconds=1800 if half=='half1' else max(0.0,3600-status['completed_training_seconds'])
            execute('train_jointall.py',['--host',config['host'],'--device',config['device'],
                '--parent',donor['checkpoint'],'--parent-sha',donor['checkpoint_sha256'],
                '--manifest',donor['balanced_manifest'],'--supervision',config['supervision'],
                '--segment',half,'--seconds',str(segment_seconds)],f'{half}-train.log')
            campaign=json.loads((BASE/half/'run/campaign.json').read_text())
            assert campaign['outcome']=='training_completed' and campaign['successful_update_seconds']>=segment_seconds
            status['completed_training_seconds']+=campaign['successful_update_seconds']
            status.update(outcome='evaluating',additional_successful_seconds=status['completed_training_seconds'],
                          checkpoint_sha256=campaign['checkpoint_sha256'],checkpoint_update=campaign['checkpoint_update'])
            save(BASE/'status.json',status)
            execute('evaluate.py',['--manifest',campaign['balanced_manifest'],'--checkpoint',campaign['checkpoint'],
                '--expected-sha',campaign['checkpoint_sha256'],'--device',config['device'],
                '--bank-root',Path.home()/'experiments/openml12-frozen-icl-20260927',
                '--output',BASE/'evaluations'/half],f'{half}-eval.log')
            evaluation=json.loads((BASE/'evaluations'/half/'terminal.json').read_text())
            assert evaluation['outcome']=='completed' and evaluation['total_predictions']==7894
            status.setdefault('evaluations',{})[half]=dict(macro=evaluation['macro'],checkpoint_sha256=campaign['checkpoint_sha256'])
            save(BASE/'status.json',status)
            donor=campaign
        assert status['completed_training_seconds']>=3600
        status.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat())
        save(BASE/'status.json',status)
    except Exception as error:
        status.update(outcome='failed',error_type=type(error).__name__,error=str(error))
        save(BASE/'status.json',status)
        raise


if __name__=='__main__':
    main()
