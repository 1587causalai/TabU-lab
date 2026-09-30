"""One launch, 900 + remaining successful seconds, frozen OpenML12 evaluation."""
import pathlib, json, os, subprocess
from datetime import datetime, timezone
B = pathlib.Path(__file__).resolve().parent

def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temporary, path)

def execute(script, args, log):
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'PYTORCH_ENABLE_MPS_FALLBACK': '0', 'PYTORCH_MPS_FAST_MATH': '0'}
    cmd = [str(pathlib.Path.home()/'.local/bin/wehub-python'), '--profile',
           'train-20260920', '-u', '-c',
           f"import runpy;runpy.run_path({str(B/script)!r},run_name='__main__')",
           *map(str, args)]
    with (B/log).open('w') as stream:
        subprocess.run(cmd, cwd=B/'source/src', env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=14400)

def main():
    q = json.loads((B/'queue.json').read_text())
    h = q['host']; donor = q['parent']
    s = dict(outcome='starting', host=h, started_utc=datetime.now(timezone.utc).isoformat(),
             completed_training_seconds=0, additional_seconds_target=1800,
             start_successful_seconds=0, start_sha256=donor['checkpoint_sha256'],
             objective='broadcast_allcell_restoration')
    save(B/'status.json', s)
    try:
        before = json.loads((B/'before-eval.json').read_text())
        assert before['outcome']=='completed' and before['total_predictions']==7894
        assert before['checkpoint_sha256']==donor['checkpoint_sha256']
        assert before['evaluation_supervision']=='target_only'
        for phase in ['half1', 'half2']:
            budget = 900 if phase=='half1' else 1800-s['completed_training_seconds']
            assert budget > 0
            s.update(outcome='training', phase=phase, phase_budget=budget)
            save(B/'status.json', s)
            extra = ['--reset-sampling-cursor'] if phase=='half1' else []
            execute('train.py', ['--host',h,'--device',q['device'],
                '--parent',donor['checkpoint'],'--parent-sha',donor['checkpoint_sha256'],
                '--manifest',donor.get('balanced_manifest',donor.get('manifest')),
                '--supervision','target_only','--segment',phase,'--seconds',budget]+extra,
                phase+'-train.log')
            donor = json.loads((B/phase/'run/campaign.json').read_text())
            assert donor['outcome']=='training_completed'
            s['completed_training_seconds'] += donor['successful_update_seconds']
            s.update(outcome='evaluating', checkpoint_sha256=donor['checkpoint_sha256'])
            save(B/'status.json',s)
            execute('evaluate.py',['--manifest',donor['balanced_manifest'],
                '--checkpoint',donor['checkpoint'],'--expected-sha',donor['checkpoint_sha256'],
                '--device',q['device'],'--bank-root',B.parent/'openml12-frozen-icl-20260927',
                '--output',B/'evaluations'/phase,'--memory-fraction',.75 if h=='gongqian-mini' else .5],
                phase+'-eval.log')
            ev=json.loads((B/'evaluations'/phase/'terminal.json').read_text())
            assert ev['outcome']=='completed' and ev['total_predictions']==7894
            assert ev['optimizer_updates']==0 and ev['checkpoint_sha256']==donor['checkpoint_sha256']
            assert ev['bank_sha256']==before['bank_sha256']
            s.setdefault('evaluations',{})[phase]=dict(macro=ev['macro'],checkpoint_sha256=ev['checkpoint_sha256'])
            save(B/'status.json',s)
        assert 1800 <= s['completed_training_seconds'] < 1900
        s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat())
        save(B/'status.json',s)
    except Exception as error:
        s.update(outcome='failed',error_type=type(error).__name__,error=str(error))
        save(B/'status.json',s)
        raise

if __name__=='__main__':
    main()
