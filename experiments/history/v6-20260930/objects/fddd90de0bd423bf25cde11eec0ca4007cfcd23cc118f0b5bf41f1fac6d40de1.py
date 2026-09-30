"""Resume the frozen broadcast-only implementation for one additional hour."""
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE = Path('/home/cms/experiments/v6-broadcast-oldloss-openml12-round2-20260927')
SOURCE = Path('/home/cms/experiments/v6-broadcast-oldloss-openml12-20260927')
MANIFEST = '/home/cms/experiments/openml12-joint2h-squared-20260927/manifests/joint-openml12-squared.json'
BANK = '/home/cms/experiments/openml12-frozen-icl-20260927'
PARENT_SHA = '86f380c39c4da737eb92de301e1c69ac11c47304aa23b86ee034a40bfb3ad952'
PARENT = f'/home/cms/experiments/openml12-joint30m-squared-20260927/half2/run/checkpoints/{PARENT_SHA}.pt'
START_SHA = '1ee9f701aa79c8a014042dcb40b45637663a7947f22131c3056761162d4a4ea4'
START_SECONDS = 3600.178439423209
LAUNCHER = str(Path.home()/'.local/bin/wehub-python')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def execute(script, arguments, logfile):
    # W&B runs in the independent local observer; preserve the frozen runner hash.
    command = [LAUNCHER, '--profile', 'train-20260920', '-u', '-c',
               f"import sys,runpy; sys.modules['wandb']=None; runpy.run_path({str(SOURCE/script)!r},run_name='__main__')",
               *map(str, arguments)]
    with logfile.open('w') as log:
        subprocess.run(command, cwd=SOURCE/'source/src', env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    status = dict(outcome='starting', started_utc=datetime.now(timezone.utc).isoformat(),
                  start_sha256=START_SHA, start_update=9372, start_successful_seconds=START_SECONDS,
                  additional_seconds_target=3600, total_seconds_target=START_SECONDS+3600,
                  continuation='strict_resume_optimizer_rng_cursor_exposure',
                  sampling='Puma:3,kin8nm:3,other10:1 each,history618:4 per 20 updates')
    save(BASE/'status.json', status)
    resume = SOURCE/'stage1h/run/checkpoints'/f'{START_SHA}.pt'
    checksum = START_SHA
    try:
        for label, budget in [('half1', 1800), ('half2', 3600)]:
            root = BASE/label/'run'
            status.update(outcome='training', phase=label)
            save(BASE/'status.json', status)
            execute('run_broadcast_oldloss.py', [
                '--root', root, '--manifest', MANIFEST, '--parent', PARENT,
                '--parent-sha', PARENT_SHA, '--device', 'cuda:0',
                '--seconds', START_SECONDS+budget, '--save-every-seconds', 300,
                '--resume-checkpoint', resume, '--resume-sha', checksum,
            ], BASE/f'{label}-train.log')
            campaign = json.loads((root/'campaign.json').read_text())
            assert campaign['outcome'] == 'training_completed'
            assert campaign['successful_update_seconds'] >= START_SECONDS+budget
            save(BASE/'receipts'/f'{label}-train.json', campaign)
            resume, checksum = Path(campaign['checkpoint']), campaign['checkpoint_sha256']
            status.update(outcome='evaluating', checkpoint_sha256=checksum,
                          checkpoint_update=campaign['checkpoint_update'],
                          additional_successful_seconds=campaign['successful_update_seconds']-START_SECONDS)
            save(BASE/'status.json', status)
            execute('evaluate.py', [
                '--manifest', MANIFEST, '--checkpoint', resume, '--expected-sha', checksum,
                '--device', 'cuda:0', '--bank-root', BANK, '--output', BASE/'evaluations'/label,
            ], BASE/f'{label}-eval.log')
            result = json.loads((BASE/'evaluations'/label/'terminal.json').read_text())
            assert result['outcome'] == 'completed' and result['total_predictions'] == 7894
            save(BASE/'receipts'/f'{label}-eval.json', result)
            status.setdefault('evaluations', {})[label] = dict(macro=result['macro'],
                                                               tables=result['tables'])
            save(BASE/'status.json', status)
        status.update(outcome='completed', completed_utc=datetime.now(timezone.utc).isoformat())
        save(BASE/'status.json', status)
    except Exception as error:
        status.update(outcome='failed', error_type=type(error).__name__, error=str(error))
        save(BASE/'status.json', status)
        raise


if __name__ == '__main__':
    main()
