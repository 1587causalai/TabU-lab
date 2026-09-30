"""Mirror only this experiment's small receipts; never restart training."""
import json,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent
for _ in range(150):
    p=subprocess.run(['rsync','-a','--exclude=source/','--exclude=checkpoints/','--exclude=*.pt','--exclude=sync_results.py','--exclude=observer*',
        'gongqian-mini:experiments/'+B.name+'/',str(B)+'/'],capture_output=True,text=True,timeout=90)
    if p.returncode:
        (B/'observer-error.txt').write_text(p.stderr)
    else:
        path=B/'status.json'
        if path.exists() and json.loads(path.read_text()).get('outcome') in ('completed','failed'):
            break
    time.sleep(60)
