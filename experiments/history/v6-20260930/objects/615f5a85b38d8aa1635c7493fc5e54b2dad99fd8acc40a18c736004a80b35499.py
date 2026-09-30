"""One launch, wait for prior experiment, preflight then bounded sequential jobs."""
import json,os,subprocess,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
q=json.loads((B/'queue.json').read_text());cfg=json.loads((B/'experiment.json').read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def save():
 p=B/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');os.replace(tmp,p)
s=dict(outcome='waiting',phase='wait_prior',host=q['host'],branch=q['branch'],parent_sha256=q['parent']['checkpoint_sha256'],pid=os.getpid(),started_utc=utc(),tables={},baselines={},successful_seconds_budget=10800)
def run(script,root=None,preflight=False):
 args=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u',str(B/script)]
 if root:args+=['--root',str(root)]
 if preflight:args+=['--preflight']
 logfile=B/('setup.log' if not root else f'{root.name}-{script}-{preflight}.log')
 with logfile.open('a') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)
try:
 save()
 while True:
  p=B.parent/cfg['wait_for']/'terminal.json'
  if p.exists():
   prior=json.loads(p.read_text())
   if prior['outcome']!='completed':raise RuntimeError('Prior Puma experiment failed; queued experiment will not start')
   break
  time.sleep(30)
 s.update(outcome='running',phase='setup',prior_completed_utc=prior.get('completed_utc'));save()
 assert hashlib.sha256(Path(q['parent']['checkpoint']).read_bytes()).hexdigest()==q['parent']['checkpoint_sha256']
 if not (B/'setup-receipt.json').exists():run('setup.py')
 receipt=json.loads((B/'setup-receipt.json').read_text());assert receipt['parent_sha256']==q['parent']['checkpoint_sha256']
 # Complete all table smoke checks before using formal training budget.
 for name in cfg['tables']:
  s.update(phase='preflight',active_table=name);save();root=B/'tables'/name
  run('run_table.py',root,True)
  receipt=json.loads((root/'preflight/campaign.json').read_text());assert receipt['outcome']=='passed' and receipt['formal_updates']==0
  s['tables'][name]={'preflight':'passed'};save()
 for name in cfg['tables']:
  root=B/'tables'/name
  if q['host']=='dgx2':
   s.update(phase='baselines',active_table=name);save();run('baselines.py',root)
   r=json.loads((root/'baselines/terminal.json').read_text());assert r['outcome']=='completed';s['baselines'][name]={'outcome':'completed'};save()
  s.update(phase='tabu_training',active_table=name);save();run('run_table.py',root)
  r=json.loads((root/'terminal.json').read_text());assert r['outcome']=='completed'
  s['tables'][name].update(outcome='completed',successful_seconds=r['total_successful_seconds']);save()
 s.update(outcome='completed',phase='completed',completed_utc=utc(),total_successful_seconds=sum(t['successful_seconds'] for t in s['tables'].values()))
 assert 10800<=s['total_successful_seconds']<11160
 save();(B/'terminal.json').write_text(json.dumps(s,indent=2)+'\n')
except Exception as ex:
 s.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save();(B/'terminal.json').write_text(json.dumps(s,indent=2)+'\n');raise
