"""One authorized Dustin recovery; completed tables are never rerun."""
import json,os,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
q=json.loads((B/'queue.json').read_text());cfg=json.loads((B/'experiment.json').read_text())
assert q['host']=='dustinstudio' and q['branch']=='v6'
os.close(os.open(B/'recovery-controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
def utc():return datetime.now(timezone.utc).isoformat()
def save():
 p=B/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');os.replace(tmp,p)
s=json.loads((B/'attempt1-gradient-failure/status.json').read_text())
s.pop('error',None);s.pop('error_type',None)
s.update(outcome='running',phase='recovery',pid=os.getpid(),recovery_started_utc=utc(),
         recovery='bounded-identical-backward-replay-v1',successful_seconds_budget=10800)
try:
 assert hashlib.sha256(Path(q['parent']['checkpoint']).read_bytes()).hexdigest()==q['parent']['checkpoint_sha256']
 assert json.loads((B/'finite-backward-test.json').read_text())['outcome']=='passed'
 # User requested immediate restart; extra repair preflight is not required.
 save()
 for name in cfg['tables']:
  root=B/'tables'/name
  if s['tables'][name].get('outcome')=='completed':
   r=json.loads((root/'terminal.json').read_text());assert r['outcome']=='completed'
   continue
  assert not (root/'run').exists(), 'Existing run forbids a duplicate launch'
  assert json.loads((root/'preflight/campaign.json').read_text())['outcome']=='passed'
  s.update(phase='tabu_training',active_table=name);save()
  args=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u',str(B/'run_table_repaired.py'),'--root',str(root)]
  with (B/(name+'-repaired.log')).open('a') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
  r=json.loads((root/'terminal.json').read_text());assert r['outcome']=='completed'
  s['tables'][name].update(outcome='completed',successful_seconds=r['total_successful_seconds'],
       prior_attempt_successful_seconds=r.get('prior_attempt_successful_seconds',0));save()
 s.update(outcome='completed',phase='completed',completed_utc=utc(),
    total_successful_seconds=sum(t['successful_seconds'] for t in s['tables'].values()))
 assert 10800<=s['total_successful_seconds']<11160
 save();(B/'terminal.json').write_text(json.dumps(s,indent=2)+'\n')
except Exception as ex:
 s.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save()
 (B/'terminal.json').write_text(json.dumps(s,indent=2)+'\n');raise
