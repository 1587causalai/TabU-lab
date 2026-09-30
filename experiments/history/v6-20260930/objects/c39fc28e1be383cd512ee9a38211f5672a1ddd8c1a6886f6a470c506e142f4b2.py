"""Audit mirrored receipts; no remote mutations or training actions."""
import json,math
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
cfg=json.loads((B/'experiment.json').read_text());parents=json.loads((B/'parents.json').read_text())
def read(p):return json.loads(p.read_text())
report={'checked_utc':datetime.now(timezone.utc).isoformat(),'hosts':{},'paired_complete':[]}
banks=[]
for host in cfg['hosts']:
 p=B/'monitor'/host;state=read(p/'status.json');setup=read(p/'setup-receipt.json');banks.append(setup['tables'])
 assert state['parent_sha256']==parents[host]['checkpoint_sha256']
 initial=set();completed=[];formal=[];preflight=[]
 for name in cfg['tables']:
  root=p/'tables'/name;dims=setup['tables'][name];f=root/'preflight/campaign.json'
  if f.exists():
   s=read(f)
   if s['outcome']=='passed':
    assert s['formal_updates']==0 and s['parent_unchanged'];preflight.append(name)
  f=root/'status.json'
  if not f.exists():continue
  s=read(f);assert s['parent_sha256']==parents[host]['checkpoint_sha256']
  assert s['bank_sha256']==dims['bank_sha256'] and s['split_sha256']==dims['split_sha256']
  assert list(s['trials'])==['lr1e-4']
  t=s['trials']['lr1e-4'];initial.add(t['initial_state_hash']);assert t['initial_rng_restored']
  log=root/'run/lr1e-4/updates.jsonl'
  if log.exists():
   rows=[json.loads(r) for r in log.read_text().splitlines() if r.strip()]
   assert all(r['learning_rate']==1e-4 and r['forward_passes']==r['optimizer_steps']==1 and r['train_rows']==dims['window_rows'] and r['query_rows']==r['scored_cells']==round(dims['window_rows']/3) for r in rows)
   if rows:formal.append(name)
  if s.get('test_started'):assert s['selection_completed_utc']<s['test_started_utc']
  if s['outcome']=='completed':
   assert 900<=t['successful_update_seconds']<930 and len(rows)==t['updates']
   assert math.isclose(sum(r['seconds'] for r in rows)+t.get('prior_attempt_successful_seconds',0),t['successful_update_seconds'],abs_tol=1e-6)
   assert t['selected_node']==min(t['nodes'],key=lambda k:(t['nodes'][k]['metrics']['mse'],int(k)))
   for group,n in [('train',dims['fit_rows']),('test',dims['test_rows'])]:
    ev=t['final_evaluations'][group];assert ev['outcome']=='completed' and ev['predictions']==n and ev['optimizer_updates']==0 and ev['model_state_unchanged']
    assert ev['checkpoint_sha256']==t['selected']['checkpoint_sha256']
   completed.append(name)
 assert len(initial)<=1
 report['hosts'][host]={'outcome':state['outcome'],'phase':state['phase'],'active_table':state.get('active_table'),'preflight_verified':preflight,'formal_update_tables':formal,'completed':completed}
assert banks[0]==banks[1]
for name in cfg['tables']:
 f=B/'monitor/dgx2/tables'/name/'baselines/terminal.json'
 if f.exists():
  s=read(f);assert s['outcome']=='completed'
  assert s['selection_completed_utc']<s['test_started_utc'] and s['split_sha256']==banks[0][name]['split_sha256']
  assert set(s['models'])=={'mlp-256x3','xgboost-depth6'}
  assert all(v['metrics']['test']['n']==banks[0][name]['test_rows'] and not v['refit'] for v in s['models'].values())
  if all(name in h['completed'] for h in report['hosts'].values()):report['paired_complete'].append(name)
(B/'audit-progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
