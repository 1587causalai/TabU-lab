"""One launch, two bounded 30-minute segments; evaluation is outside train budget."""
import pathlib,json,os,subprocess
from datetime import datetime,timezone
B=pathlib.Path(__file__).resolve().parent
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def execute(script,args,log):
 env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
 cmd=[str(pathlib.Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',f"import runpy;runpy.run_path({str(B/script)!r},run_name='__main__')",*map(str,args)]
 with (B/log).open('w') as f:subprocess.run(cmd,cwd=B/'source/src',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=21600)
def evaluate(q,donor,phase):
 result={}
 for branch in ['v6','v55']:
  execute('evaluate.py',['--manifest',donor.get('balanced_manifest',donor.get('manifest')),'--checkpoint',donor['checkpoint'],'--expected-sha',donor['checkpoint_sha256'],'--device',q['device'],'--branch',branch,'--bank-root',B.parent/'openml12-frozen-icl-20260927','--output',B/'evaluations'/phase/('openml12-'+branch),'--memory-fraction',.75 if q['host']=='gongqian-mini' else .5],phase+'-openml12-'+branch+'.log')
  e=json.loads((B/'evaluations'/phase/('openml12-'+branch)/'terminal.json').read_text())
  assert e['outcome']=='completed' and e['total_predictions']==7894 and e['optimizer_updates']==0 and e['checkpoint_sha256']==donor['checkpoint_sha256']
  result['openml12-'+branch]=e
 execute('eval_fit.py',['--host',q['host'],'--device',q['device'],'--checkpoint',donor['checkpoint'],'--expected-sha',donor['checkpoint_sha256'],'--output',B/'evaluations'/phase/'fit718'],phase+'-fit718.log')
 e=json.loads((B/'evaluations'/phase/'fit718/terminal.json').read_text());assert e['outcome']=='completed' and e['optimizer_updates']==0 and e['total_masks']==1436 and e['total_predictions_per_branch']==97648 and e['checkpoint_sha256']==donor['checkpoint_sha256'];result['fit718']=e
 return result
def compact(e):return {k:(v['by_family'] if k=='fit718' else v['macro']) for k,v in e.items()}
def main():
 q=json.loads((B/'queue.json').read_text());donor=q['parent'];s=dict(outcome='verifying_reused_before',host=q['host'],objective='v6_v55_bernoulli50_sparse30',start_sha256=donor['checkpoint_sha256'],completed_training_seconds=0,additional_seconds_target=3600,started_utc=datetime.now(timezone.utc).isoformat());save(B/'status.json',s)
 try:
  baseline={k:json.loads((B/'evaluations/before'/k/'terminal.json').read_text()) for k in ['openml12-v6','openml12-v55','fit718']};
  for k,e in baseline.items():
   assert e['outcome']=='completed' and e['optimizer_updates']==0 and e['checkpoint_sha256']==donor['checkpoint_sha256']
   if k=='fit718':assert e['total_masks']==1436 and e['total_predictions_per_branch']==97648 and e['bank_sha256']==json.loads((B/'fixed-fit-bank.json').read_text())['sha256']
   else:assert e['total_predictions']==7894 and e['inference_branch']==k.split('-')[1]
  s['evaluations']={'before':compact(baseline)};save(B/'status.json',s)
  for i in range(1,3):
   phase='stage'+str(i);budget=i*1800-s['completed_training_seconds'];assert 0<budget<=1800
   s.update(outcome='training',phase=phase,phase_budget=budget);save(B/'status.json',s)
   execute('train.py',['--host',q['host'],'--device',q['device'],'--parent',donor['checkpoint'],'--parent-sha',donor['checkpoint_sha256'],'--manifest',donor.get('balanced_manifest',donor.get('manifest')),'--supervision','target_only','--segment',phase,'--seconds',budget]+(['--reset-sampling-cursor'] if i==1 else []),phase+'-train.log')
   c=json.loads((B/phase/'run/campaign.json').read_text());assert c['outcome']=='training_completed' and c['parent_sha256']==donor['checkpoint_sha256']
   assert c['schedule_cursor_reset'] == (i==1) and c['branch_selector_restored'] is True and c['forward_passes_per_update']==c['optimizer_steps_per_update']==1
   assert sum(c['branch_counts'].values())==sum(c['table_updates'].values())
   s['completed_training_seconds']+=c['successful_update_seconds'];donor=c;s.update(outcome='evaluating',checkpoint_sha256=c['checkpoint_sha256']);save(B/'status.json',s)
   ev=evaluate(q,c,phase)
   for kind,e in ev.items():assert e['bank_sha256']==baseline[kind]['bank_sha256']
   s['evaluations'][phase]=compact(ev);save(B/'status.json',s)
  assert 3600<=s['completed_training_seconds']<3660
  s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(B/'status.json',s)
 except Exception as ex:
  s.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save(B/'status.json',s);raise
if __name__=='__main__':main()
