"""Read-only mirror and W&B observer of queued 12-table exploration."""
import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
NAMES=json.loads((B/'experiment.json').read_text())['tables']
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def main(a):
 local=B/'monitor'/a.host;local.mkdir(parents=True,exist_ok=True)
 state=json.loads((local/'observer.json').read_text()) if (local/'observer.json').exists() else dict(offsets={},nodes=[],logged_updates=0)
 state['outcome']='monitoring';run=None
 try:
  import wandb
  run=wandb.init(entity='zj3712',project='restoration-v6-target-broadcast',id='openml12-single-earlystop-'+a.host+'-20260929',name='OpenML12-single-earlystop-'+a.host,resume='allow',settings=wandb.Settings(disable_git=True,disable_code=True,console='off',silent=True,x_disable_stats=True,x_disable_meta=True,init_timeout=30),config=dict(branch=a.branch,seconds_per_table=900,learning_rate=1e-4,refit=False,validation_only_selection=True))
  state['wandb_url']=run.url
 except Exception as e:state['wandb_error']=repr(e)
 while True:
  try:
   code="""import json,pathlib
b=pathlib.Path(ROOT);offsets=OFFSETS;out={'files':{},'rows':{},'offsets':{}}
rels=['status.json','terminal.json','setup-receipt.json','launch.json']
for name in NAMES:
 prefix='tables/'+name+'/'
 rels += [prefix+r for r in ['status.json','terminal.json','selection.json','recovery-budget.json','preflight/campaign.json','baselines/status.json','baselines/terminal.json','baselines/selection.json']]
 rels += [str(p.relative_to(b)) for p in (b/prefix/'run').glob('*/monitor/*/terminal.json')]
 rels += [str(p.relative_to(b)) for p in (b/prefix/'run').glob('*/selected-eval/*/terminal.json')]
 p=b/prefix/'run/lr1e-4/updates.jsonl';out['rows'][name]=[];out['offsets'][name]=offsets.get(name,0)
 if p.exists():
  with p.open('rb') as f:
   f.seek(offsets.get(name,0))
   while True:
    line=f.readline()
    if not line or not line.endswith(b'\\n'):break
    out['rows'][name].append(json.loads(line));out['offsets'][name]=f.tell()
for rel in rels:
 p=b/rel
 if p.exists():out['files'][rel]=json.loads(p.read_text())
print(json.dumps(out))
""".replace('ROOT',repr(a.root)).replace('OFFSETS',repr(state['offsets'])).replace('NAMES',repr(NAMES))
   r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=45);d=json.loads(r.stdout)
   for rel,obj in d['files'].items():save(local/rel,obj)
   for name,rows in d['rows'].items():
    if rows:
     p=local/'tables'/name/'run/lr1e-4/updates.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
     with p.open('a') as f:
      for row in rows:
       f.write(json.dumps(row)+'\n');state['logged_updates']+=1
       if run:run.log({name+'/loss':row['loss'],name+'/seconds':row['successful_update_seconds'],name+'/lr':row['learning_rate']},step=state['logged_updates'])
   state['offsets']=d['offsets']
   for rel,obj in d['files'].items():
    if rel.endswith('terminal.json') and 'metrics' in obj and rel not in state['nodes']:
     state['nodes'].append(rel)
     if run:
      for key,value in obj['metrics'].items():run.summary[rel.removesuffix('/terminal.json')+'/'+key]=value
   status=d['files'].get('status.json',{});state['controller_outcome']=status.get('outcome');state['last_sync_utc']=datetime.now(timezone.utc).isoformat();state.pop('fetch_error',None);save(local/'observer.json',state)
   if status.get('outcome') in ['completed','failed']:
    state['outcome']=status['outcome'];save(local/'observer.json',state)
    if run:run.finish(exit_code=int(status['outcome']=='failed'))
    return
  except Exception as e:state['fetch_error']=repr(e);save(local/'observer.json',state)
  time.sleep(30)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ['host','root','branch']:p.add_argument('--'+key,required=True)
 main(p.parse_args())
