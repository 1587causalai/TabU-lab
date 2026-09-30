"""Local receipt mirror/W&B observer. Never launches or restarts training."""
import argparse,json,os,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def main(a):
    local=B/'monitor'/a.host;local.mkdir(parents=True,exist_ok=True)
    state=dict(outcome='monitoring',offsets={},nodes=[],logged_updates=0);run=None
    try:
        import wandb
        run=wandb.init(entity='zj3712',project='restoration-v6-target-broadcast',id='puma-lrval-'+a.host+'-20260929',
            name='Puma-lrval-'+a.host,resume='allow',
            settings=wandb.Settings(disable_git=True,disable_code=True,console='off',silent=True,x_disable_stats=True,x_disable_meta=True,init_timeout=30),
            config=dict(table='pumadyn32nh',branch=a.branch,learning_rates=[1e-4,3e-5,1e-5],seconds_per_trial=900,monitor_excluded_from_updates=True))
        state['wandb_url']=run.url
    except Exception as e:state['wandb_error']=repr(e)
    while True:
        try:
            code="""import json,pathlib
b=pathlib.Path(ROOT); offsets=OFFSETS; out={'files':{},'rows':{},'offsets':{}}
rels=['status.json','terminal.json','selection.json','setup-receipt.json','preflight/campaign.json','run/campaign.json']
rels += [str(p.relative_to(b)) for p in (b/'run').glob('*/monitor/*/terminal.json')]
rels += [str(p.relative_to(b)) for p in (b/'run').glob('*/selected-eval/*/terminal.json')]
for rel in rels:
 p=b/rel
 if p.exists():out['files'][rel]=json.loads(p.read_text())
for tag in ['lr1e-4','lr3e-5','lr1e-5']:
 p=b/'run'/tag/'updates.jsonl';out['rows'][tag]=[];out['offsets'][tag]=offsets.get(tag,0)
 if p.exists():
  with p.open('rb') as f:
   f.seek(offsets.get(tag,0))
   while True:
    line=f.readline()
    if not line or not line.endswith(b'\\n'):break
    out['rows'][tag].append(json.loads(line));out['offsets'][tag]=f.tell()
print(json.dumps(out))
""".replace('ROOT',repr(a.root)).replace('OFFSETS',repr(state['offsets']))
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=45)
            d=json.loads(r.stdout)
            for rel,obj in d['files'].items():save(local/rel,obj)
            for tag,rows in d['rows'].items():
                if rows:
                    p=local/'run'/tag/'updates.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('a') as f:
                        for row in rows:
                            f.write(json.dumps(row)+'\n');state['logged_updates']+=1
                            if run:run.log({tag+'/loss':row['loss'],tag+'/seconds':row['successful_update_seconds'],tag+'/lr':row['learning_rate']},step=state['logged_updates'])
            state['offsets']=d['offsets']
            for rel,obj in d['files'].items():
                if rel.endswith('terminal.json') and 'metrics' in obj and rel not in state['nodes']:
                    state['nodes'].append(rel)
                    if run:
                        for key,value in obj['metrics'].items():run.summary[rel.removesuffix('/terminal.json')+'/'+key]=value
            s=d['files'].get('status.json',{});state['controller_outcome']=s.get('outcome');state.pop('fetch_error',None)
            save(local/'observer.json',state)
            if s.get('outcome') in ['completed','failed']:
                state['outcome']=s['outcome'];save(local/'observer.json',state)
                if run:run.finish(exit_code=int(s['outcome']=='failed'))
                return
        except Exception as e:state['fetch_error']=repr(e);save(local/'observer.json',state)
        time.sleep(25)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['host','root','branch']:p.add_argument('--'+key,required=True)
    main(p.parse_args())
