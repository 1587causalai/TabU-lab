"""Mirror receipts and record existing W&B project; never starts remote work."""
import argparse,json,os,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def fetch(host,root,offset):
    code='''import pathlib,json
b=pathlib.Path(ROOT);out={'files':{},'rows':[],'offset':OFFSET}
for rel in ['status.json','run/campaign.json','preflight/campaign.json','setup-receipt.json','baselines/status.json','baselines/terminal.json']+[f'evaluations/{n}/terminal.json' for n in ['before','minute05','minute10','minute15']]:
 p=b/rel
 if p.exists():out['files'][rel]=json.loads(p.read_text())
p=b/'run/updates.jsonl'
if p.exists():
 with p.open('rb') as f:
  f.seek(OFFSET)
  while True:
   line=f.readline()
   if not line or not line.endswith(b'\\n'):break
   out['rows'].append(json.loads(line));out['offset']=f.tell()
print(json.dumps(out))
'''.replace('ROOT',repr(root)).replace('OFFSET',str(offset))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=45)
    return json.loads(r.stdout)
def main(a):
    local=B/'monitor'/a.host;local.mkdir(parents=True,exist_ok=True)
    observer={'outcome':'monitoring','offset':0,'evaluated':[]};run=None
    try:
        import wandb
        run=wandb.init(entity='zj3712',project='restoration-v6-target-broadcast',id='puma-single15m-'+a.host+'-20260929',name='Puma-single-'+a.host+'-'+a.branch,
             resume='allow',settings=wandb.Settings(disable_git=True,disable_code=True,console='off',silent=True,x_disable_stats=True,x_disable_meta=True,init_timeout=30),
             config=dict(table='pumadyn32nh',branch=a.branch,train_seconds=900,window_rows=512,replay=False))
        observer['wandb_url']=run.url
    except Exception as e:observer['wandb_error']=type(e).__name__+': '+str(e)
    while True:
        try:
            d=fetch(a.host,a.root,observer['offset'])
            for rel,obj in d['files'].items():save(local/rel,obj)
            for row in d['rows']:
                with (local/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if run:run.log({'train/loss':row['loss'],'train/successful_seconds':row['successful_update_seconds'],'train/gradient_norm':row['gradient_norm']},step=row['update'])
            observer['offset']=d['offset']
            for rel,obj in d['files'].items():
                if rel.startswith('evaluations/') and obj.get('outcome')=='completed' and rel not in observer['evaluated']:
                    observer['evaluated'].append(rel)
                    if run:
                        for group,m in obj['metrics'].items():
                            for k,v in m.items():run.summary[rel.split('/')[1]+'/'+group+'/'+k]=v
            status=d['files'].get('status.json',{});base=d['files'].get('baselines/status.json',{})
            observer['controller_outcome']=status.get('outcome');observer.pop('fetch_error',None)
            complete=status.get('outcome') in ('completed','failed') and (a.host!='dgx2' or base.get('outcome') in ('completed','failed'))
            save(local/'observer.json',observer)
            if complete:
                observer['outcome']='completed' if status['outcome']=='completed' and (a.host!='dgx2' or base['outcome']=='completed') else 'failed'
                save(local/'observer.json',observer)
                if run:run.finish(exit_code=int(observer['outcome']=='failed'))
                return
        except Exception as e:
            observer['fetch_error']=type(e).__name__+': '+str(e);save(local/'observer.json',observer)
        time.sleep(20)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['host','root','branch']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
