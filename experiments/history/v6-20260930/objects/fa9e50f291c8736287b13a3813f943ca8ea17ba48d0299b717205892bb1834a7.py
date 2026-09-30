"""Local online W&B monitoring and receipt collection for both training halves."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import wandb


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def fetch(host, root, offsets):
    script = f'''import pathlib,json
root=pathlib.Path({root!r})
offsets={offsets!r}
out={{'rows':[],'offsets':offsets,'campaigns':{{}},'evals':{{}}}}
for half in ('half1','half2'):
 p=root/half/'run'
 if (p/'updates.jsonl').exists():
  with (p/'updates.jsonl').open('rb') as f:
   f.seek(offsets.get(half,0))
   while True:
    line=f.readline()
    if not line or not line.endswith(b'\\n'):break
    out['rows'].append(json.loads(line))
    offsets[half]=f.tell()
 if (p/'campaign.json').exists():out['campaigns'][half]=json.loads((p/'campaign.json').read_text())
 e=root/'evaluations'/half/'terminal.json'
 if e.exists():out['evals'][half]=json.loads(e.read_text())
out['controller']=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {{}}
print(json.dumps(out))
'''
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3','-'],
                            input=script,text=True,capture_output=True,check=True,timeout=30)
    return json.loads(result.stdout)


def main(args):
    local = Path(__file__).resolve().parent/'monitor'/args.host
    local.mkdir(parents=True, exist_ok=True)
    settings=wandb.Settings(disable_git=True,disable_code=True,console='off',silent=True,
                            x_disable_stats=True,x_disable_meta=True,init_timeout=45)
    run=wandb.init(entity='zj3712',project='restoration-v6-target-broadcast',
                   id=f'v6broadcastoldloss-round2-{args.host}-20260927',resume='allow',
                   name=f'{args.host}-broadcast-oldloss-second-round',mode='online',settings=settings,
                   config=dict(host=args.host,additional_train_seconds=3600,
                               supervision='target_only',sampling='focus2:6,other10:10,history618:4',
                               continuation=('strict_resume' if args.host=='dgx2' else 'V55_weights_only')))
    state=dict(outcome='monitoring',url=run.url,last_update=0,offsets={},evaluated=[])
    atomic(local/'status.json',state)
    try:
        while True:
            try:
                response=fetch(args.host,args.remote_root,state['offsets'])
            except (subprocess.SubprocessError,ValueError) as error:
                state['last_fetch_error']=str(error)
                atomic(local/'status.json',state)
                time.sleep(15)
                continue
            controller=response['controller']
            start=controller.get('start_successful_seconds',0)
            for row in response['rows']:
                if row['update']<=state['last_update']:
                    raise ValueError('update journal regressed')
                run.log({'train/loss':row['loss'],'train/gradient_norm':row['gradient_norm'],
                         'train/successful_seconds':row['successful_update_seconds'],
                         'train/additional_seconds':row['successful_update_seconds']-start,
                         'train/query_rows':row['query_rows'],'train/scored_cells':row['scored_cells'],
                         f'loss/{row["table"]}':row['loss']},step=row['update'])
                state['last_update']=row['update']
                state['additional_successful_seconds']=row['successful_update_seconds']-start
            state['offsets']=response['offsets']
            atomic(local/'controller.json',controller)
            for half,campaign in response['campaigns'].items():
                atomic(local/f'{half}-train.json',campaign)
            for half,evaluation in response['evals'].items():
                atomic(local/f'{half}-eval.json',evaluation)
                if half not in state['evaluated'] and evaluation.get('outcome')=='completed':
                    for metric,value in evaluation['macro'].items():
                        run.summary[f'{half}/macro/{metric}']=value
                    for table,metrics in evaluation['tables'].items():
                        for metric in ('r2','slog'):
                            run.summary[f'{half}/{table}/{metric}']=metrics[metric]
                    state['evaluated'].append(half)
            state['controller_outcome']=controller.get('outcome')
            atomic(local/'status.json',state)
            if controller.get('outcome') in ('completed','failed'):
                state['outcome']=controller['outcome']
                atomic(local/'status.json',state)
                run.finish(exit_code=0 if state['outcome']=='completed' else 1)
                return
            time.sleep(10)
    except Exception as error:
        state.update(outcome='observer_failed',error=str(error))
        atomic(local/'status.json',state)
        run.finish(exit_code=1)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',required=True)
    parser.add_argument('--remote-root',required=True)
    main(parser.parse_args())
