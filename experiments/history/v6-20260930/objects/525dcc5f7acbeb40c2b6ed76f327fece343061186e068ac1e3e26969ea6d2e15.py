import sys,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
HOSTS={
'dustinstudio':dict(home='/Users/dustinstudio',device='mps',supervision='joint_all',sha='dfaf831b193d08422cb7bd1d315d4c71988573c7695912951b93b1f2a4b95844'),
'dgx2':dict(home='/home/cms',device='cuda:0',supervision='target_only',sha='55a9883e8a85cb798ba36500cbd115d324125b732cf429d8bfac00fa615ca0fb'),
'gongqian-mini':dict(home='/Users/gongqian',device='mps',supervision='target_only',sha='38c843d9a85f8abe3dfb3c43ce03ae479694a4f51125ab245d7588a89d507ea2')}
def remote(h,code):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=90)
 return r.stdout
for h in sys.argv[1:]:
 cfg=HOSTS[h];root=cfg['home']+'/experiments/'+BASE.name
 info=remote(h,"from pathlib import Path\nimport subprocess\nr=subprocess.run([str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','--runtime-info'],capture_output=True,text=True,check=True)\nprint(r.stdout)\n")
 runtime=json.loads(info);assert runtime['ok'];(BASE/f'runtime-{h}.json').write_text(info)
 q=dict(host=h,device=cfg['device'],supervision=cfg['supervision'],parent_root=cfg['home']+'/experiments/v6-window512-replay40-extend60m-20260928',expected_parent_sha=cfg['sha'])
 remote(h,f'from pathlib import Path\nPath({root!r}).mkdir(exist_ok=False)\n')
 files=['train_jointall.py','training_support.py','evaluate.py','evaluation_support.py','frozen_evaluate.py','controller.py','source.tar','data.tar.gz','data-manifest.json','evaluation-bank.json']
 subprocess.run(['scp','-q',*[str(BASE/n) for n in files],h+':'+root+'/'],check=True)
 code=f'''from pathlib import Path
import json,subprocess
b=Path({root!r})
for n in ['source.tar','data.tar.gz']:subprocess.run(['tar','-xf',str(b/n),'-C',str(b)],check=True)
(b/'queue.json').write_text(json.dumps({q!r},indent=2)+'\\n')
with (b/'controller.log').open('w') as log:
 p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(b))))
'''
 receipt=json.loads(remote(h,code));(BASE/f'launch-{h}.json').write_text(json.dumps(receipt)+'\n')
 with (BASE/f'observer-{h}.log').open('w') as log:
  p=subprocess.Popen(['/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python',str(BASE/'observe.py'),'--host',h,'--remote-root',root],cwd=BASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (BASE/f'observer-launch-{h}.json').write_text(json.dumps(dict(pid=p.pid,host=h))+'\n')
 print(h,receipt,flush=True)
