import sys,json,subprocess,shlex
from pathlib import Path
BASE=Path(__file__).resolve().parent
HOSTS={
 'dgx2':dict(home='/home/cms',device='cuda:0',supervision='target_only',parent='v6-replay40-5h-20260928',phase='hour5',sha='34b9ab75ed580a0cd3fccfa23137913594aa985b31da9fd72236adbfc0f2f2b0'),
 'dustinstudio':dict(home='/Users/dustinstudio',device='mps',supervision='joint_all',parent='v6-window1024-continue60m-20260928',phase='half2',sha='77a0526f1cdfa81beadadcbe6e42b38b1d2a7c92d94763836809291650d75b6c'),
 'gongqian-mini':dict(home='/Users/gongqian',device='mps',supervision='target_only',parent='v6-window1024-continue60m-20260928',phase='half2',sha='4ae9ba2de41188f6b3712569534057df5bf0463f776a94236995533eac7561dd')}
def remote(h,code):
 r=subprocess.run(['ssh','-o','ConnectTimeout=8',h,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40)
 return r.stdout
for h in sys.argv[1:]:
 cfg=HOSTS[h];root=cfg['home']+'/experiments/'+BASE.name
 q=dict(host=h,device=cfg['device'],supervision=cfg['supervision'],parent_root=cfg['home']+'/experiments/'+cfg['parent'],parent_phase=cfg['phase'],expected_parent_sha=cfg['sha'])
 remote(h,f'from pathlib import Path\nPath({root!r}).mkdir(exist_ok=False)\n')
 for n in ['train_jointall.py','training_support.py','evaluate.py','frozen_evaluate.py','controller.py','source.tar']:
  subprocess.run(['scp','-q',str(BASE/({'train_jointall.py':'train_mini1024.py','controller.py':'controller_mini1024.py','evaluate.py':'evaluate_mini.py'}.get(n,n) if h=='gongqian-mini' else n)),h+':'+root+'/'+n],check=True)
 code=f'''from pathlib import Path
import json,subprocess,tarfile,os
b=Path({root!r})
subprocess.run(['tar','-xf',str(b/'source.tar'),'-C',str(b)],check=True)
(b/'queue.json').write_text(json.dumps({q!r},indent=2)+'\\n')
subprocess.run(['python3','-m','py_compile',str(b/'train_jointall.py'),str(b/'controller.py')],check=True)
with (b/'controller.log').open('w') as log:
 p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(b))))
'''
 receipt=json.loads(remote(h,code));(BASE/f'launch-{h}.json').write_text(json.dumps(receipt)+'\n')
 with (BASE/f'observer-{h}.log').open('w') as log:
  p=subprocess.Popen(['/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python',str(BASE/'observe.py'),'--host',h,'--remote-root',root],cwd=BASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (BASE/f'observer-launch-{h}.json').write_text(json.dumps(dict(pid=p.pid,host=h))+'\n')
 print(h,receipt,flush=True)
