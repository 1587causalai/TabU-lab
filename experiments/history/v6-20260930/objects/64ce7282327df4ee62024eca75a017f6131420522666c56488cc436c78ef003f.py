import sys,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent
HOSTS={'dustinstudio':('/Users/dustinstudio','mps','ced305ae38d9ac5e648f4ae0cd517253f88a2849bb2232139c2d33d47c2b6231'),'dgx2':('/home/cms','cuda:0','b210401a5a31e81386ad907f501dba54c5d58daf77693a980e8263420f2b38f7'),'gongqian-mini':('/Users/gongqian','mps','e249ad031408d2171c803697baca35583664d1aaddfe6e59af10fd186cb85514')}
def remote(h,code,timeout=120):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(h+': '+r.stderr[-3000:]+r.stdout[-3000:])
 return r.stdout
h=sys.argv[1];action=sys.argv[2];home,device,sha=HOSTS[h];root=home+'/experiments/'+B.name
if action=='prepare':
 remote(h,f'from pathlib import Path\nPath({root!r}).mkdir(exist_ok=False)\n')
 files=['dual.py','train.py','prepare_run.py','smoke.py','evaluate.py','controller.py','training_support.py']
 subprocess.run(['scp','-q',*[str(B/n) for n in files],h+':'+root+'/'],check=True)
 q=dict(host=h,device=device,parent_root=home+'/experiments/v6-openml12-recover30m-20260928',expected_parent_sha=sha)
 code=f'''import pathlib,json,subprocess,os
b=pathlib.Path({root!r});old=pathlib.Path({q['parent_root']!r})
assert json.loads((old/'status.json').read_text())['outcome']=='completed'
subprocess.run(['cp','-R',str(old/'source'),str(b/'source')],check=True)
(b/'queue.json').write_text(json.dumps({q!r},indent=2))
env={{**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}}
for script in ['prepare_run.py','smoke.py']:
 cmd=[str(pathlib.Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',"import runpy;runpy.run_path("+repr(str(b/script))+",run_name='__main__')"]
 with (b/(script+'.log')).open('w') as f:
  r=subprocess.run(cmd,cwd=b/'source/src',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 if r.returncode:raise RuntimeError((b/(script+'.log')).read_text()[-4000:])
print((b/'smoke.json').read_text())
'''
 out=remote(h,code,420);(B/f'smoke-{h}.json').write_text(out);print(h,out,flush=True)
elif action=='launch':
 code=f'''import pathlib,json,subprocess
b=pathlib.Path({root!r})
assert json.loads((b/'smoke.json').read_text())['outcome']=='passed'
assert not (b/'status.json').exists()
with (b/'controller.log').open('w') as f:
 p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(b))))
'''
 out=remote(h,code);(B/f'launch-{h}.json').write_text(out)
 with (B/f'observer-{h}.log').open('w') as f:
  ob=subprocess.Popen(['/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python',str(B/'observe.py'),'--host',h,'--remote-root',root],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (B/f'observer-launch-{h}.json').write_text(json.dumps(dict(pid=ob.pid)))
 print(h,out,flush=True)
