import pathlib,json,subprocess,sys
B=pathlib.Path(__file__).resolve().parent;h=sys.argv[1];home='/home/cms' if h=='dgx2' else '/Users/'+('gongqian' if h=='gongqian-mini' else h);root=home+'/experiments/'+B.name
local=json.loads((B.parent/'v6-v55-dual718-continue60m-20260928/monitor'/h/'half2-train.json').read_text());sha=local['checkpoint_sha256']
q=dict(host=h,device='cuda:0' if h=='dgx2' else 'mps',supervision='joint_all' if h=='dustinstudio' else 'target_only',expected_parent_sha=sha)
def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',h,'python3 -'],input=code,text=True,capture_output=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
code=f'''import pathlib,json,subprocess
b=pathlib.Path({root!r});p=b.parent/'v6-v55-dual718-continue60m-20260928'
assert json.loads((p/'status.json').read_text())['outcome']=='completed'
c=json.loads((p/'half2/run/campaign.json').read_text());assert c['checkpoint_sha256']=={sha!r}
b.mkdir(exist_ok=False);subprocess.run(['cp','-R',str(p/'source'),str(b/'source')],check=True)
syn=b.parent/'v6-sparse-relevance100-20260928';(b/'evaluation-bank.json').write_bytes((syn/'evaluation-bank.json').read_bytes());(b/'data').symlink_to(syn/'data',target_is_directory=True)
(b/'queue.json').write_text(json.dumps({q!r},indent=2)+'\\n')
print(json.dumps(dict(prepared=True,root=str(b),parent_sha256=c['checkpoint_sha256'])))
'''
(B/f'prepare-{h}.json').write_text(remote(code))
files=['controller.py','train.py','training_support.py','evaluate.py','eval_sparse.py','evaluation_support.py','frozen_evaluate.py']
subprocess.run(['scp','-q',*[str(B/n) for n in files],h+':'+root+'/'],check=True)
code=f'''import pathlib,json,subprocess
b=pathlib.Path({root!r});subprocess.run(['python3','-m','py_compile',*[str(b/n) for n in ['controller.py','train.py']]],check=True)
with (b/'controller.log').open('w') as f:p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(b))))
'''
r=remote(code);(B/f'launch-{h}.json').write_text(r)
with (B/f'observer-{h}.log').open('w') as f:p=subprocess.Popen(['/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python',str(B/'observe.py'),'--host',h,'--remote-root',root],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(B/f'observer-launch-{h}.json').write_text(json.dumps(dict(pid=p.pid)))
print(h,r)
