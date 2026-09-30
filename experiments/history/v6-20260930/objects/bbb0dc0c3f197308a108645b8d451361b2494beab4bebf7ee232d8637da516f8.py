import pathlib,subprocess,json,sys
B=pathlib.Path(__file__).resolve().parent
h=sys.argv[1];home='/home/cms' if h=='dgx2' else '/Users/'+('gongqian' if h=='gongqian-mini' else h);root=home+'/experiments/'+B.name;old=home+'/experiments/v6-v55-dual718-30m-20260928';sha=json.loads((B/'parents.json').read_text())[h]
def remote(code):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -'],input=code,text=True,capture_output=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr[-3000:])
 return r.stdout
code=f'''import pathlib,json,subprocess
b=pathlib.Path({root!r});o=pathlib.Path({old!r})
assert json.loads((o/'status.json').read_text())['outcome']=='completed'
c=json.loads((o/'half2/run/campaign.json').read_text());e=json.loads((o/'evaluations/half2/terminal.json').read_text())
assert c['outcome']=='training_completed' and e['outcome']=='completed'
assert c['checkpoint_sha256']==e['checkpoint_sha256']=={sha!r}
r=subprocess.run([str(pathlib.Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','--runtime-info'],capture_output=True,text=True,check=True)
assert json.loads(r.stdout)['ok']
b.mkdir(exist_ok=False)
(b/'runtime.json').write_text(r.stdout)
subprocess.run(['cp','-R',str(o/'source'),str(b/'source')],check=True)
for n in ['manifest.json','fixed-fit-bank.json']:(b/n).write_bytes((o/n).read_bytes())
(b/'before-eval.json').write_text(json.dumps(e,indent=2))
(b/'original-before-eval.json').write_bytes((o/'before-eval.json').read_bytes())
(b/'initial.json').write_text(json.dumps(dict(parent=c['checkpoint'],parent_sha256=c['checkpoint_sha256'])))
(b/'queue.json').write_text(json.dumps(dict(host={h!r},device={'cuda:0' if h=='dgx2' else 'mps'!r},parent_root=str(o),expected_parent_sha={sha!r})))
print(json.dumps(dict(prepared=True,runtime=json.loads(r.stdout),load=subprocess.getoutput('uptime'),disk=subprocess.getoutput('df -h .'))))
'''
(B/f'prepare-{h}.json').write_text(remote(code))
files=['dual.py','train.py','training_support.py','controller.py','evaluate.py']
subprocess.run(['scp','-q',*[str(B/n) for n in files],h+':'+root+'/'],check=True)
code=f'''import pathlib,subprocess,json
b=pathlib.Path({root!r})
subprocess.run(['python3','-m','py_compile',*[str(b/n) for n in ['train.py','controller.py','dual.py','evaluate.py']]],check=True)
assert not (b/'status.json').exists()
with (b/'controller.log').open('w') as f:p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(b))))
'''
out=remote(code);(B/f'launch-{h}.json').write_text(out)
with (B/f'observer-{h}.log').open('w') as f:ob=subprocess.Popen(['/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python',str(B/'observe.py'),'--host',h,'--remote-root',root],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(B/f'observer-launch-{h}.json').write_text(json.dumps(dict(pid=ob.pid)))
print(h,out)
