"""Prepare new directory, validate, then launch only on explicit 'launch'."""
import pathlib,json,subprocess,sys
B=pathlib.Path(__file__).resolve().parent
h,action=sys.argv[1:]
home='/home/cms' if h=='dgx2' else '/Users/'+('gongqian' if h=='gongqian-mini' else h)
root=home+'/experiments/'+B.name
parents=json.loads((B/'parents.json').read_text());parent=parents[h]
q=dict(host=h,device='cuda:0' if h=='dgx2' else 'mps',parent=parent)
def remote(code,timeout=90):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',h,'python3 -'],
                     input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout
if action=='prepare':
    code=f'''import pathlib,json,subprocess
b=pathlib.Path({root!r});p=b.parent/'v6-dual718-openml12-adapt60m-20260928'
assert pathlib.Path({parent['checkpoint']!r}).is_file()
assert json.loads((p/'status.json').read_text())['outcome']=='completed'
b.mkdir(exist_ok=False)
subprocess.run(['cp','-R',str(p/'source'),str(b/'source')],check=True)
(b/'queue.json').write_text(json.dumps({q!r},indent=2)+'\\n')
print(json.dumps(dict(prepared=True,root=str(b))))
'''
    (B/f'prepare-{h}.json').write_text(remote(code))
    old=B.parent/'v6-dual718-openml12-adapt60m-20260928'
    baseline={'dustinstudio':'original-parent-eval','dgx2':'half1-eval','gongqian-mini':'half2-eval'}[h]
    ev=json.loads((old/'monitor'/h/(baseline+'.json')).read_text())
    assert ev['checkpoint_sha256']==parent['checkpoint_sha256']
    local=B/'baselines'/h;local.mkdir(parents=True)
    (local/'before-eval.json').write_text(json.dumps(ev,indent=2)+'\n')
    subprocess.run(['scp','-q',*[str(p) for p in B.glob('*.py')],str(B/'experiment.json'),
                    str(local/'before-eval.json'),h+':'+root+'/'],check=True)
    print(h,'prepared')
elif action=='smoke':
    code=f'''import pathlib,sys
b=pathlib.Path({root!r});sys.path.insert(0,str(b));import controller
controller.execute('smoke.py',[],'smoke.log')
print((b/'smoke.json').read_text())
'''
    result=remote(code,600);(B/f'smoke-{h}.json').write_text(result);print(h,result)
elif action=='launch':
    code=f'''import pathlib,json,subprocess
b=pathlib.Path({root!r})
assert json.loads((b/'smoke.json').read_text())['outcome']=='passed'
assert not (b/'controller-launch.json').exists() and not (b/'status.json').exists()
with (b/'controller.log').open('w') as f:
 p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,root=str(b));(b/'controller-launch.json').write_text(json.dumps(r))
print(json.dumps(r))
'''
    result=remote(code);(B/f'launch-{h}.json').write_text(result)
    with (B/f'observer-{h}.log').open('w') as f:
        p=subprocess.Popen(['/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python',str(B/'observe.py'),
             '--host',h,'--remote-root',root],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (B/f'observer-launch-{h}.json').write_text(json.dumps(dict(pid=p.pid)))
    print(h,result)
else:raise ValueError(action)
