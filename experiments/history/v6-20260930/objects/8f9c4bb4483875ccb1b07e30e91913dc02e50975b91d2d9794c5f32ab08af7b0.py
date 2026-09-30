import pathlib,json,subprocess,sys
B=pathlib.Path(__file__).resolve().parent
h=sys.argv[1];home='/home/cms' if h=='dgx2' else '/Users/'+('gongqian' if h=='gongqian-mini' else h);root=home+'/experiments/'+B.name

def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',h,'python3 -'],input=code,text=True,capture_output=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
r=remote(f"import pathlib,json\nb=pathlib.Path({root!r});b.mkdir(exist_ok=False)\n(b/'queue.json').write_text(json.dumps(dict(host={h!r})))\nprint(b)\n");(B/f'prepare-{h}.txt').write_text(r)
subprocess.run(['scp','-q',*[str(B/n) for n in ['controller.py','eval_openml.py','eval_sparse.py','evaluation_support.py','frozen_evaluate.py']],h+':'+root+'/'],check=True)
code=f'''import pathlib,subprocess,json
b=pathlib.Path({root!r});syn=b.parent/'v6-sparse-relevance100-20260928'
(b/'evaluation-bank.json').write_bytes((syn/'evaluation-bank.json').read_bytes());(b/'data').symlink_to(syn/'data',target_is_directory=True)
subprocess.run(['python3','-m','py_compile',str(b/'controller.py')],check=True)
with (b/'controller.log').open('w') as f:p=subprocess.Popen(['python3',str(b/'controller.py')],cwd=b,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(b))))
'''
r=remote(code);(B/f'launch-{h}.json').write_text(r);print(h,r)
