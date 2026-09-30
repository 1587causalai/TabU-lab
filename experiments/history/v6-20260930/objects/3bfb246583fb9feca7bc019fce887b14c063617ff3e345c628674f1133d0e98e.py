import pathlib,subprocess,json,concurrent.futures
B=pathlib.Path(__file__).resolve().parent
code="""import pathlib,json
p=pathlib.Path.home()/'experiments/v6-dual718-checkpoint-test-20260928'
files=['status.json','source-checks.json','checkpoints.json']
files += [str(x.relative_to(p)) for x in (p/'evaluations').glob('*/*/terminal.json')]
print(json.dumps({n:json.loads((p/n).read_text()) for n in files if (p/n).exists()}))
"""
def get(h):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',h,'python3 -'],input=code,text=True,capture_output=True,timeout=30)
 if r.returncode:return h,{'error':r.stderr}
 d=json.loads(r.stdout);local=B/'monitor'/h
 for n,v in d.items():
  p=local/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
 s=d['status.json'];return h,{k:s.get(k) for k in ['outcome','node','dataset','completed','error']}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for h,s in ex.map(get,['dustinstudio','dgx2','gongqian-mini']):print(h,json.dumps(s))
