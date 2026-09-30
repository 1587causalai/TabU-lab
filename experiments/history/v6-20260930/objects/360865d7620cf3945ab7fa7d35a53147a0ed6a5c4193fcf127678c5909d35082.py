import concurrent.futures,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
CODE="""from pathlib import Path
import json
b=Path.home()/'experiments/v6-sparse-relevance100-20260928'
a=Path.home()/'experiments/v6-window512-replay40-extend60m-20260928'
r={}
for k,p in [('transfer',b/'evaluations/openml12-transfer/terminal.json'),('progress',b/'evaluations/openml12-transfer/progress.json'),('baseline',a/'evaluations/half2/terminal.json'),('synthetic_final',b/'evaluations/half2/terminal.json')]:
 if p.exists():r[k]=json.loads(p.read_text())
r['log_tail']=(b/'openml12-transfer.log').read_text()[-1500:]
print(json.dumps(r))
"""
def fetch(h):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h,'python3 -'],input=CODE,text=True,capture_output=True,check=True,timeout=25);x=json.loads(r.stdout)
 d=BASE/'monitor'/h
 for k in ['transfer','baseline','synthetic_final']:
  if k in x:(d/('openml12-'+k+'.json')).write_text(json.dumps(x[k],indent=2)+'\n')
 e=x.get('transfer',x.get('progress',{}));return h,dict(outcome=e.get('outcome'),tables=len(e.get('tables',{})),macro=e.get('macro'),error=e.get('error'),log_tail=x['log_tail'] if e.get('outcome')=='failed' or not e else '')
with concurrent.futures.ThreadPoolExecutor() as ex:
 for h,r in ex.map(fetch,['dustinstudio','dgx2','gongqian-mini']):print(h,json.dumps(r))
