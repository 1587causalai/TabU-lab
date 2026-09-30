import subprocess,time,json,re,os,signal
from pathlib import Path
b=Path.home()/'experiments/v6-window2048-probe-mini-20260928'
with (b/'train.log').open('w') as log:
 p=subprocess.Popen([str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u',str(b/'probe.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 start=time.monotonic()
 with (b/'gpu-samples.jsonl').open('w') as f:
  while p.poll() is None:
   q=subprocess.run(['ioreg','-r','-c','AGXAccelerator','-d','1'],capture_output=True,text=True)
   line=next((l for l in q.stdout.splitlines() if 'PerformanceStatistics' in l),'')
   keys=['Device Utilization %','Renderer Utilization %','Tiler Utilization %','In use system memory','Alloc system memory']
   record={'elapsed':time.monotonic()-start}
   for k in keys:
    m=re.search('"'+re.escape(k)+'"=([0-9]+)',line)
    if m:record[k]=int(m.group(1))
   f.write(json.dumps(record)+'\n');f.flush()
   if time.monotonic()-start>480:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15);break
   time.sleep(2)
 (b/'process.json').write_text(json.dumps({'returncode':p.wait(),'wall_seconds':time.monotonic()-start}))
