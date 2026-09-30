"""Fetch only compact experiment status and final per-table metrics over SSH."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import subprocess

BASE = Path(__file__).parent
ROOTS = {
    'dgx2': '/home/cms/experiments/openml12-finetune-20260927',
    'dustinstudio': '/Users/dustinstudio/experiments/openml12-finetune-20260927',
    'gongqian-mini': '/Users/gongqian/experiments/openml12-finetune-20260927',
}
CODE = '''from pathlib import Path
import json
p=Path(root); d=json.loads((p/'campaign.json').read_text())
d['remote_root']=root
d['active_updates']=0; d['active_successful_seconds']=0.0
if d.get('active_table'):
 for attempt in ('admission','main'):
  f=p/'runs'/d['active_table']/attempt/'updates.jsonl'
  if f.exists():
   with f.open() as handle:
    for line in handle:
     try: row=json.loads(line)
     except json.JSONDecodeError: continue
     d['active_updates']+=1; d['active_successful_seconds']+=row['seconds']
 f=p/'runs'/d['active_table']/'main/terminal.json'
 if f.exists(): d['active_terminal']=json.loads(f.read_text()).get('outcome')
if d['outcome']=='failed':d['log_tail']=(p/'campaign.stdout').read_text()[-1800:]
print(json.dumps(d))
'''


def read(item):
    host, root = item
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3 -'],
                       input='root='+repr(root)+'\n'+CODE, text=True, capture_output=True, timeout=30)
    if p.returncode:
        return host, {'outcome': 'status_read_failed', 'error': p.stderr[-1000:]}
    return host, json.loads(p.stdout)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = dict(pool.map(read, ROOTS.items()))
snapshot = dict(refreshed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), hosts=results)
(BASE/'status.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
lines = ['# OpenML12 单表独立微调', '', '最新远端读取：'+snapshot['refreshed_utc'], '',
         '| 主机 | 状态 | 已完成测试 | 当前表 | 当前有效更新秒数 |',
         '|---|---|---:|---|---:|']
for host, d in results.items():
    lines.append(f"| {host} | {d['outcome']} | {len(d.get('datasets', {}))}/12 | {d.get('active_table')} | {d.get('active_successful_seconds', 0):.1f} |")
    compact = {k: d.get(k) for k in ('outcome','active_table','active_updates',
                                   'active_successful_seconds','macro_r2','error','log_tail')}
    compact['completed'] = {n: round(v['metrics']['r2'], 5) for n,v in d.get('datasets',{}).items()}
    print(host, json.dumps(compact, ensure_ascii=False))
lines += ['', '每表从本机 joint618 父模型重新初始化；每表约120秒成功参数更新；终点测试使用历史固定 bank。',
          '准备与配方见 [PLAN.md](PLAN.md)。逐表测试指标、checkpoint、更新数及实际训练秒数见 [status.json](status.json)。']
(BASE/'CURRENT.md').write_text('\n'.join(lines)+'\n')
